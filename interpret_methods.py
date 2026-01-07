import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from lime import lime_image

from captum.attr import (
    LayerGradCam,
    IntegratedGradients,
    Occlusion,
    GradientShap
)

from config import (
    DEVICE,
    GRADCAM_LAYER_NAME, GRADCAM_USE_LAYER3_IF_EXISTS, GRADCAM_CLIP_PCT, GRADCAM_ALPHA,
    IG_STEPS, IG_BASELINE, IG_NOISE_TUNNEL_SAMPLES,
    OCCLUSION_WINDOW_FRAC, OCCLUSION_STRIDE_FRAC,
    SHAP_N_SAMPLES, SHAP_N_BASELINES
)

# ============================================================
# HELPERS (suporte 28x28 e 224x224, 1 ou 3 canais)
# ============================================================

def _ensure_4d(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        return image.unsqueeze(0)
    return image

def _first_conv_in_channels(model: torch.nn.Module) -> int:
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m.in_channels
    return 1

def _adapt_channels_to_model(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:

    x = _ensure_4d(x)
    expected = _first_conv_in_channels(model)
    c = x.shape[1]

    if expected == c:
        return x

    if expected == 3 and c == 1:
        return x.repeat(1, 3, 1, 1)

    if expected == 1 and c == 3:
        return x.mean(dim=1, keepdim=True)

    # fallback: tenta cortar/replicar
    if expected < c:
        return x[:, :expected, :, :]
    else:
        reps = expected // c + 1
        return x.repeat(1, reps, 1, 1)[:, :expected, :, :]

def _to_numpy_map(attr: torch.Tensor) -> np.ndarray:

    attr = attr.detach().cpu()
    if attr.dim() == 4:
        attr = attr[0]
    if attr.dim() == 3:
        # agrega canais por média do abs (mais estável)
        attr = attr.abs().mean(dim=0)
    return attr.numpy()

def _normalize_map(m: np.ndarray, clip_pct=GRADCAM_CLIP_PCT) -> np.ndarray:
    m = m.astype(np.float32)
    lo, hi = np.percentile(m, clip_pct[0]), np.percentile(m, clip_pct[1])
    m = np.clip(m, lo, hi)
    m = m - m.min()
    denom = (m.max() + 1e-8)
    return m / denom

def _find_target_layer(model: torch.nn.Module, layer_name: str | None):

    if layer_name is not None:
        candidates = []
        for name, module in model.named_modules():
            if layer_name in name:
                candidates.append((name, module))
        # escolhe o último candidato que seja Conv2d ou contenha Conv2d
        for name, module in reversed(candidates):
            if isinstance(module, torch.nn.Conv2d):
                return module
            # procura última conv dentro
            last_conv = None
            for m in module.modules():
                if isinstance(m, torch.nn.Conv2d):
                    last_conv = m
            if last_conv is not None:
                return last_conv

    # auto: resnet -> layer3 se existir
    if GRADCAM_USE_LAYER3_IF_EXISTS:
        for nm, md in model.named_modules():
            if nm.endswith("layer3"):
                last_conv = None
                for m in md.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        last_conv = m
                if last_conv is not None:
                    return last_conv

    # fallback: última Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("Não encontrei nenhuma Conv2d no modelo para GradCAM.")
    return last_conv


# ============================================================
# 1) GradCAM (melhor alvo + normalização + suporta 224 e 28)
# ============================================================

def explain_gradcam(model, image, target_class=None):
    model.eval()
    x = image.to(DEVICE)
    x = _adapt_channels_to_model(model, x)

    target_layer = _find_target_layer(model, GRADCAM_LAYER_NAME)
    gradcam = LayerGradCam(model, target_layer)

    if target_class is None:
        with torch.no_grad():
            pred = model(x)
            target_class = int(torch.argmax(pred, dim=1).item())

    attr = gradcam.attribute(x, target=target_class)  # (1,1,h,w) ou (1,c,h,w)
    # Upsample para tamanho original
    attr = F.interpolate(attr, size=x.shape[-2:], mode="bilinear", align_corners=False)

    m = _to_numpy_map(attr)
    m = _normalize_map(m)
    return m


# ============================================================
# 2) Integrated Gradients (mais steps + baseline blur opcional + noise tunnel)
# ============================================================

def _baseline_for_ig(x: torch.Tensor) -> torch.Tensor:
    if IG_BASELINE == "blur":
        # blur simples por avg pool (sem libs extra)
        k = 11 if min(x.shape[-2:]) >= 64 else 5
        pad = k // 2
        return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
    return torch.zeros_like(x)

def explain_integrated_gradients(model, image, target_class=None):
    model.eval()
    x = image.to(DEVICE)
    x = _adapt_channels_to_model(model, x)

    if target_class is None:
        with torch.no_grad():
            pred = model(x)
            target_class = int(torch.argmax(pred, dim=1).item())

    ig = IntegratedGradients(model)
    baseline = _baseline_for_ig(x)

    if IG_NOISE_TUNNEL_SAMPLES and IG_NOISE_TUNNEL_SAMPLES > 0:
        # média de várias IG com ruído pequeno para estabilizar
        attrs = []
        for _ in range(int(IG_NOISE_TUNNEL_SAMPLES)):
            noise = torch.randn_like(x) * 0.02
            xn = x + noise
            attr = ig.attribute(xn, baselines=baseline, target=target_class, n_steps=int(IG_STEPS))

            attrs.append(attr)
        attr = torch.mean(torch.stack(attrs, dim=0), dim=0)
    else:
        attr = ig.attribute(x, baselines=baseline, target=target_class, n_steps=int(IG_STEPS))

    m = _to_numpy_map(attr)
    m = _normalize_map(m)
    return m


# ============================================================
# 3) Occlusion (janela/stride adaptativos ao tamanho)
# ============================================================

def explain_occlusion(model, image, target_class=None):
    model.eval()
    x = image.to(DEVICE)
    x = _adapt_channels_to_model(model, x)

    if target_class is None:
        with torch.no_grad():
            pred = model(x)
            target_class = int(torch.argmax(pred, dim=1).item())

    H, W = x.shape[-2], x.shape[-1]
    win = max(4, int(min(H, W) * float(OCCLUSION_WINDOW_FRAC)))
    stride = max(2, int(min(H, W) * float(OCCLUSION_STRIDE_FRAC)))
    # garante ímpares p/ centrar melhor
    if win % 2 == 0:
        win += 1

    occ = Occlusion(model)
    attr = occ.attribute(
        x,
        target=target_class,
        sliding_window_shapes=(x.shape[1], win, win),
        strides=(x.shape[1], stride, stride),
        baselines=0
    )

    m = _to_numpy_map(attr)
    m = _normalize_map(m)
    return m


# ============================================================
# 2. SHAP
# ============================================================

# [ALTERAR AQUI] — troca a assinatura da função
def explain_shap(model, image, target_class=None, background_batch=None):

    model.eval()

    x = image.to(DEVICE)
    x = _ensure_4d(x)
    x = _adapt_channels_to_model(model, x)

    # target class automático
    if target_class is None:
        with torch.no_grad():
            pred = model(x)
            target_class = int(torch.argmax(pred, dim=1).item())

    # baselines (background)
    if background_batch is None:
        baselines = torch.zeros(
            (max(1, int(SHAP_N_BASELINES)),) + tuple(x.shape[1:]),
            device=DEVICE
        )
    else:
        b = background_batch.to(DEVICE)
        b = _ensure_4d(b)
        b = _adapt_channels_to_model(model, b)
        baselines = b[:int(SHAP_N_BASELINES)]

        # se vierem menos do que SHAP_N_BASELINES, garante pelo menos 1
        if baselines.shape[0] == 0:
            baselines = torch.zeros(
                (1,) + tuple(x.shape[1:]),
                device=DEVICE
            )

    gs = GradientShap(model)
    attr = gs.attribute(
        x,
        baselines=baselines,
        n_samples=int(SHAP_N_SAMPLES),
        target=target_class
    )

    m = _to_numpy_map(attr)
    m = _normalize_map(m, clip_pct=(0, 100))
    return m



# ============================================================
# 5) LIME (corrige canais + predict_fn robusto)
# ============================================================

def explain_lime(model, image, target_class=None):
    model.eval()
    x = image.to(DEVICE)
    x = _ensure_4d(x)

    # LIME quer HWC com 3 canais
    x_for_lime = x[0].detach().cpu()
    if x_for_lime.shape[0] == 1:
        x_for_lime = x_for_lime.repeat(3, 1, 1)
    img_np = x_for_lime.permute(1, 2, 0).numpy()  # HWC

    def predict_fn(images):
        images = torch.tensor(images).permute(0, 3, 1, 2).float().to(DEVICE)
        images = _adapt_channels_to_model(model, images)
        with torch.no_grad():
            out = model(images)
            return torch.softmax(out, dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    label = explanation.top_labels[0]
    mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )[1]

    mask = mask.astype(np.float32)
    mask = _normalize_map(mask, clip_pct=(0, 100))
    return mask
