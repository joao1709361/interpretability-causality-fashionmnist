# ============================================================
# APLICAÇÃO DAS MÉTRICAS AOS MÉTODOS DE INTERPRETABILIDADE
# ============================================================

import torch
import pandas as pd
from tqdm import tqdm

from config import DEVICE, METRICS_CSV
from data_loader import get_data_loaders
from model import get_model
import matplotlib.pyplot as plt

import numpy as np
import os
from config import EXPLANATIONS_DIR
from interpret_methods import (
    explain_lime,
    explain_shap,
    explain_gradcam,
    explain_integrated_gradients,
    explain_occlusion
)

from metrics import avg_sensitivity, complexity
from config import SHAP_N_BASELINES

from config import HEATMAP_TOP_PCT, HEATMAP_BLUR_KERNEL, HEATMAP_ALPHA
import numpy.ma as ma
import torch.nn.functional as F
import time


# ============================================================
# 1. Função principal de avaliação
# ============================================================
def evaluate_all_methods(model, image, background_batch=None):
    methods = {
        "GradCAM": lambda m, x: explain_gradcam(m, x),
        "IntegratedGradients": lambda m, x: explain_integrated_gradients(m, x),
        "Occlusion": lambda m, x: explain_occlusion(m, x),
        "SHAP": lambda m, x: explain_shap(m, x, background_batch=background_batch),
        "LIME": lambda m, x: explain_lime(m, x),
    }

    results = []
    for method_name, func in methods.items():
        expl = func(model, image)
        avg_sens = avg_sensitivity(model, func, image)
        comp = complexity(expl)

        results.append({
            "Method": method_name,
            "Avg-Sensitivity": avg_sens,
            "Complexity": comp
        })

    return results



# ============================================================
# 2. Execução completa (para chamar no main)
# ============================================================
def run_evaluation(sample_size=30, SAVE_PATH=None, MODE='pretrained', metrics_csv_path=None):


    if MODE == "simple":
        from model import get_model
        model = get_model()
    else:
        from pretrained_model import get_pretrained_model
        model = get_pretrained_model(freeze_backbone=False)

    if SAVE_PATH is None:
        raise ValueError("SAVE_PATH não foi fornecido. Passa o caminho do modelo treinado (SAVE_PATH).")

    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    if MODE == "simple":
        from data_loader import get_data_loaders
        _, _, test_loader = get_data_loaders()
    else:
        from data_loader_pretrained import get_data_loaders_resnet
        _, _, test_loader = get_data_loaders_resnet()

    images_list = []
    background_list = []

    n_baselines = int(SHAP_N_BASELINES)

    for batch_images, _ in test_loader:
        for img in batch_images:
            if len(background_list) < n_baselines:
                background_list.append(img)
            if len(images_list) < sample_size:
                images_list.append(img)

            if len(images_list) >= sample_size and len(background_list) >= n_baselines:
                break

        if len(images_list) >= sample_size and len(background_list) >= n_baselines:
            break

    if len(images_list) == 0:
        raise RuntimeError("Não foi possível recolher imagens do test_loader.")

    images = torch.stack(images_list, dim=0)
    background_batch = torch.stack(background_list, dim=0) if len(background_list) > 0 else None
    if background_batch is not None:
        background_batch = background_batch.to(DEVICE)

    all_rows = []
    t0 = time.time()
    for i in range(len(images)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ti = time.time()

        img_tensor = images[i].unsqueeze(0).to(DEVICE)
        rows = evaluate_all_methods(model, img_tensor, background_batch=background_batch)
        all_rows.extend(rows)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - ti
        done = i + 1
        total = len(images)
        avg = (time.time() - t0) / done
        remaining = avg * (total - done)
        print(f"[timing] {done}/{total} | {dt:.2f}s esta imagem | ~{avg:.2f}s/img em média | ETA ~ {remaining/60:.1f} min" )

    df_all = pd.DataFrame(all_rows)

    df_results = df_all.groupby("Method", as_index=False)[["Avg-Sensitivity", "Complexity"]].mean()

    if metrics_csv_path is None:
        metrics_csv_path = METRICS_CSV

    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
    df_results.to_csv(metrics_csv_path, index=False)

    print("\nResultados guardados em:", metrics_csv_path)
    print(df_results)

    return df_results



# ============================================================
#  EXTRA: GERA E GUARDA EXPLICAÇÕES VISUAIS (HEATMAPS)
# ============================================================

def save_explanations(model, loader, methods=None, max_samples=5, save_dir=None):


    if methods is None:
        methods = ["LIME", "SHAP", "IntegratedGradients", "Occlusion", "GradCAM"]

    if save_dir is None:
        save_dir = EXPLANATIONS_DIR

    os.makedirs(save_dir, exist_ok=True)

    # Background batch para SHAP
    background_imgs = []
    n_baselines = int(SHAP_N_BASELINES)

    for batch_images, _ in loader:
        for img in batch_images:
            background_imgs.append(img)
            if len(background_imgs) >= n_baselines:
                break

        if len(background_imgs) >= n_baselines:
            break

    background_batch = torch.stack(background_imgs, dim=0).to(DEVICE) if len(background_imgs) > 0 else None

    sample_imgs = []
    sample_labels = []
    for batch_images, batch_labels in loader:
        for img, lab in zip(batch_images, batch_labels):
            sample_imgs.append(img)
            sample_labels.append(int(lab))
            if len(sample_imgs) >= max_samples:
                break
        if len(sample_imgs) >= max_samples:
            break

    for i in range(len(sample_imgs)):
        try:
            img_tensor = sample_imgs[i].unsqueeze(0).to(DEVICE)
            label = sample_labels[i]

            # Para visualização do “input”
            img_vis = sample_imgs[i].detach().cpu()

            # Desnormalizar só para visualização (FashionMNIST)
            mean = 0.2860
            std = 0.3530
            img_vis = img_vis * std + mean

            # Garantir intervalo [0,1] apenas para mostrar no plot
            img_vis = torch.clamp(img_vis, 0.0, 1.0)

            # CHW -> HWC (ou HW se 1 canal)
            if img_vis.dim() == 3 and img_vis.shape[0] in (1, 3):
                if img_vis.shape[0] == 1:
                    img_vis = img_vis[0]
                else:
                    img_vis = img_vis.permute(1, 2, 0)

            for method in methods:
                try:
                    if method == "LIME":
                        explanation = explain_lime(model, img_tensor)
                    elif method == "SHAP":
                        explanation = explain_shap(model, img_tensor, background_batch=background_batch)
                    elif method == "IntegratedGradients":
                        explanation = explain_integrated_gradients(model, img_tensor)
                    elif method == "Occlusion":
                        explanation = explain_occlusion(model, img_tensor)
                    elif method == "GradCAM":
                        explanation = explain_gradcam(model, img_tensor)
                    else:
                        continue

                    # --- normaliza 0..1 ---
                    exp_norm = explanation.astype(np.float32)
                    exp_norm = exp_norm - float(np.min(exp_norm))
                    exp_norm = exp_norm / (float(np.max(exp_norm)) + 1e-8)

                    # --- blur leve (reduz “salpicos” e espalhamento) ---
                    k = int(HEATMAP_BLUR_KERNEL)
                    if k >= 3 and k % 2 == 1:
                        t = torch.tensor(exp_norm)[None, None, ...]  # (1,1,H,W)
                        pad = k // 2
                        t = F.avg_pool2d(t, kernel_size=k, stride=1, padding=pad)
                        exp_norm = t[0, 0].cpu().numpy()

                    # --- threshold por percentil (só mostra zonas mais fortes) ---
                    thr = np.percentile(exp_norm, float(HEATMAP_TOP_PCT))
                    exp_thr = exp_norm.copy()
                    exp_thr[exp_thr < thr] = np.nan
                    exp_thr = ma.masked_invalid(exp_thr)

                    plt.figure(figsize=(5, 5))
                    plt.imshow(img_vis, cmap="gray")
                    plt.imshow(exp_thr, cmap="jet", alpha=float(HEATMAP_ALPHA))
                    plt.axis("off")
                    plt.title(f"{method} — Label {label}")

                    out_path = os.path.join(save_dir, f"{method.lower()}_{i}.png")
                    plt.tight_layout()
                    plt.savefig(out_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"[explanation] guardado: {out_path}")

                except Exception as e:
                    print(f"[explanation] falhou {method} na amostra {i}: {e}")

        except Exception as e:
            print(f"[explanation] falhou na amostra {i}: {e}")





if __name__ == "__main__":
    # Ajusta estes 3 valores
    MODE = "pretrained"  # ou "simple"
    SAVE_PATH = r"C:\UBI\3Ano-1Semestre\Interpretabilidade e Causalidade\Projeto Pratico\results\pretrained\experimento_resnet224_v3_com_melhorias\trained_model.pth"
    METRICS_OUT = r"C:\UBI\3Ano-1Semestre\Interpretabilidade e Causalidade\Projeto Pratico\results\pretrained\experimento_resnet224_v3_com_melhorias\metrics_results.csv"

    run_evaluation(sample_size=30, SAVE_PATH=SAVE_PATH, MODE=MODE, metrics_csv_path=METRICS_OUT)

