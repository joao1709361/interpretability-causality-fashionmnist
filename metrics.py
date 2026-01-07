# ============================================================
# IMPLEMENTAÇÃO DAS MÉTRICAS: AVG-SENSITIVITY E COMPLEXITY
# ============================================================

import torch
import numpy as np
from config import RADIUS, N_PERTURBATIONS, DEVICE


# ============================================================
# 1. AVG-SENSITIVITY (Robustez)
# ============================================================
def avg_sensitivity(model, explain_func, image, radius=RADIUS, n_perturb=N_PERTURBATIONS):
    """
    Mede a sensibilidade média (robustez) das explicações geradas por um método.
    Quanto menor, mais robusta é a explicação.
    """
    model.eval()

    def normalize_expl(e):
        e = np.abs(e).astype(np.float32)
        s = float(np.sum(e))
        if s <= 1e-12:
            return e
        return e / (s + 1e-8)

    # Explicação original
    base_expl = explain_func(model, image).flatten()
    base_expl = normalize_expl(base_expl)

    diffs = []
    for _ in range(n_perturb):
        # Perturbação gaussiana pequena
        noise = torch.randn_like(image) * radius
        perturbed = image + noise.to(DEVICE)

        # Explicação perturbada
        pert_expl = explain_func(model, perturbed).flatten()
        pert_expl = normalize_expl(pert_expl)

        # Distância L2 entre distribuições
        diff = np.linalg.norm(base_expl - pert_expl, ord=2)
        diffs.append(diff)

    return float(np.mean(diffs))



# ============================================================
# 2. COMPLEXITY (Simplicidade / Clareza)
# ============================================================
def complexity(explanation):
    """
    Mede a complexidade da explicação (entropia da distribuição de importâncias).

    Args:
        explanation: mapa de importância (numpy array)

    Returns:
        float: entropia da explicação (quanto menor, mais simples)
    """
    # Converter para valores positivos
    expl = np.abs(explanation.flatten())
    total = np.sum(expl)

    if total == 0:
        return 0.0

    # Normalizar (probabilidades das features)
    p = expl / total
    p = p[p > 0]  # evitar log(0)

    # Entropia (Shannon)
    entropy = -np.sum(p * np.log(p + 1e-10))
    return float(entropy)


# ============================================================
# Função auxiliar: avaliar método
# ============================================================
def evaluate_metric(model, explain_func, images):
    """
    Aplica ambas as métricas a uma lista de imagens.
    Retorna média das métricas.
    """
    avg_sens_vals = []
    comp_vals = []

    for image in images:
        image = image.to(DEVICE).unsqueeze(0)
        expl = explain_func(model, image)
        avg_sens_vals.append(avg_sensitivity(model, explain_func, image))
        comp_vals.append(complexity(expl))

    return {
        "Avg-Sensitivity": np.mean(avg_sens_vals),
        "Complexity": np.mean(comp_vals)
    }
