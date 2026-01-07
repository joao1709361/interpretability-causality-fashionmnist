# ============================================================
# CONFIGURAÇÕES GLOBAIS DO TRABALHO DE INTERPRETABILIDADE
# ============================================================

import torch
import os
import random
import numpy as np


MODE = "pretrained"  # "simple" ou "pretrained"
EXPERIMENT_NAME = "experimento_resnet224_v3_com_melhorias"

# ------------------------------
# 1. Reprodutibilidade
# ------------------------------
def set_seed(seed: int = 42):
    """Garante resultados reprodutíveis em todas as execuções."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# 2. Caminhos das pastas
# ------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
EXPLANATIONS_DIR = os.path.join(RESULTS_DIR, "explanations")

for folder in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, EXPLANATIONS_DIR]:
    os.makedirs(folder, exist_ok=True)

# ------------------------------
# 3. Parâmetros do modelo
# ------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
NUM_CLASSES = 10       # FashionMNIST
IMAGE_SIZE = 28        # 28x28 (FashionMNIST)


# Otimizador e regularização (extra)
WEIGHT_DECAY = 1e-4     # penaliza pesos grandes - menos overfitting
DROPOUT_RATE = 0.3       # usado dentro da CNN para regularizar

# ------------------------------
# 4. Parâmetros das métricas
# ------------------------------
# Usados na Avg-Sensitivity
RADIUS = 0.02           # raio de perturbação (percentagem)
N_PERTURBATIONS = 10    # número de amostras perturbadas por imagem

# ------------------------------
# 5. Dispositivo (GPU/CPU)
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 6. Outros parâmetros
# ------------------------------
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics_results.csv")

# Subpastas para separar CNN simples e modelo pré-treinado
RESULTS_SIMPLE = os.path.join(RESULTS_DIR, "simple")
RESULTS_PRETRAINED = os.path.join(RESULTS_DIR, "pretrained")

for folder in [RESULTS_SIMPLE, RESULTS_PRETRAINED]:
    os.makedirs(folder, exist_ok=True)



# ============================================================
# INTERPRETABILIDADE - MELHORIAS (qualidade/estabilidade)
# ============================================================

# GradCAM
GRADCAM_LAYER_NAME = None
GRADCAM_USE_LAYER3_IF_EXISTS = True
GRADCAM_CLIP_PCT = (2, 98)       # percentis para cortar extremos antes de normalizar
GRADCAM_ALPHA = 0.45             # alpha do overlay

# Integrated Gradients
IG_STEPS = 200                   # mais steps = mapa mais suave/estável
IG_BASELINE = "zeros"            # "zeros" ou "blur"
IG_NOISE_TUNNEL_SAMPLES = 0      # 0 desliga; ex: 8 melhora estabilidade

# Occlusion
OCCLUSION_WINDOW_FRAC = 0.07     # fração do H/W para janela (ex: 0.07 -> ~16 em 224)
OCCLUSION_STRIDE_FRAC = 0.035    # fração do H/W para stride

# SHAP
SHAP_METHOD = "captum_gradient"  # "captum_gradient"
SHAP_N_SAMPLES = 32
SHAP_N_BASELINES = 8            # numero de imagens no background


# Visualização dos heatmaps
HEATMAP_TOP_PCT = 90          # mostra só top 10% do mapa (ajusta 85–95)
HEATMAP_BLUR_KERNEL = 5       # blur leve para reduzir ruído (5 ou 7)
HEATMAP_ALPHA = 0.55          # alpha do overlay depois do threshold


