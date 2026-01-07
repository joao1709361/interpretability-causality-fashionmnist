# ============================================================
# plots.py — Geração de gráficos e análise visual das métricas
# ============================================================

import os

# --- HOTFIX Windows: conflito OpenMP (libomp.dll vs libiomp5md.dll) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR, PLOTS_DIR, METRICS_CSV, RESULTS_SIMPLE, RESULTS_PRETRAINED, MODE, EXPERIMENT_NAME


# ============================================================
# Funções auxiliares
# ============================================================
def _safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_")


# ============================================================
# Gráfico de barras (linear ou log)
# ============================================================
def plot_bar(df: pd.DataFrame, metric: str, fname: str, out_dir: str, logy: bool = False):
    df_ord = df.sort_values(metric, ascending=False).copy()
    x = df_ord["Method"].tolist()
    y = df_ord[metric].tolist()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, y, color="cornflowerblue", edgecolor="black")
    plt.title(f"{metric} por Método" + (" (escala log10)" if logy else ""))
    plt.ylabel(metric)
    plt.xlabel("Método")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if logy:
        ymin = min(v for v in y if v > 0) if any(v > 0 for v in y) else 1e-8
        plt.yscale("log")
        plt.ylim(bottom=ymin * 0.8)
        for b, v in zip(bars, y):
            txt = f"{v:.2e}" if v != 0 else "0"
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), txt,
                     ha="center", va="bottom", fontsize=8, rotation=90)
    else:
        for b, v in zip(bars, y):
            txt = f"{v:.3g}"
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), txt,
                     ha="center", va="bottom", fontsize=8, rotation=90)

    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plots] gravado: {out_path}")


# ============================================================
# Tabela de ranking
# ============================================================
def plot_rank_table(df: pd.DataFrame, fname: str, out_dir):
    metrics = [c for c in df.columns if c not in ("Method")]
    ranks = {}
    for m in metrics:
        #asc = True if m.lower().startswith("complex") else False
        # métricas onde MENOR é melhor
        lower_is_better = {"Complexity", "Avg-Sensitivity"}
        asc = True if m in lower_is_better else False

        ranks[m] = df[m].rank(ascending=asc, method="min").astype(int)

    rank_df = pd.DataFrame(ranks)
    rank_df.insert(0, "Method", df["Method"])

    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.45 * len(rank_df)))
    ax.axis("off")
    tbl = ax.table(cellText=rank_df.values,
                   colLabels=rank_df.columns,
                   loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plots] gravado: {out_path}")


# ============================================================
# Novo gráfico: correlação entre métricas
# ============================================================
def plot_correlation(df: pd.DataFrame, out_dir):
    plt.figure(figsize=(5, 5))
    plt.scatter(df["Complexity"], df["Avg-Sensitivity"], s=100, color="mediumseagreen", edgecolor="black")
    for i, method in enumerate(df["Method"]):
        plt.text(df["Complexity"][i] * 1.02, df["Avg-Sensitivity"][i] * 0.98, method, fontsize=8)
    plt.xlabel("Complexity")
    plt.ylabel("Avg-Sensitivity")
    plt.title("Relação entre Sensibilidade e Complexidade")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "correlation_plot.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plots] gravado: {out_path}")



def plot_radar(df: pd.DataFrame, out_dir):

    labels = df["Method"].tolist()

    # --- Normalizações (0 = melhor, 1 = pior) ---
    sens = df["Avg-Sensitivity"].astype(float).values
    comp = df["Complexity"].astype(float).values

    # Evitar divisão por zero caso max == min
    sens_den = (sens.max() - sens.min()) if (sens.max() - sens.min()) != 0 else 1.0
    comp_den = (comp.max() - comp.min()) if (comp.max() - comp.min()) != 0 else 1.0

    norm_sens = (sens - sens.min()) / sens_den
    norm_comp = (comp - comp.min()) / comp_den

    # Distância ao ideal (0,0) no espaço normalizado
    tradeoff = np.sqrt(norm_sens**2 + norm_comp**2)
    trade_den = (tradeoff.max() - tradeoff.min()) if (tradeoff.max() - tradeoff.min()) != 0 else 1.0
    norm_trade = (tradeoff - tradeoff.min()) / trade_den

    # --- Scores (1 = melhor, 0 = pior) ---
    robustness_score = 1.0 - norm_sens
    simplicity_score = 1.0 - norm_comp
    balance_score = 1.0 - norm_trade

    radar_df = pd.DataFrame({
        "Method": labels,
        "Robustez (↑ melhor)": robustness_score,
        "Simplicidade (↑ melhor)": simplicity_score,
        "Equilíbrio (↑ melhor)": balance_score,
    })

    metrics = ["Robustez (↑ melhor)", "Simplicidade (↑ melhor)", "Equilíbrio (↑ melhor)"]
    data = radar_df[metrics].values

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for i in range(len(labels)):
        values = data[i].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=labels[i])
        ax.fill(angles, values, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Escala 0..1 porque agora é score
    ax.set_ylim(0, 1)

    plt.title("Comparação dos Métodos — Radar de Scores (maior = melhor)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "radar_plot.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plots] gravado: {out_path}")


# ============================================================
# Resumo automático dos melhores métodos
# ============================================================
def print_summary(df: pd.DataFrame):
    # Avg-Sensitivity: quanto MENOR, mais robusto
    best_robust = df.loc[df["Avg-Sensitivity"].idxmin(), "Method"]

    # Complexity: quanto MENOR, mais simples
    best_simple = df.loc[df["Complexity"].idxmin(), "Method"]


    # Normalizar métricas (0 = melhor, 1 = pior)
    norm_sens = (df["Avg-Sensitivity"] - df["Avg-Sensitivity"].min()) / (
                df["Avg-Sensitivity"].max() - df["Avg-Sensitivity"].min())
    norm_comp = (df["Complexity"] - df["Complexity"].min()) / (df["Complexity"].max() - df["Complexity"].min())

    # Score de equilíbrio: distância ao ponto ideal (0,0)
    tradeoff_score = np.sqrt(norm_sens ** 2 + norm_comp ** 2)
    best_balance = df.loc[tradeoff_score.idxmin(), "Method"]

    print("\n Resumo Automático dos Resultados:")
    print(f"• Mais robusto (menor Avg-Sensitivity): {best_robust}")
    print(f"• Mais simples (menor Complexity): {best_simple}")
    print(f"• Melhor equilíbrio robustez/simplicidade: {best_balance}\n")



# ============================================================
# Execução principal
# ============================================================
def main(csv_path=None, out_dir=None):
    if csv_path is None:
        csv_path = METRICS_CSV
    if out_dir is None:
        out_dir = PLOTS_DIR

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Não encontrei o CSV de métricas em: {csv_path}")

    df = pd.read_csv(csv_path)


    # 1) Barplots (linear e log)
    plot_bar(df, "Avg-Sensitivity", "bar_avg_sensitivity_linear.png", out_dir, logy=False)
    plot_bar(df, "Complexity", "bar_complexity_linear.png", out_dir, logy=False)
    plot_bar(df, "Avg-Sensitivity", "bar_avg_sensitivity_log.png", out_dir, logy=True)
    plot_rank_table(df, "ranking_methods.png", out_dir)
    plot_correlation(df, out_dir)
    plot_radar(df, out_dir)

    # 5) Resumo automático
    print_summary(df)


# ============================================================
# Ponto de entrada
# ============================================================

if __name__ == "__main__":

    # Por defeito, usar a mesma lógica do main.py (experiência ativa)
    base_dir_mode = RESULTS_SIMPLE if MODE == "simple" else RESULTS_PRETRAINED
    exp_dir = os.path.join(base_dir_mode, EXPERIMENT_NAME)

    default_csv = os.path.join(exp_dir, "metrics_results.csv")
    default_out = os.path.join(exp_dir, "plots")

    # Se o CSV da experiência não existir, cai para o default global do config
    csv_path = default_csv if os.path.exists(default_csv) else METRICS_CSV
    out_dir = default_out if os.path.exists(exp_dir) else PLOTS_DIR

    print(f"[plots] A usar CSV: {csv_path}")
    print(f"[plots] A guardar em: {out_dir}")

    main(csv_path=csv_path, out_dir=out_dir)
