# ============================================================
# SCRIPT PRINCIPAL DO TRABALHO DE INTERPRETABILIDADE
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODE, EXPERIMENT_NAME
from data_loader import get_data_loaders
from model import get_model
from evaluate import run_evaluation
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from evaluate import save_explanations

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.get_lock().locks = []

import plots

from pretrained_model import get_pretrained_model
from data_loader_pretrained import get_data_loaders_resnet

from config import RESULTS_SIMPLE, RESULTS_PRETRAINED
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt


# ============================================================
# 1. Função de treino
# ============================================================
import numpy as np

def train_model(model, train_loader, val_loader, test_loader, SAVE_PATH, num_epochs=NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    if isinstance(model, torch.nn.Module) and model.__class__.__name__ == "PretrainedResNet18":
        lr = 0.0001
    else:
        lr = LEARNING_RATE




    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6
    )

    print("\n Iniciando treino com Early Stopping...\n")

    best_loss = np.inf
    patience = 15
    trigger_times = 0
    best_state = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Época {epoch + 1}/{num_epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        train_acc = 100.0 * train_correct / max(1, train_total)


        # --- validação
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / max(1, val_total)

        # Atualizar LR com base na val_loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoca [{epoch + 1}/{num_epochs}] "
            f"- Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% "
            f"- Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% "
            f"- LR: {current_lr:.6f}"
        )

        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # --- Early Stopping
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            trigger_times = 0
            best_state = model.state_dict()
        else:
            trigger_times += 1
            print(f"  Sem melhoria val ({trigger_times}/{patience})")
            if trigger_times >= patience:
                print(f"\n Early stopping na época {epoch + 1} (melhor val_loss: {best_loss:.4f})")
                model.load_state_dict(best_state)
                break


    # ------------------------------------------------------------
    # Guardar curvas de treino (loss e accuracy)
    # ------------------------------------------------------------
    plots_dir = os.path.join(os.path.dirname(SAVE_PATH), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    epochs_range = range(1, len(train_losses) + 1)

    # Curva de loss
    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "curvas_loss.png"), dpi=150)
    plt.close()

    # Curva de accuracy
    plt.figure()
    plt.plot(epochs_range, train_accs, label="Train Accuracy")
    plt.plot(epochs_range, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "curvas_accuracy.png"), dpi=150)
    plt.close()

    print(f" Curvas de treino guardadas em: {plots_dir}")



    # Avaliação no conjunto de teste
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n Acurácia no conjunto de teste: {accuracy:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), SAVE_PATH)
    print(f" Modelo guardado em: {SAVE_PATH}")
    return model

# ============================================================
# 2. Execução principal
# ============================================================
def main():

    start_time = time.time()

    # Pasta base por modo
    BASE_DIR_MODE = RESULTS_SIMPLE if MODE == "simple" else RESULTS_PRETRAINED

    # Pasta da experiência (não sobrepõe)
    SAVE_DIR = os.path.join(BASE_DIR_MODE, EXPERIMENT_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Paths desta experiência
    SAVE_PATH = os.path.join(SAVE_DIR, "trained_model.pth")
    METRICS_PATH = os.path.join(SAVE_DIR, "metrics_results.csv")
    PLOTS_DIR_EXP = os.path.join(SAVE_DIR, "plots")
    EXPL_DIR_EXP = os.path.join(SAVE_DIR, "explanations")
    os.makedirs(PLOTS_DIR_EXP, exist_ok=True)
    os.makedirs(EXPL_DIR_EXP, exist_ok=True)


    if MODE == "simple":
        model = get_model()
        train_loader, val_loader, test_loader = get_data_loaders()
    elif MODE == "pretrained":
        model = get_pretrained_model(freeze_backbone=False)
        train_loader, val_loader, test_loader = get_data_loaders_resnet()
    else:
        raise ValueError("Modo inválido! Usa 'simple' ou 'pretrained'.")

    # --- carregar modelo se existir; se não existir, treinar ---
    if os.path.isfile(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        print(f"\n Modelo existente carregado de: {SAVE_PATH}")
        model.eval()
    else:
        model = train_model(model, train_loader, val_loader, test_loader, SAVE_PATH)





    # Avaliar interpretabilidade
    print("\n Iniciando avaliação de interpretabilidade...\n")
    run_evaluation(sample_size=100, SAVE_PATH=SAVE_PATH, MODE=MODE, metrics_csv_path=METRICS_PATH)

    # Gera e guarda explicações visuais para 5 imagens
    print("\n🖼  Gerando explicações visuais...")

    if MODE == "simple":
        _, _, test_loader_vis = get_data_loaders()
    elif MODE == "pretrained":
        _, _, test_loader_vis = get_data_loaders_resnet()


    images, labels = next(iter(test_loader_vis))
    #save_explanations(model, images, labels, max_samples=5, save_dir=SAVE_DIR)
    save_explanations(model, test_loader_vis, max_samples=10, save_dir=EXPL_DIR_EXP)

    # no final do main.py, depois da avaliação e do CSV gravado
    try:
        #plots.main()
        plots.main(csv_path=METRICS_PATH, out_dir=PLOTS_DIR_EXP)

    except Exception as e:
        print(f"[plots] falhou ao gerar plots: {e}")

    #Saber quanto tempo o codigo demora a correr
    end_time = time.time()
    total_time = end_time - start_time

    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"\n Tempo total de execução: {minutes} min {seconds} s")




# ============================================================
# 3. Ponto de entrada
# ============================================================
if __name__ == "__main__":
    main()
