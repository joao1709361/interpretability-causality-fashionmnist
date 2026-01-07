# ============================================================
# DEFINIÇÃO DO MODELO DE CLASSIFICAÇÃO (CNN PARA FashionMNIST)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, NUM_CLASSES, DROPOUT_RATE


class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate: float = DROPOUT_RATE):
        super().__init__()

        # 28x28 -> 14x14
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 14x14 -> 7x7
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 7x7 -> 7x7 (feature map mais rico)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 7 * 7, 512) #256
        self.fc2 = nn.Linear(512, NUM_CLASSES) #256

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)          # última conv para GradCAM (já usada no interpret_methods.py)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model():
    model = SimpleCNN().to(DEVICE)
    return model


# Execução de teste rápida
if __name__ == "__main__":
    model = get_model()
    sample_input = torch.randn(1, 1, 28, 28).to(DEVICE)
    output = model(sample_input)
    print("Output shape:", output.shape)
