# ============================================================
# MODELO PRÉ-TREINADO — ResNet18 adaptada ao FashionMNIST
# ============================================================

import torch
import torch.nn as nn
from torchvision import models
from config import DEVICE, NUM_CLASSES


class PretrainedResNet18(nn.Module):
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()

        # Carregar ResNet18 pré-treinada em ImageNet
        try:
            # API mais recente (PyTorch >= 1.13 / 2.x)
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            # Fallback para versões mais antigas
            self.backbone = models.resnet18(pretrained=True)

        # Guardar a conv original (3 canais)
        old_conv = self.backbone.conv1

        # Substituir por uma conv com 1 canal de entrada
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Inicializar os pesos da nova conv1 a partir da média dos 3 canais originais
        with torch.no_grad():
            self.backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        # Substituir a fully connected final para ter NUM_CLASSES saídas
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, NUM_CLASSES)

        # Opcional: congelar o backbone e treinar só a FC
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


def get_pretrained_model(freeze_backbone: bool = False):
    model = PretrainedResNet18(freeze_backbone=freeze_backbone).to(DEVICE)
    return model


# Execução de teste rápida
if __name__ == "__main__":
    model = get_pretrained_model()
    x = torch.randn(1, 1, 224, 224).to(DEVICE)
    y = model(x)
    print("Output shape:", y.shape)
