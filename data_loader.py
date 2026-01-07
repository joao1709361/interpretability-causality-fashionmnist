# ============================================================
# CARREGAMENTO E PREPARAÇÃO DO DATASET (FashionMNIST)
# ============================================================

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE, DATA_DIR
from torch.utils.data import random_split

# Estatísticas aproximadas do FashionMNIST
FASHION_MEAN = (0.2860,)
FASHION_STD = (0.3530,)


def get_data_loaders(batch_size: int = BATCH_SIZE, val_split=0.1):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MEAN, FASHION_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MEAN, FASHION_STD),
    ])

    full_train_aug = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    full_train_clean = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=test_transform
    )

    val_size = int(len(full_train_aug) * val_split)
    train_size = len(full_train_aug) - val_size

    indices = torch.randperm(len(full_train_aug)).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    from torch.utils.data import Subset
    train_dataset = Subset(full_train_aug, train_idx)
    val_dataset = Subset(full_train_clean, val_idx)


    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Execução de teste rápida (opcional)
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"Número de batches no treino: {len(train_loader)}")
    print(f"Número de batches na validação: {len(val_loader)}")
    print(f"Número de batches no teste: {len(test_loader)}")

