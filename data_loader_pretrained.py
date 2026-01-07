# data_loader_pretrained.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE, DATA_DIR

# Estatísticas aproximadas do FashionMNIST
FASHION_MEAN = (0.2860,)
FASHION_STD = (0.3530,)

def get_data_loaders_resnet(batch_size: int = BATCH_SIZE, val_split: float = 0.1):

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MEAN, FASHION_STD),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value=0.0
        ),
    ])

    clean_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MEAN, FASHION_STD),
    ])

    # 1) Dataset de treino com augmentation
    full_train_aug = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    # 2) Dataset "limpo" (mesmas imagens) sem augmentation, para validação
    full_train_clean = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=clean_transform
    )

    # 3) Split por índices (para garantir que train e val usam os mesmos exemplos)
    val_size = int(len(full_train_aug) * val_split)
    train_size = len(full_train_aug) - val_size

    indices = torch.randperm(len(full_train_aug)).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    from torch.utils.data import Subset
    train_dataset = Subset(full_train_aug, train_idx)
    val_dataset = Subset(full_train_clean, val_idx)

    # 4) Test dataset (sempre clean)
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=clean_transform
    )

    # 5) Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
