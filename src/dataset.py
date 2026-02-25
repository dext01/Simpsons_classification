import torch
import random

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_dataloaders(data_path, batch_size=32, val_split=0.2, seed=42):  # ← добавь seed
    transform = get_transforms()
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=generator
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, dataset.classes
