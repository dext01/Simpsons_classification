import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

def get_transforms(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(data_path, batch_size=32, val_split=0.2, seed=42):
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

class KaggleTestDataset(Dataset):
    def __init__(self, main_data_dir, test_dir, transform=None):
        self.main_data_dir = main_data_dir
        self.test_dir = test_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(main_data_dir) if os.path.isdir(os.path.join(main_data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for filename in os.listdir(test_dir):
            if filename.lower().endswith(('.jpg', '.png')):
                name = filename.split('.')[0]
                if '_on_' in name:
                    character = name.split('_on_')[0]
                else:
                    parts = name.split('_')
                    if len(parts) >= 2 and parts[-1].isdigit():
                        character = '_'.join(parts[:-1])
                    else:
                        character = name
                if character in self.class_to_idx:
                    img_path = os.path.join(test_dir, filename)
                    label = self.class_to_idx[character]
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
