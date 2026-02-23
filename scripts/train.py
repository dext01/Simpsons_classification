import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import get_resnet18_finetune
from src.utils import train_model, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    print(f"⚙️ Config: seed={args.seed}, epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(args.data_path, args.batch_size)
    print(f"📊 Number of classes: {len(classes)}")
    print(f"📦 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = get_resnet18_finetune(num_classes=len(classes)).to(device)

    train_model(model, train_loader, val_loader, device, args.epochs, args.lr, args.artifact_dir)


if __name__ == "__main__":
    main()