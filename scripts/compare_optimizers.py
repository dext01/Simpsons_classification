import sys
import os
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import get_resnet18_finetune
from src.utils import train_model, seed_everything


def main():
    DATA_PATH = "./data"
    EPOCHS = 20
    BATCH_SIZE = 32
    SEED = 42
    SAVE_DIR = "artifacts/optimizers_comparison"

    OPTIMIZER_CONFIGS = {
        "adam": {"lr": 1e-4},
        "sgd": {"lr": 1e-2},
        "sgd_momentum": {"lr": 1e-2}
    }

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(DATA_PATH, BATCH_SIZE, seed=SEED)
    num_classes = len(classes)
    print(f"📊 Number of classes: {num_classes}")

    histories = {}

    for opt_name, config in OPTIMIZER_CONFIGS.items():
        print(f"\n{'=' * 50}")
        print(f"Training with {opt_name.upper()} (lr={config['lr']})")
        print(f"{'=' * 50}")

        model = get_resnet18_finetune(num_classes=num_classes).to(device)

        history = train_model(
            model, train_loader, val_loader, device,
            epochs=EPOCHS, lr=config["lr"], optimizer_name=opt_name,
            save_dir=SAVE_DIR
        )
        histories[opt_name] = history

    plot_comparison(histories, SAVE_DIR)
    print(f"\n✅ All training completed! Results saved to {SAVE_DIR}")


def plot_comparison(histories, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # График потерь
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for opt_name, history in histories.items():
        plt.plot(history['train_losses'], label=f"{opt_name.upper()}")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # График точности
    plt.subplot(1, 2, 2)
    for opt_name, history in histories.items():
        plt.plot(history['val_accuracies'], label=f"{opt_name.upper()}")
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "optimizers_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print("🏆 BEST RESULTS COMPARISON")
    print("=" * 60)
    for opt_name, history in histories.items():
        print(f"{opt_name.upper():15} | Best Val Acc: {history['best_val_acc']:.2f}% | LR: {history['lr']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
