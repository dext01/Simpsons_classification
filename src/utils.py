import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import random
import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to: {seed}")


def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4, save_dir="artifacts"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_accuracies = [], []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100 * correct / total
        val_accuracies.append(acc)

        if acc > best_val_acc:
            best_val_acc = acc
            best_epoch = epoch + 1
            best_model_path = os.path.join(save_dir, "best_model.pth")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 NEW BEST! Model saved at epoch {best_epoch} | Val Acc: {acc:.2f}%")

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% | Best: {best_val_acc:.2f}%")

    final_model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    print(f"🏆 BEST model (epoch {best_epoch}) saved: {best_model_path} | Val Acc: {best_val_acc:.2f}%")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.axhline(y=min(train_losses), color='r', linestyle='--', label=f'Min: {min(train_losses):.4f}')
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy", color="orange")
    plt.axhline(y=best_val_acc, color='g', linestyle='--', label=f'Best: {best_val_acc:.2f}%')
    plt.title("Validation Accuracy (%)")
    plt.legend()

    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    plt.close()
    print(f"📊 Training curves saved to {save_dir}/training_curve.png")