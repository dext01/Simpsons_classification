import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4, save_dir="artifacts"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # все параметры

    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # прямой проход
            outputs = model(inputs)

            # loss
            loss = criterion(outputs, labels)

            # обратный проход
            optimizer.zero_grad()
            loss.backward()  # вычисляет градиенты для слоёв
            optimizer.step()  # обновляет веса

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # валидация
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

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    # график
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy", color="orange")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    plt.close()