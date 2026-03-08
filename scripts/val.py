import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import get_resnet18_finetune
from src.utils import seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Путь к весам")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_cm", type=str, default=None, help="Сохранить матрицу ошибок в файл (путь)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print(f"⚙️ Config: seed={args.seed}, batch_size={args.batch_size}")

    _, val_loader, classes = get_dataloaders(args.data_path, args.batch_size, seed=args.seed)
    num_classes = len(classes)

    print(f"📊 Classes: {num_classes}")
    print(f"📦 Val batches: {len(val_loader)}")

    # модель
    model = get_resnet18_finetune(num_classes=num_classes).to(device)

    try:
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded: {args.model_path}")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    model.eval()
    criterion = nn.CrossEntropyLoss() # ф-ия потеррь

    all_preds = [] # все предсказания
    all_labels = [] # истинные
    total_loss = 0.0 # накопление общего лосса
    total_samples = 0 # счетчик обработанных обьектов

    print("🔄 Running validation...")

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_samples += labels.size(0)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    avg_loss = total_loss / total_samples

    acc = accuracy_score(y_true, y_pred)

    prec_macro = precision_score(y_true, y_pred, average='macro')
    rec_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    prec_weighted = precision_score(y_true, y_pred, average='weighted')
    rec_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print("\n" + "=" * 50)
    print("📈 ОБЩИЕ МЕТРИКИ")
    print("=" * 50)
    print(f"Total Samples: {total_samples}")
    print(f"Average Loss:  {avg_loss:.4f}")
    print(f"Accuracy:      {acc * 100:.2f}%")
    print("-" * 50)
    print("Macro Average (все классы равны):")
    print(f"  Precision:   {prec_macro:.4f}")
    print(f"  Recall:      {rec_macro:.4f}")
    print(f"  F1-Score:    {f1_macro:.4f}")
    print("-" * 50)
    print("Weighted Average (учет размера класса):")
    print(f"  Precision:   {prec_weighted:.4f}")
    print(f"  Recall:      {rec_weighted:.4f}")
    print(f"  F1-Score:    {f1_weighted:.4f}")
    print("=" * 50)

    print("\n📋 ПОДРОБНЫЙ ОТЧЕТ ПО КЛАССАМ:")
    labels = list(range(len(classes)))
    print(classification_report(y_true, y_pred, target_names=classes, labels=labels, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    print("\n🔥 Confusion Matrix (первые 10 классов для краткости):")
    print(cm[:10, :10])

    save_path = args.save_cm or "artifacts/confusion_matrix.png"
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    plt.figure(figsize=(15, 15))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Confusion Matrix saved to: {save_path}")


if __name__ == "__main__":
    main()