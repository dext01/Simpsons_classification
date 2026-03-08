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
import pandas as pd
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import get_resnet18_finetune
from src.utils import seed_everything


def prepare_kaggle_test_data(output_dir="data/test_kaggle"):

    print("📥 Скачиваем тестовый датасет с Kaggle...")

    try:
        from kagglehub import dataset_download
        path = dataset_download("alexattia/the-simpsons-characters-dataset")
    except ImportError:
        print("❌ Ошибка: kagglehub не установлен. Установите: pip install kagglehub")
        return None
    except Exception as e:
        print(f"❌ Ошибка при скачивании датасета: {e}")
        return None

    test_images_dir = os.path.join(path, "testset", "testset")
    csv_file = os.path.join(path, "testset", "testset.csv")

    if not os.path.exists(test_images_dir):
        print(f"❌ Папка с тестовыми изображениями не найдена: {test_images_dir}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(csv_file):
        print(f"📄 Читаем разметку из {csv_file}...")
        df = pd.read_csv(csv_file)

        classes = df['character'].unique()
        for cls in classes:
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

        print(f"🔄 Организуем {len(df)} изображений в структуру папок...")
        for _, row in df.iterrows():
            img_id = row['id']
            character = row['character']
            src = os.path.join(test_images_dir, f"{img_id}.jpg")
            dst = os.path.join(output_dir, character, f"{img_id}.jpg")

            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"⚠️  Внимание: {src} не найден")

        print(f"✅ Тестовые данные организованы в {output_dir}/")
        print(f"📊 Классов: {len(classes)}, Всего изображений: {len(df)}")
        return output_dir
    else:
        print(f"❌ Файл разметки не найден: {csv_file}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="artifacts/best_model.pth", help="Путь к весам модели")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Путь к тестовым данным (если не указан, скачает с Kaggle)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_cm", type=str, default="artifacts/test_confusion_matrix.png",
                        help="Сохранить матрицу ошибок")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print(f"⚙️ Config: seed={args.seed}, batch_size={args.batch_size}")

    # Если путь к тестовым данным не указан или не существует - скачиваем с Kaggle
    if args.test_path is None or not os.path.exists(args.test_path):
        print(f"⚠️  Тестовые данные не найдены. Скачиваем с Kaggle...")
        test_path = prepare_kaggle_test_data()
        if test_path is None:
            print("❌ Не удалось подготовить тестовые данные. Завершение.")
            return
        args.test_path = test_path

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=args.test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    classes = test_dataset.classes
    num_classes = len(classes)

    print(f"📊 Classes: {num_classes}")
    print(f"📦 Test batches: {len(test_loader)}")
    print(f"📁 Test path: {args.test_path}")

    model = get_resnet18_finetune(num_classes=num_classes).to(device)

    try:
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded: {args.model_path}")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    print("🔄 Running test evaluation...")

    with torch.no_grad():
        for inputs, labels in test_loader:
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
    print("📈 ТЕСТОВЫЕ МЕТРИКИ (Kaggle Test Set)")
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

    # Сохраняем матрицу ошибок
    save_path = args.save_cm
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    plt.figure(figsize=(15, 15))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title("Normalized Confusion Matrix (Test Set)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Confusion Matrix saved to: {save_path}")


if __name__ == "__main__":
    main()