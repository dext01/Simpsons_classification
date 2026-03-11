import os
import shutil
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,_score, recall_score, f1, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, './src')

from model import get_resnet18_finetune
from utils import seed_everything

def extract_character_from_filename(filename):
    if '_on_' in filename:
        return filename.split('_on_')[0]
    # Если шаблон другой — адаптируй под свой случай
    return filename.split('.')[0]  # как fallback

def organize_test_data(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"🔍 Найдено {len(files)} тестовых изображений")

    for filename in files:
        character = extract_character_from_filename(filename)
        char_dir = os.path.join(dst_dir, character)
        os.makedirs(char_dir, exist_ok=True)
        src = os.path.join(src_dir, filename)
        dst = os.path.join(char_dir, filename)
        shutil.copy(src, dst)
    
    print(f"✅ Структура создана в: {dst_dir}")
    # Показываем первые 5 классов
    classes = sorted(os.listdir(dst_dir))
    print(f"📁 Классы: {classes[:5]} (всего {len(classes)})")
    return dst_dir

def main():
    TEST_DIR = "/content/kaggle_simpson_testset/kaggle_simpson_testset"  # ← замени на путь к твоей папке с .jpg
    TEMP_TEST_DIR = "data/test_simple"
    MODEL_PATH = "artifacts/best_model.pth"
    BATCH_SIZE = 32
    SEED = 42

    seed_everything(SEED)

    test_root = organize_test_data(TEST_DIR, TEMP_TEST_DIR)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=test_root, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    classes = dataset.classes
    print(f"📊 Классы: {len(classes)} | Батчи: {len(loader)}")

    model = get_resnet18_finetune(num_classes=len(classes)).to('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(MODEL_PATH, map_location=model.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Модель загружена")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro = (
        precision_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='macro'),
        f1_score(y_true, y_pred, average='macro')
    )

    print("\n" + "="*60)
    print("🎯 ТЕСТОВЫЕ МЕТРИКИ (на простом датасете)")
    print("="*60)
    print(f"Всего образцов: {len(y_true)}")
    print(f"Accuracy:      {acc*100:.2f}%")
    print(f"Macro Precision: {prec_macro:.4f}")
    print(f"Macro Recall:    {rec_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")
    print("="*60)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("artifacts/test_confusion_simple.png")
    print("✅ Confusion matrix saved to: artifacts/test_confusion_simple.png")

if __name__ == "__main__":
    main()
