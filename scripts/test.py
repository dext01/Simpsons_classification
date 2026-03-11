import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import get_resnet18_finetune
from utils import seed_everything
from dataset import KaggleTestDataset


def main():
    MAIN_DATA_DIR = "/home/stepan/ai/labs/Simpsons_classification/data"
    TEST_DIR = "/home/stepan/Загрузки/archive(2)/kaggle_simpson_testset/kaggle_simpson_testset"
    MODEL_PATH = "/home/stepan/ai/labs/Simpsons_classification/artifacts/best_model.pth"
    BATCH_SIZE = 32
    SEED = 42

    seed_everything(SEED)

    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "/home/stepan/ai/labs/Simpsons_classification/artifacts/model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18_finetune(num_classes=42).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = KaggleTestDataset(MAIN_DATA_DIR, TEST_DIR, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro Precision: {prec_macro:.4f}")
    print(f"Macro Recall: {rec_macro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.title("Confusion Matrix (Kaggle Test Set)")
    plt.savefig("artifacts/test_confusion_matrix.png")


if __name__ == "__main__":
    main()
