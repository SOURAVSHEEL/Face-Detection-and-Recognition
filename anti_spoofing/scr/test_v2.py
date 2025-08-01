import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import logging
import matplotlib.pyplot as plt

# Setup logging: log to file and console
logging.basicConfig(
    filename="logs/test.log",
    filemode="w",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def test_model(model, test_loader, device, threshold=0.5, log_path=None):
    model.eval()
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, global_logits = model(images)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(torch.sigmoid(global_logits).cpu().numpy())

    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    preds = (all_logits > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average=None, labels=[0,1])
    acc = (preds == all_labels).mean()
    cm = confusion_matrix(all_labels, preds, labels=[0,1])

    report = f"""
Threshold: {threshold:.2f}
Accuracy: {acc:.4f}

Confusion Matrix:
{cm}

Precision (Spoof=0): {precision[0]:.4f}
Recall (Spoof=0): {recall[0]:.4f}
F1 (Spoof=0): {f1[0]:.4f}

Precision (Live=1): {precision[1]:.4f}
Recall (Live=1): {recall[1]:.4f}
F1 (Live=1): {f1[1]:.4f}
"""

    logging.info(report.strip())
    if log_path:
        with open(log_path, "w") as f:
            f.write(report)

def evaluate_at_thresholds(model, dataloader, device, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.4, 0.8, 0.05)

    model.eval()
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            _, global_logits = model(images)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(torch.sigmoid(global_logits).cpu().numpy())

    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    best_f1 = 0
    best_threshold = 0.5
    precisions, recalls, f1s = [], [], []

    logging.info("Evaluating thresholds:")
    for thresh in thresholds:
        preds = (all_logits > thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='macro')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        logging.info(f"Threshold: {thresh:.2f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    logging.info(f"Best Threshold: {best_threshold:.2f} with Macro F1: {best_f1:.4f}")

    # Plot threshold metrics (optional)
    try:
        plt.plot(thresholds, precisions, label="Precision")
        plt.plot(thresholds, recalls, label="Recall")
        plt.plot(thresholds, f1s, label="F1 Score")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold Tuning")
        plt.legend()
        plt.show()
    except Exception as e:
        logging.warning(f"Plotting failed: {e}")

    return best_threshold
