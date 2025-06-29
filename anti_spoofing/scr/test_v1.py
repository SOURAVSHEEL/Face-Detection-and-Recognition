import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

def test_model(model, test_loader, device, log_path="test_metrics.txt", threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device).view(-1)

            _, global_pred = model(images)

            # Apply sigmoid and threshold
            probs = torch.sigmoid(global_pred).view(-1)
            preds = (probs > threshold).int().cpu().numpy()
            true = labels.int().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(true)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Spoof", "Live"])

    # Display
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("Classification Report:\n", report)

    # Save to file
    with open(log_path, "w") as f:
        f.write("==== Anti-Spoofing Test Metrics ====\n")
        f.write(f"Threshold     : {threshold:.2f}\n")
        f.write(f"Accuracy      : {acc:.4f}\n")
        f.write(f"F1 Score      : {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"\n📄 Metrics saved to: {log_path}")
