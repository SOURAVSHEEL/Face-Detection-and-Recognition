import logging
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

# Setup logging: log to file and console
logging.basicConfig(
    filename="logs/train.log",
    filemode="w",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    pos_weight = torch.tensor(1.2).to(device)  # Penalize false positive "live"
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_f1 = 0
    patience = 3
    wait = 0

    logging.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            _, global_logits = model(images)
            loss = criterion(global_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, global_logits = model(images)
                probs = torch.sigmoid(global_logits)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend((probs > 0.5).cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro')

        log_msg = (f"Epoch {epoch+1}/{num_epochs} | "
                   f"Loss: {epoch_loss:.4f} | "
                   f"Val Precision: {precision:.4f} | "
                   f"Val Recall: {recall:.4f} | "
                   f"Val F1: {f1:.4f}")
        logging.info(log_msg)

        # Early stopping
        if f1 > best_val_f1:
            best_val_f1 = f1
            wait = 0
            torch.save(model.state_dict(), "best_model.pth")
            logging.info("Saved best model checkpoint.")
        else:
            wait += 1
            if wait > patience:
                logging.info("Early stopping triggered.")
                break

    logging.info("Training complete.")
