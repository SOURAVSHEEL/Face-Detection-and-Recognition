import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import copy

def train_model(model, train_loader, val_loader, device, num_epochs=20, patience=3, save_path="best_model.pth"):
    model.to(device)

    # Weighted loss (can be tuned if data is imbalanced)
    pos_weight = torch.tensor([1.0]).to(device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    # Adam with regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device).view(-1)

            optimizer.zero_grad()
            heatmap, global_pred = model(images)

            pixel_labels = labels.view(-1, 1, 1).expand_as(heatmap)
            pixel_loss = criterion(heatmap, pixel_labels)
            global_loss = criterion(global_pred, labels.float())

            total_loss = pixel_loss + global_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += total_loss.item()

            preds = (torch.sigmoid(global_pred) > 0.5).int().cpu().numpy()
            true_labels = labels.int().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(true_labels)

        avg_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)

        # Evaluate on validation set
        val_f1, val_acc = evaluate_on_val(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping and model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_model_wts, save_path)
            print(f"New best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best Val F1: {best_val_f1:.4f}")
                break

    # Load best model
    model.load_state_dict(best_model_wts)
    return model


def evaluate_on_val(model, val_loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    running_val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).view(-1)

            heatmap, global_pred = model(images)

            pixel_labels = labels.view(-1, 1, 1).expand_as(heatmap)
            pixel_loss = criterion(heatmap, pixel_labels)
            global_loss = criterion(global_pred, labels.float())
            total_loss = pixel_loss + global_loss
            running_val_loss += total_loss.item()

            preds = (torch.sigmoid(global_pred) > 0.5).int().cpu().numpy()
            true_labels = labels.int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(true_labels)

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds)
    return val_f1, val_acc
