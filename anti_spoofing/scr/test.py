from data_preparation import test_dataset, test_loader
from deepPixBiS_model import DeepPiXBiS
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepPiXBiS().to(device)

def test_model(test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            heatmap, global_features = model(images)
            predicted_labels = (global_features > 0.5).float()  # Thresholding at 0.5

            total_correct += (predicted_labels == labels.float()).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")