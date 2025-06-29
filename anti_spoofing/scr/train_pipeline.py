import os
import torch
from deepPixBiS_model import DeepPiXBiS
from data_preparation_v1 import get_dataloaders
from train_v1 import train_model
from test_v1 import test_model

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Dataset paths
    train_dir = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\dataset\train"
    test_dir = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\dataset\test"

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(train_dir, test_dir, batch_size=32)

    # Initialize model
    model = DeepPiXBiS().to(device)

    # Train
    train_model(model, train_loader, val_loader, device=device, num_epochs=10)


    # Evaluate
    test_log_path = "logs/test_metrics_v2.txt"
    os.makedirs("logs", exist_ok=True)
    test_model(model, test_loader, device=device, log_path=test_log_path)

    # Save model
    output_dir = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\model"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "deepPixBiS_v3.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")
    print(f"Test log saved at: {test_log_path}")

if __name__ == "__main__":
    main()
