# train_pipeline.py
import os
import torch
from deepPixBiS_model import DeepPixBiS
from data_preparation import get_dataloaders
from train import train_model
from test import evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths to your data
    train_dir = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\dataset\train"
    test_dir = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\dataset\test"
    
    # Load data
    train_loader, test_loader = get_dataloaders(train_dir, test_dir, batch_size=32)

    # Initialize model
    model = DeepPixBiS().to(device)

    # Train
    train_model(model, train_loader, device, num_epochs=20, lr=0.0001)

    # Evaluate
    evaluate(model, test_loader, device)

    # Save model
    output_dir = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(output_dir, "deepPixBiS.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path} successfully.")

if __name__ == "__main__":
    main()
