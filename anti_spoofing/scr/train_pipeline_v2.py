import os
import logging
import torch
from deepPixBiS_model_v2 import DeepPiXBiS
from data_preparation_v2 import get_dataloaders
from train_v2 import train_model
from test_v2 import test_model, evaluate_at_thresholds

# Setup logging to both console and file for pipeline
logging.basicConfig(
    filename="logs/pipeline.log",
    filemode="w",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dir = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\dataset\train"
    test_dir = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\dataset\test"

    batch_size = 32
    num_epochs = 10

    logging.info(f"Train directory: {train_dir}")
    logging.info(f"Test directory: {test_dir}")
    logging.info(f"Batch size: {batch_size}, Number of epochs: {num_epochs}")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(train_dir, test_dir, batch_size=batch_size)

    # Initialize model
    model = DeepPiXBiS().to(device)

    # Train
    train_model(model, train_loader, val_loader, device=device, num_epochs=num_epochs)

    # Load best saved model for testing
    model.load_state_dict(torch.load("best_model.pth"))
    logging.info("Loaded best model checkpoint for testing.")

    # Find best threshold
    best_threshold = evaluate_at_thresholds(model, test_loader, device)
    logging.info(f"Best threshold selected: {best_threshold:.2f}")

    # Final test with best threshold
    log_path = "logs/test_metrics_v3.txt"
    os.makedirs("logs", exist_ok=True)
    test_model(model, test_loader, device, threshold=best_threshold, log_path=log_path)
    logging.info(f"Test results saved to {log_path}")

    # Save final model
    output_dir = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\model"  # <-- Change this!
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "deepPixBiS_final.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Final model saved at {model_path}")

if __name__ == "__main__":
    main()
