import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class AntiSpoofingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.live_dir = os.path.join(root_dir, 'live')
        self.spoof_dir = os.path.join(root_dir, 'spoof')
        self.live_images = os.listdir(self.live_dir)
        self.spoof_images = os.listdir(self.spoof_dir)
        self.transform = transform

    def __len__(self):
        return len(self.live_images) + len(self.spoof_images)

    def __getitem__(self, index):
        if index < len(self.live_images):
            img_path = os.path.join(self.live_dir, self.live_images[index])
            label = 1  # live
        else:
            img_path = os.path.join(self.spoof_dir, self.spoof_images[index - len(self.live_images)])
            label = 1 # spoof

        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0   # Normalize to [0,1]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Automatically converts [0,255] to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Use these inside train_pipeline.py instead
def get_dataloaders(train_root, test_root, batch_size=32):
    train_dataset = AntiSpoofingDataset(root_dir=train_root, transform=transform)
    test_dataset = AntiSpoofingDataset(root_dir=test_root, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
