import torch
import os
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

class AntiSpoofingDataset(Dataset):
    def __init__(self, image_paths, labels, live_transform=None, spoof_transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.live_transform = live_transform
        self.spoof_transform = spoof_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        img = Image.open(img_path).convert("RGB")

        # Apply stronger augmentation for live, simpler for spoof
        if label == 1:
            if self.live_transform:
                img = self.live_transform(img)
        else:
            if self.spoof_transform:
                img = self.spoof_transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


def load_balanced_paths(root_dir):
    """
    Loads balanced live and spoof image paths with labels (1 = live, 0 = spoof).
    """
    live_dir = os.path.join(root_dir, "live")
    spoof_dir = os.path.join(root_dir, "spoof")

    live_imgs = [os.path.join(live_dir, f) for f in os.listdir(live_dir) if f.lower().endswith((".jpg", ".png"))]
    spoof_imgs = [os.path.join(spoof_dir, f) for f in os.listdir(spoof_dir) if f.lower().endswith((".jpg", ".png"))]

    min_len = min(len(live_imgs), len(spoof_imgs))
    live_imgs = random.sample(live_imgs, min_len)
    spoof_imgs = random.sample(spoof_imgs, min_len)

    paths = live_imgs + spoof_imgs
    labels = [1] * len(live_imgs) + [0] * len(spoof_imgs)

    combined = list(zip(paths, labels))
    random.shuffle(combined)
    paths, labels = zip(*combined)

    return list(paths), list(labels)


def get_dataloaders(train_root, test_root, batch_size=32, val_split=0.2, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    # Strong augmentation for live samples
    live_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.5),
        transforms.GaussianBlur(3),
        transforms.RandomAffine(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Simpler transform for spoof samples
    spoof_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load and balance training data
    train_paths, train_labels = load_balanced_paths(train_root)
    train_dataset = AntiSpoofingDataset(train_paths, train_labels, live_transform=live_transform, spoof_transform=spoof_transform)

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # For val, use simpler transforms, so override with spoof_transform for both classes
    val_dataset.dataset.live_transform = spoof_transform
    val_dataset.dataset.spoof_transform = spoof_transform

    # Load test data similarly with simpler transform
    test_paths, test_labels = load_balanced_paths(test_root)
    test_dataset = AntiSpoofingDataset(test_paths, test_labels, live_transform=spoof_transform, spoof_transform=spoof_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
