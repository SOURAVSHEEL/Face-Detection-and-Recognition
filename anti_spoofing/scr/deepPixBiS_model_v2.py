import torch
import torch.nn as nn
import torchvision.models as models

class DeepPiXBiS(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(DeepPiXBiS, self).__init__()
        # Load pre-trained ResNet18 and remove the last two layers
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Output: [B, 512, H, W]

        # Increase dropout rate for regularization
        self.dropout = nn.Dropout2d(p=0.5)

        # Pixel-wise classification layer
        self.conv_last = nn.Conv2d(512, 1, kernel_size=1)  # Output: [B, 1, H, W]

        # Global average pooling for global spoof score
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Output: [B, 1, 1, 1]

    def forward(self, x):
        features = self.backbone(x)        # Shape: [B, 512, H, W]
        features = self.dropout(features)
        heatmap_logits = self.conv_last(features)  # No sigmoid here, raw logits
        global_logits = self.avg_pool(heatmap_logits).view(-1)  # Flatten to [B]
        return heatmap_logits.squeeze(1), global_logits
