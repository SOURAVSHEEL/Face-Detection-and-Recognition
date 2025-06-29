import torch
import torch.nn as nn
import torchvision.models as models

# class DeepPiXBiS(nn.Module):
#     def __init__(self, backbone='resnet18'):
#         super(DeepPiXBiS, self).__init__()

#         # pre-trained backbone
#         self.backbone =  models.resnet18(pretrained=True)

#         # replace the last fully connected layer

#         self.backbone = nn.Sequential(*list(self.backbone.children()))[:-2]

#         ## pixel wise binary classification 
#         self.conv_last = nn.Conv2d(512, 1, kernel_size=1)

#         # global average pooling
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         features = self.backbone(x)
#         heatmap = torch.sigmoid(self.conv_last(features))
#         global_features = self.avg_pool(heatmap).view(-1)

#         return heatmap.squeeze(1), global_features
    

class DeepPiXBiS(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(DeepPiXBiS, self).__init__()

        # Load pre-trained ResNet18 and remove the final layers
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Output: [B, 512, H, W]  (32, 512, 7,7)

        self.dropout  = nn.Dropout2d(p=0.3)  

        # Pixel-wise classification layer
        self.conv_last = nn.Conv2d(512, 1, kernel_size=1)  # Output: [B, 1, H, W]

        # Global average pooling for global spoof score
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Output: [B, 1, 1, 1]

    def forward(self, x):
        features = self.backbone(x)  # Shape: [B, 512, H, W]
        features = self.dropout(features)
        heatmap_logits = self.conv_last(features)  # No sigmoid here
        global_logits = self.avg_pool(heatmap_logits).view(-1)  # Flatten to shape: [B]

        return heatmap_logits.squeeze(1), global_logits  # Return raw logits

