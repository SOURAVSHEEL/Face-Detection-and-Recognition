import torch
import torch.nn as nn
import torchvision.models as models

class DeepPiXBiS(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(DeepPiXBiS, self).__init__()

        # pre-trained backbone
        self.backbone =  models.resnet18(pretrained=True)

        # replace the last fully connected layer

        self.backbone = nn.Sequential(*list(self.backbone.children()))[:-2]

        ## pixel wise binary classification 
        self.conv_last = nn.Conv2d(512, 1, kernel_size=1)

        # global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.backbone(x)
        heatmap = torch.sigmoid(self.conv_last(features))
        global_features = self.avg_pool(heatmap).view(-1)

        return heatmap.squeeze(1), global_features
    

    


