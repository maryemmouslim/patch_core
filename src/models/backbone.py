import torch
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class WideResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer3 = torch.nn.functional.interpolate(
            layer3,
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat((layer2, layer3), dim=1)
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(features.shape[0], -1, features.shape[-1])
        return features
