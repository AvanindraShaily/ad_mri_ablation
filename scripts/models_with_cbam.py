import torch.nn as nn
import torchvision.models as models
import timm
from cbam import CBAM, SpatialOnly


class ResNetWithCBAM(nn.Module):
    def __init__(self, base="resnet18", num_classes=4):
        super().__init__()
        if base == "resnet18":
            resnet = models.resnet18(weights="IMAGENET1K_V1")
        else:
            resnet = models.resnet50(weights="IMAGENET1K_V1")

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # CBAM after each stage
        # ResNet-18 channels: 64, 128, 256, 512
        # ResNet-50 channels: 256, 512, 1024, 2048
        if base == "resnet18":
            channels = [64, 128, 256, 512]
        else:
            channels = [256, 512, 1024, 2048]

        self.cbam1 = CBAM(channels[0])
        self.cbam2 = CBAM(channels[1])
        self.cbam3 = CBAM(channels[2])
        self.cbam4 = CBAM(channels[3])

        self.fc = nn.Linear(channels[3], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MobileNetWithCBAM(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")

        # MobileNetV2 has 19 inverted residual blocks in features[0] through features[18]
        # We split into stages and add CBAM after the last few stages
        self.features_early = mobilenet.features[:7]    # up to channel 32
        self.features_mid = mobilenet.features[7:14]    # up to channel 96
        self.features_late = mobilenet.features[14:]    # up to channel 1280

        self.cbam_mid = CBAM(96)
        self.cbam_late = CBAM(1280)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features_early(x)
        x = self.features_mid(x)
        x = self.cbam_mid(x)
        x = self.features_late(x)
        x = self.cbam_late(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EfficientNetWithSpatialAttention(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        effnet = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # EfficientNet-B0 features: 8 MBConv stages + final conv
        self.features_early = effnet.features[:5]    # up to channel 40
        self.features_late = effnet.features[5:]     # up to channel 1280

        # Spatial attention only — SE already handles channel attention
        self.spatial_mid = SpatialOnly()
        self.spatial_late = SpatialOnly()

        self.pool = effnet.avgpool
        self.classifier = nn.Sequential(
            effnet.classifier[0],  # dropout
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features_early(x)
        x = self.spatial_mid(x)
        x = self.features_late(x)
        x = self.spatial_late(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model_with_attention(name, num_classes=4):
    """Load a model with appropriate attention modules."""
    if name == "resnet18":
        return ResNetWithCBAM("resnet18", num_classes)
    elif name == "resnet50":
        return ResNetWithCBAM("resnet50", num_classes)
    elif name == "mobilenet_v2":
        return MobileNetWithCBAM(num_classes)
    elif name == "efficientnet_b0":
        return EfficientNetWithSpatialAttention(num_classes)
    elif name == "deit_tiny":
        # No CBAM for DeiT — return base model
        model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {name}")