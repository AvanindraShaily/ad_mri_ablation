import torch
import torch.nn as nn
import torchvision.models as models
from cbam import CBAM
from base_models import _replace_first_conv


class ResNet18_CBAM_LastOnly(nn.Module):
    """CBAM only after the final stage."""
    def __init__(self, num_classes=4, in_channels=3):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        if in_channels != 3:
            _replace_first_conv(resnet, "conv1", in_channels)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # CBAM only after the last stage
        self.cbam4 = CBAM(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CBAMResBlock(nn.Module):
    """Wrapper that adds CBAM inside a residual block, before skip addition."""
    def __init__(self, block, channels):
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = block.downsample
        self.cbam = CBAM(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)  # CBAM before skip addition
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
class CBAMBottleneck(nn.Module):
    """Wrapper that adds CBAM inside a ResNet-50 Bottleneck block."""
    def __init__(self, block, channels):
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.conv3 = block.conv3
        self.bn3 = block.bn3
        self.relu = block.relu
        self.downsample = block.downsample
        self.cbam = CBAM(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.cbam(out)  # CBAM before skip addition
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet50_CBAM_BlockLevel(nn.Module):
    """CBAM inside every bottleneck block for ResNet-50."""
    def __init__(self, num_classes=4, in_channels=3):
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        if in_channels != 3:
            _replace_first_conv(resnet, "conv1", in_channels)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        # ResNet-50 channels after each bottleneck: 256, 512, 1024, 2048
        layer_channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }

        for layer_name, channels in layer_channels.items():
            layer = getattr(resnet, layer_name)
            new_blocks = []
            for block in layer:
                new_blocks.append(CBAMBottleneck(block, channels))
            setattr(self, layer_name, nn.Sequential(*new_blocks))

        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet18_CBAM_BlockLevel(nn.Module):
    """CBAM inside every residual block (original paper's approach)."""
    def __init__(self, num_classes=4, in_channels=3):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        if in_channels != 3:
            _replace_first_conv(resnet, "conv1", in_channels)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        # Replace each BasicBlock with CBAMResBlock
        channels = [64, 64, 128, 128, 256, 256, 512, 512]
        block_idx = 0

        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(resnet, layer_name)
            new_blocks = []
            for block in layer:
                new_blocks.append(CBAMResBlock(block, channels[block_idx]))
                block_idx += 1
            setattr(self, layer_name, nn.Sequential(*new_blocks))

        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x