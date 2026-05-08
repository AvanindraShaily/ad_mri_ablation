import torch
import torch.nn as nn
import torchvision.models as models
import timm
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss
from base_models import _replace_first_conv


def get_ordinal_model(name, num_classes=4, in_channels=3, use_cbam=False):
    """Create a model with CORAL ordinal regression head."""

    if use_cbam:
        from models_with_cbam import get_model_with_attention
        model = get_model_with_attention(name, num_classes=num_classes, in_channels=in_channels)
        # Replace the classification head with CoralLayer
        if name in ("resnet18", "resnet50"):
            in_features = model.fc.in_features
            model.fc = CoralLayer(in_features, num_classes)
        elif name == "mobilenet_v2":
            in_features = model.classifier.in_features
            model.classifier = CoralLayer(in_features, num_classes)
        elif name == "efficientnet_b0":
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                model.classifier[0],
                CoralLayer(in_features, num_classes)
            )
    else:
        if name == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1")
            if in_channels != 3:
                _replace_first_conv(model, "conv1", in_channels)
            in_features = model.fc.in_features
            model.fc = CoralLayer(in_features, num_classes)

        elif name == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V1")
            if in_channels != 3:
                _replace_first_conv(model, "conv1", in_channels)
            in_features = model.fc.in_features
            model.fc = CoralLayer(in_features, num_classes)

        elif name == "efficientnet_b0":
            model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            if in_channels != 3:
                _replace_first_conv(model.features[0], "0", in_channels)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                model.classifier[0],
                CoralLayer(in_features, num_classes)
            )

        elif name == "mobilenet_v2":
            model = models.mobilenet_v2(weights="IMAGENET1K_V1")
            if in_channels != 3:
                _replace_first_conv(model.features[0], "0", in_channels)
            in_features = model.classifier[1].in_features
            model.classifier = CoralLayer(in_features, num_classes)

        elif name == "deit_tiny":
            model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
            if in_channels != 3:
                old_proj = model.patch_embed.proj
                new_proj = nn.Conv2d(
                    in_channels, old_proj.out_channels,
                    kernel_size=old_proj.kernel_size, stride=old_proj.stride,
                    padding=old_proj.padding, bias=(old_proj.bias is not None),
                )
                with torch.no_grad():
                    nn.init.kaiming_normal_(new_proj.weight, mode="fan_out")
                    new_proj.weight[:, :3] = old_proj.weight
                    if old_proj.bias is not None:
                        new_proj.bias.copy_(old_proj.bias)
                model.patch_embed.proj = new_proj
            in_features = model.head.in_features
            model.head = CoralLayer(in_features, num_classes)

    return model


def coral_predict(logits):
    """Convert CORAL logits to class predictions."""
    probas = torch.sigmoid(logits)
    predicted = torch.sum(probas > 0.5, dim=1)
    return predicted