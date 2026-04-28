import torch
import torch.nn as nn
import torchvision.models as models
import timm


def _replace_first_conv(module, attr, in_channels):
    # make the first convolutional layer have the same number of input channels as the input data, copying pretrained weights where possible
    old_conv = getattr(module, attr)
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, bias=(old_conv.bias is not None),
    )
    # copy pretrained weights for first 3 channels, init the rest
    with torch.no_grad():
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        new_conv.weight[:, :3] = old_conv.weight
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    setattr(module, attr, new_conv)


def get_model(name, num_classes=4, in_channels=3):
    if name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if in_channels != 3:
            _replace_first_conv(model, "conv1", in_channels)
    
    elif name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if in_channels != 3:
            _replace_first_conv(model, "conv1", in_channels)
    
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if in_channels != 3:
            _replace_first_conv(model.features[0], "0", in_channels)
    
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if in_channels != 3:
            _replace_first_conv(model.features[0], "0", in_channels)
    
    elif name == "deit_tiny":
        model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)
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
    
    else:
        raise ValueError(f"Unknown model: {name}")
    
    return model