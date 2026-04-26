import torch
import torchvision.models as models
import timm
import time

device = torch.device("cuda")
dummy_input = torch.randn(1, 3, 224, 224).to(device)

model_configs = {
    "ResNet-18": lambda: models.resnet18(weights="IMAGENET1K_V1"),
    "ResNet-50": lambda: models.resnet50(weights="IMAGENET1K_V1"),
    "EfficientNet-B0": lambda: models.efficientnet_b0(weights="IMAGENET1K_V1"),
    "MobileNetV2": lambda: models.mobilenet_v2(weights="IMAGENET1K_V1"),
    "DeiT-Tiny": lambda: timm.create_model("deit_tiny_patch16_224", pretrained=True),
}

for name, loader in model_configs.items():
    print(f"\n{'='*50}")
    print(f"Loading {name}...")
    model = loader().to(device).eval()
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    # Timed inference
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            output = model(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Output shape: {output.shape}")
    print(f"Inference: {elapsed/100*1000:.1f}ms per image")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e6:.0f}MB")
    
    del model
    torch.cuda.empty_cache()

print(f"\n{'='*50}")
print("All models loaded and ran successfully!")