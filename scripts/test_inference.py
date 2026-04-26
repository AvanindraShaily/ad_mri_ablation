import torch
import torchvision.models as models
import torchvision.transforms as transforms
import timm
from PIL import Image
import os

device = torch.device("cuda")
base = r"C:\Users\avash\OneDrive\Desktop\projects\ad_ablation_study\Alzheimer (Preprocessed Data)\Alzheimer (Preprocessed Data)"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

classes = sorted(os.listdir(base))
print("Classes:", classes)

images = []
labels = []
for cls in classes:
    cls_path = os.path.join(base, cls)
    if os.path.isdir(cls_path):
        img_name = os.listdir(cls_path)[0]
        img = Image.open(os.path.join(cls_path, img_name)).convert("RGB")
        images.append(transform(img))
        labels.append(cls)
        print(f"Loaded: {cls} -> {img_name}")

batch = torch.stack(images).to(device)
print(f"\nBatch shape: {batch.shape}")

model_configs = {
    "ResNet-18": lambda: models.resnet18(weights="IMAGENET1K_V1"),
    "ResNet-50": lambda: models.resnet50(weights="IMAGENET1K_V1"),
    "EfficientNet-B0": lambda: models.efficientnet_b0(weights="IMAGENET1K_V1"),
    "MobileNetV2": lambda: models.mobilenet_v2(weights="IMAGENET1K_V1"),
    "DeiT-Tiny": lambda: timm.create_model("deit_tiny_patch16_224", pretrained=True),
}

for name, loader in model_configs.items():
    model = loader().to(device).eval()
    with torch.no_grad():
        output = model(batch)
    print(f"{name}: output shape {output.shape} - OK")
    del model
    torch.cuda.empty_cache()

print("\nAll models verified on real dataset!")