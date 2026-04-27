import sys
sys.path.append(".")
import torch
from scripts.base_models import get_model

device = torch.device("cuda")
dummy = torch.randn(4, 3, 224, 224).to(device)

model_names = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v2", "deit_tiny"]

for name in model_names:
    model = get_model(name, num_classes=4).to(device).eval()
    with torch.no_grad():
        output = model(dummy)
    print(f"{name}: {output.shape} -> {output[0].tolist()}")
    del model
    torch.cuda.empty_cache()