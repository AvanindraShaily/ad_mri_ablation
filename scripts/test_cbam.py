import torch
from base_models import get_model
from models_with_cbam import get_model_with_attention

device = torch.device("cuda")
dummy = torch.randn(4, 3, 224, 224).to(device)

model_names = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v2", "deit_tiny"]

for name in model_names:
    # Baseline
    base = get_model(name, num_classes=4).to(device).eval()
    base_params = sum(p.numel() for p in base.parameters()) / 1e6

    # With attention
    attn = get_model_with_attention(name, num_classes=4).to(device).eval()
    attn_params = sum(p.numel() for p in attn.parameters()) / 1e6

    with torch.no_grad():
        out_base = base(dummy)
        out_attn = attn(dummy)

    print(f"{name}:")
    print(f"  Baseline: {base_params:.2f}M params, output {out_base.shape}")
    print(f"  +Attention: {attn_params:.2f}M params, output {out_attn.shape}")
    print(f"  Overhead: +{attn_params - base_params:.4f}M params")
    print()

    del base, attn
    torch.cuda.empty_cache()