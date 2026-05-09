import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = r"c:\Users\avash\OneDrive\Desktop\projects\ad_ablation_study"
RAW_DIR = os.path.join(PROJECT_ROOT, "datalib", "sachin_kumar_ad_dataset")
HAAR_DIR = os.path.join(RAW_DIR, "haar_preprocessed")

CLASS_NAMES = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]

# Check 1: Do raw and Haar have the same number of files per class?
print("=" * 60)
print("CHECK 1: File counts match between raw and Haar")
print("=" * 60)
for cls in CLASS_NAMES:
    raw_files = sorted([f for f in os.listdir(os.path.join(RAW_DIR, cls))])
    haar_files = sorted([f for f in os.listdir(os.path.join(HAAR_DIR, cls)) if f.endswith(".npy")])
    print(f"{cls:25s}  raw: {len(raw_files):5d}  haar: {len(haar_files):5d}  match: {len(raw_files) == len(haar_files)}")

# Check 2: Do filenames correspond? Print first 5 sorted filenames for one class.
print("\n" + "=" * 60)
print("CHECK 2: First 5 sorted filenames for Mild_Demented")
print("=" * 60)
raw_files = sorted(os.listdir(os.path.join(RAW_DIR, "Mild_Demented")))[:5]
haar_files = sorted([f for f in os.listdir(os.path.join(HAAR_DIR, "Mild_Demented")) if f.endswith(".npy")])[:5]
for r, h in zip(raw_files, haar_files):
    print(f"  raw: {r}    haar: {h}")

# Check 3: Load one Haar .npy and inspect contents
print("\n" + "=" * 60)
print("CHECK 3: Inspect a Haar .npy file")
print("=" * 60)
sample_path = os.path.join(HAAR_DIR, "Mild_Demented", haar_files[0])
data = np.load(sample_path)
print(f"Path: {sample_path}")
print(f"Shape: {data.shape}")
print(f"Dtype: {data.dtype}")
print(f"Per-channel ranges:")
for i in range(data.shape[0]):
    print(f"  Channel {i}: min={data[i].min():8.3f}  max={data[i].max():8.3f}  mean={data[i].mean():8.3f}  std={data[i].std():8.3f}")

# Check 4: Visualize all 6 channels alongside the corresponding raw image
print("\n" + "=" * 60)
print("CHECK 4: Visualizing channels (saves to verify_haar_output.png)")
print("=" * 60)

# Find the corresponding raw image
raw_basename = haar_files[0].replace(".npy", "")
# Try a few common extensions
for ext in [".jpg", ".jpeg", ".png"]:
    candidate = os.path.join(RAW_DIR, "Mild_Demented", raw_basename + ext)
    if os.path.exists(candidate):
        raw_path = candidate
        break
else:
    print(f"WARNING: Could not find raw image matching {raw_basename}")
    raw_path = None

fig, axes = plt.subplots(1, 7, figsize=(21, 3))
if raw_path:
    raw_img = np.array(Image.open(raw_path).convert("RGB"))
    axes[0].imshow(raw_img)
    axes[0].set_title("Raw RGB")
else:
    axes[0].set_title("Raw not found")
axes[0].axis("off")

for i in range(6):
    axes[i+1].imshow(data[i], cmap="gray")
    axes[i+1].set_title(f"Channel {i}")
    axes[i+1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "verify_haar_output.png"), dpi=100, bbox_inches="tight")
print(f"Saved visualization to verify_haar_output.png")

# Check 5: Sanity check the Haar checkpoint metadata
print("\n" + "=" * 60)
print("CHECK 5: deit_tiny_haar checkpoint metadata")
print("=" * 60)
import torch
ckpt_path = os.path.join(PROJECT_ROOT, "results", "deit_tiny_haar_best.pth")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"model_name:    {ckpt.get('model_name')}")
print(f"mode:          {ckpt.get('mode')}")
print(f"in_channels:   {ckpt.get('in_channels')}")
print(f"num_classes:   {ckpt.get('num_classes')}")
print(f"class_names:   {ckpt.get('class_names')}")
print(f"best_val_acc:  {ckpt.get('best_val_acc'):.4f}")
print(f"epoch:         {ckpt.get('epoch')}")

# Verify the patch_embed.proj has 6 input channels
state_dict = ckpt['model_state_dict']
proj_weight = state_dict['patch_embed.proj.weight']
print(f"patch_embed.proj.weight shape: {proj_weight.shape}")
print(f"  -> Expected (192, 6, 16, 16) for Haar DeiT-Tiny")
print(f"  -> Match: {tuple(proj_weight.shape) == (192, 6, 16, 16)}")