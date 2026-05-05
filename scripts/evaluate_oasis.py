import os
import sys
import time
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from base_models import get_model
from models_with_cbam import get_model_with_attention
from dataset import AlzheimerDataset, WaveletDataset, get_transforms
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OASIS_DIR = os.path.join(PROJECT_ROOT, "OASIS")
OASIS_HAAR_DIR = os.path.join(OASIS_DIR, "haar_preprocessed")
OASIS_DTCWT_DIR = os.path.join(OASIS_DIR, "dtcwt_preprocessed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (excluding preprocessed folders)
class_names_oasis = sorted([
    d for d in os.listdir(OASIS_DIR)
    if os.path.isdir(os.path.join(OASIS_DIR, d))
    and d not in ("dtcwt_preprocessed", "haar_preprocessed")
])
class_to_idx = {name: i for i, name in enumerate(class_names_oasis)}

# Kaggle class mapping
kaggle_classes = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
oasis_to_kaggle = {}
for i, name in enumerate(class_names_oasis):
    if name in kaggle_classes:
        oasis_to_kaggle[i] = kaggle_classes.index(name)

print(f"OASIS classes: {class_names_oasis}")
print(f"OASIS to Kaggle mapping: {oasis_to_kaggle}")


def build_oasis_loader(mode, batch_size=32):
    """Build OASIS data loader for a given mode."""
    if mode == "raw":
        data_dir = OASIS_DIR
        ext = None
    elif mode == "haar":
        data_dir = OASIS_HAAR_DIR
        ext = ".npy"
    elif mode == "wavelet":
        data_dir = OASIS_DTCWT_DIR
        ext = ".npy"
    else:
        return None

    all_paths = []
    all_labels = []
    for cls_name in class_names_oasis:
        cls_path = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_path):
            print(f"  Warning: {cls_path} not found")
            continue
        for fname in sorted(os.listdir(cls_path)):
            if ext and not fname.endswith(ext):
                continue
            all_paths.append(os.path.join(cls_path, fname))
            all_labels.append(class_to_idx[cls_name])

    if len(all_paths) == 0:
        return None

    if mode == "raw":
        dataset = AlzheimerDataset(all_paths, all_labels, get_transforms("test"))
    else:
        dataset = WaveletDataset(all_paths, all_labels, target_size=224)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"  Loaded {len(dataset)} images in '{mode}' mode")
    return loader


def load_model_from_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    name = checkpoint["model_name"]
    use_cbam = checkpoint["use_cbam"]
    in_channels = checkpoint.get("in_channels", 3)
    num_classes = checkpoint["num_classes"]

    if checkpoint.get("ordinal", False):
        return None, checkpoint

    if use_cbam:
        model = get_model_with_attention(name, num_classes=num_classes, in_channels=in_channels)
    else:
        model = get_model(name, num_classes=num_classes, in_channels=in_channels)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model, checkpoint


def evaluate_on_oasis(model, loader):
    all_preds = []
    all_labels_mapped = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for pred, label in zip(predicted.cpu().tolist(), labels.tolist()):
                kaggle_label = oasis_to_kaggle[label]
                all_preds.append(pred)
                all_labels_mapped.append(kaggle_label)

    correct = sum(p == l for p, l in zip(all_preds, all_labels_mapped))
    total = len(all_preds)
    acc = correct / total
    bal_acc = balanced_accuracy_score(all_labels_mapped, all_preds)

    per_class = {}
    for i, name in enumerate(class_names_oasis):
        kaggle_idx = oasis_to_kaggle[i]
        mask = [j for j, l in enumerate(all_labels_mapped) if l == kaggle_idx]
        if mask:
            cls_correct = sum(1 for j in mask if all_preds[j] == kaggle_idx)
            per_class[name] = (cls_correct, len(mask), cls_correct / len(mask))

    return acc, bal_acc, per_class


# Pre-build loaders for each mode
print("\nBuilding OASIS loaders...")
oasis_loaders = {}
for mode in ["raw", "haar", "wavelet"]:
    print(f"  Mode: {mode}")
    loader = build_oasis_loader(mode)
    if loader:
        oasis_loaders[mode] = loader

# Find all checkpoints and evaluate
checkpoints = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith("_best.pth")])

print(f"\n{'='*80}")
print("OASIS CROSS-DATASET GENERALIZATION")
print(f"{'='*80}")

results = []

for ckpt_name in checkpoints:
    ckpt_path = os.path.join(RESULTS_DIR, ckpt_name)
    model, checkpoint = load_model_from_checkpoint(ckpt_path)

    if model is None:
        ordinal = checkpoint.get("ordinal", False)
        if ordinal:
            print(f"\nSkipping {ckpt_name} (ordinal model)")
        continue

    mode = checkpoint["mode"]

    if mode not in oasis_loaders:
        print(f"\nSkipping {ckpt_name} (no OASIS {mode} data available)")
        continue

    loader = oasis_loaders[mode]
    acc, bal_acc, per_class = evaluate_on_oasis(model, loader)

    config = f"{checkpoint['model_name']} | {mode}"
    if checkpoint['use_cbam']:
        config += " + CBAM"

    print(f"\n{'-'*60}")
    print(f"Config: {config}")
    print(f"Accuracy: {acc:.4f} | Balanced Accuracy: {bal_acc:.4f}")
    for name, (correct, total, class_acc) in per_class.items():
        print(f"  {name}: {correct}/{total} = {class_acc:.4f}")

    results.append({"config": config, "acc": acc, "bal_acc": bal_acc})

    del model
    torch.cuda.empty_cache()

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"{'Config':<45} {'Acc':>8} {'Bal Acc':>8}")
print(f"{'-'*61}")
for r in sorted(results, key=lambda x: -x['bal_acc']):
    print(f"{r['config']:<45} {r['acc']:>8.4f} {r['bal_acc']:>8.4f}")