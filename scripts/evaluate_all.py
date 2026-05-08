import os
import sys
import time
import torch
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from base_models import get_model
from models_with_cbam import get_model_with_attention
from dataset import load_dataset

DATASET = "datalib/sachin_kumar_ad_dataset"
BATCH_SIZE = 32

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, DATASET)
WAVELET_DATA_DIR = os.path.join(RAW_DATA_DIR, "dtcwt_preprocessed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load both datasets once
print("Loading raw dataset...")
raw_train, raw_val, raw_test, class_names = load_dataset(
    RAW_DATA_DIR, mode="raw", batch_size=BATCH_SIZE
)
print("\nLoading wavelet dataset...")
wav_train, wav_val, wav_test, _ = load_dataset(
    WAVELET_DATA_DIR, mode="wavelet", batch_size=BATCH_SIZE
)

def load_model_from_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    name = checkpoint["model_name"]
    mode = checkpoint["mode"]
    use_cbam = checkpoint["use_cbam"]
    in_channels = checkpoint["in_channels"]
    num_classes = checkpoint["num_classes"]

    if use_cbam:
        model = get_model_with_attention(name, num_classes=num_classes, in_channels=in_channels)
    else:
        model = get_model(name, num_classes=num_classes, in_channels=in_channels)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model, checkpoint

def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []
    total_time = 0
    num_images = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE)

            start = time.time()
            outputs = model(images)
            torch.cuda.synchronize()
            total_time += time.time() - start

            _, predicted = outputs.max(1)
            num_images += labels.size(0)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Metrics
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1)
    cm = confusion_matrix(all_labels, all_preds)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    inf_time = (total_time / num_images) * 1000  # ms per image

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "params_M": params,
        "inference_ms": inf_time,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }

# Find all checkpoints
checkpoints = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith("_best.pth")])

print(f"\n{'='*80}")
print(f"Found {len(checkpoints)} checkpoints")
print(f"{'='*80}")

all_results = []

for ckpt_name in checkpoints:
    ckpt_path = os.path.join(RESULTS_DIR, ckpt_name)
    model, checkpoint = load_model_from_checkpoint(ckpt_path)

    mode = checkpoint["mode"]
    test_loader = wav_test if mode == "wavelet" else raw_test

    results = evaluate_model(model, test_loader)

    config_str = f"{checkpoint['model_name']} | {mode}"
    if checkpoint["use_cbam"]:
        config_str += " + CBAM"

    print(f"\n{'-'*80}")
    print(f"Config: {config_str}")
    print(f"File:   {ckpt_name}")
    print(f"Params: {results['params_M']:.2f}M | Inference: {results['inference_ms']:.2f}ms/image")
    print(f"Accuracy: {results['accuracy']:.4f} | Balanced Accuracy: {results['balanced_accuracy']:.4f} | Macro F1: {results['macro_f1']:.4f}")
    print(f"\nPer-class breakdown:")
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for i, name in enumerate(class_names):
        print(f"  {name:<25} {results['precision'][i]:>10.4f} {results['recall'][i]:>10.4f} {results['f1'][i]:>10.4f}")
    print(f"\nConfusion Matrix:")
    print(results["confusion_matrix"])

    all_results.append({
        "config": config_str,
        "file": ckpt_name,
        **{k: v for k, v in results.items() if k not in ("all_preds", "all_labels", "confusion_matrix", "precision", "recall", "f1")}
    })

    del model
    torch.cuda.empty_cache()

# Summary table
print(f"\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}")
print(f"{'Config':<40} {'Acc':>8} {'Bal Acc':>8} {'F1':>8} {'Params':>8} {'ms/img':>8}")
print(f"{'-'*80}")
for r in all_results:
    print(f"{r['config']:<40} {r['accuracy']:>8.4f} {r['balanced_accuracy']:>8.4f} {r['macro_f1']:>8.4f} {r['params_M']:>7.2f}M {r['inference_ms']:>7.2f}")