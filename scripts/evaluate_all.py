import os
import sys
import time
import torch
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
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
HAAR_DATA_DIR = os.path.join(RAW_DATA_DIR, "haar_preprocessed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all three datasets once
print("Loading raw dataset...")
raw_train, raw_val, raw_test, class_names = load_dataset(
    RAW_DATA_DIR, mode="raw", batch_size=BATCH_SIZE
)
print("\nLoading wavelet dataset...")
wav_train, wav_val, wav_test, _ = load_dataset(
    WAVELET_DATA_DIR, mode="wavelet", batch_size=BATCH_SIZE
)
print("\nLoading haar dataset...")
haar_train, haar_val, haar_test, _ = load_dataset(
    HAAR_DATA_DIR, mode="haar", batch_size=BATCH_SIZE
)
def is_ordinal(ckpt_filename):
    return "ordinal" in ckpt_filename


def coral_predict(logits):
    """Convert CORAL threshold logits to class predictions."""
    probas = torch.sigmoid(logits)
    predicted = torch.sum(probas > 0.5, dim=1)
    return predicted

def load_model_from_checkpoint(path):
    ckpt_filename = os.path.basename(path)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    name = checkpoint["model_name"]
    use_cbam = checkpoint.get("use_cbam", False)
    in_channels = checkpoint.get("in_channels", 3)
    num_classes = checkpoint["num_classes"]
    
    model = None
    
    # Ordinal models — use the ordinal model factory
    if is_ordinal(ckpt_filename):
        from ordinal_models import get_ordinal_model
        model = get_ordinal_model(name, num_classes=num_classes, 
                                   in_channels=in_channels, use_cbam=use_cbam)
    # CBAM placement variants
    elif "cbam_last" in ckpt_filename and "resnet18" in ckpt_filename:
        from models_cbam_placement import ResNet18_CBAM_LastOnly
        model = ResNet18_CBAM_LastOnly(num_classes=num_classes, in_channels=in_channels)
    elif "cbam_last" in ckpt_filename and "resnet50" in ckpt_filename:
        from models_cbam_placement import ResNet50_CBAM_LastOnly
        model = ResNet50_CBAM_LastOnly(num_classes=num_classes, in_channels=in_channels)
    elif "cbam_block" in ckpt_filename and "resnet18" in ckpt_filename:
        from models_cbam_placement import ResNet18_CBAM_BlockLevel
        model = ResNet18_CBAM_BlockLevel(num_classes=num_classes, in_channels=in_channels)
    elif "cbam_block" in ckpt_filename and "resnet50" in ckpt_filename:
        from models_cbam_placement import ResNet50_CBAM_BlockLevel
        model = ResNet50_CBAM_BlockLevel(num_classes=num_classes, in_channels=in_channels)
    elif "_cbam" in ckpt_filename:
        model = get_model_with_attention(name, num_classes=num_classes, in_channels=in_channels)
    else:
        model = get_model(name, num_classes=num_classes, in_channels=in_channels)
    
    if model is None:
        raise RuntimeError(f"No branch matched for checkpoint: {ckpt_filename}")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model, checkpoint

def evaluate_model(model, test_loader, ordinal=False):
    all_preds = []
    all_labels = []
    total_time = 0
    num_images = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE) if not isinstance(labels, torch.Tensor) else labels.to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            outputs = model(images)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start

            if ordinal:
                predicted = coral_predict(outputs)
            else:
                _, predicted = outputs.max(1)
            
            num_images += labels.size(0)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1)
    cm = confusion_matrix(all_labels, all_preds)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    inf_time = (total_time / num_images) * 1000

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "params_M": params,
        "inference_ms": inf_time,
    }


# Find all checkpoints
checkpoints = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith("_best.pth")])

print(f"\n{'='*80}")
print(f"Found {len(checkpoints)} checkpoints")
print(f"{'='*80}")

all_results = []

for ckpt_name in checkpoints:
    ckpt_path = os.path.join(RESULTS_DIR, ckpt_name)
    
    try:
        model, checkpoint = load_model_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"\n[ERROR] {ckpt_name}: {type(e).__name__}: {str(e)[:120]}")
        continue
    
    mode = checkpoint.get("mode", "raw")
    if mode == "wavelet":
        test_loader = wav_test
    elif mode == "haar":
        test_loader = haar_test
    else:
        test_loader = raw_test
    
    ordinal = is_ordinal(ckpt_name)
    results = evaluate_model(model, test_loader, ordinal=ordinal)
    config = ckpt_name.replace("_best.pth", "")
    
    print(f"\n{'-'*60}")
    print(f"Config: {config}{' [ORDINAL]' if ordinal else ''}")
    print(f"Params: {results['params_M']:.2f}M | Inference: {results['inference_ms']:.2f}ms/image")
    print(f"Accuracy: {results['accuracy']:.4f} | Balanced Acc: {results['balanced_accuracy']:.4f} | Macro F1: {results['macro_f1']:.4f}")

    all_results.append({
        "config": config,
        "acc": results["accuracy"],
        "bal_acc": results["balanced_accuracy"],
        "f1": results["macro_f1"],
        "params": results["params_M"],
        "ms": results["inference_ms"],
        "ordinal": ordinal,
    })

    del model
    torch.cuda.empty_cache()

# Summary table
print(f"\n{'='*90}")
print("KAGGLE TEST SET — FULL SUMMARY")
print(f"{'='*90}")
print(f"{'Config':<40} {'Acc':>8} {'Bal Acc':>8} {'F1':>8} {'Params':>8} {'ms/img':>8}")
print(f"{'-'*90}")
for r in sorted(all_results, key=lambda x: -x['bal_acc']):
    print(f"{r['config']:<40} {r['acc']:>8.4f} {r['bal_acc']:>8.4f} {r['f1']:>8.4f} {r['params']:>7.2f}M {r['ms']:>7.2f}")