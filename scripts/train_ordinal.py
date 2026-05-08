import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from coral_pytorch.losses import CoralLoss
from coral_pytorch.dataset import levels_from_labelbatch
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from ordinal_models import get_ordinal_model, coral_predict
from dataset import load_dataset

# ---- Config ----
DATASET = "datalib/sachin_kumar_ad_dataset"
MODE = "raw"                    # best config mode
MODEL_NAME = "resnet18"         # model to train
USE_CBAM = False                # best config CBAM setting
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
criterion = CoralLoss()

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, DATASET)
WAVELET_DATA_DIR = os.path.join(RAW_DATA_DIR, "dtcwt_preprocessed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DATA_DIR = WAVELET_DATA_DIR if MODE == "wavelet" else RAW_DATA_DIR
IN_CHANNELS = 13 if MODE == "wavelet" else 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"{'='*50}")
print(f"ORDINAL REGRESSION TRAINING")
print(f"Mode:        {MODE}")
print(f"Model:       {MODEL_NAME}")
print(f"CBAM:        {USE_CBAM}")
print(f"In channels: {IN_CHANNELS}")
print(f"Device:      {DEVICE}")
print(f"{'='*50}")

# ---- Data ----
train_loader, val_loader, test_loader, class_names = load_dataset(
    DATA_DIR, mode=MODE, batch_size=BATCH_SIZE
)
num_classes = len(class_names)
# ---- Weighted sampling for balanced training ----
train_labels = train_loader.dataset.labels
class_counts = [train_labels.count(i) for i in range(num_classes)]
sample_weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Rebuild train loader with sampler (can't use shuffle with sampler)
from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_loader.dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=0, pin_memory=True
)
print(f"Weighted sampler: {dict(zip(class_names, class_counts))}")

# ---- Model ----
model = get_ordinal_model(MODEL_NAME, num_classes=num_classes,
                          in_channels=IN_CHANNELS, use_cbam=USE_CBAM).to(DEVICE)
print(f"Loaded {MODEL_NAME} with CORAL ordinal head")

# ---- Optimizer, scheduler ----
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---- Build save name ----
suffix = f"_{MODE}"
if USE_CBAM:
    suffix += "_cbam"
suffix += "_ordinal_balanced"
save_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}{suffix}_best.pth")

# ---- Training loop ----
best_val_acc = 0.0

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = torch.tensor(labels).long().to(DEVICE)
        levels = levels_from_labelbatch(labels, num_classes=num_classes).float().to(DEVICE)


        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, levels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        predicted = coral_predict(logits)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    scheduler.step()
    train_loss /= train_total
    train_acc = train_correct / train_total

    # Validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = torch.tensor(labels).long().to(DEVICE)
            levels = levels_from_labelbatch(labels, num_classes=num_classes).float().to(DEVICE)


            logits = model(images)
            loss = criterion(logits, levels)

            val_loss += loss.item() * images.size(0)
            predicted = coral_predict(logits)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total
    epoch_time = time.time() - epoch_start

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"Time: {epoch_time:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "mode": MODE,
            "use_cbam": USE_CBAM,
            "in_channels": IN_CHANNELS,
            "num_classes": num_classes,
            "class_names": class_names,
            "best_val_acc": val_acc,
            "epoch": epoch + 1,
            "ordinal": True,
        }, save_path)
        print(f"  -> Saved best model (val acc: {val_acc:.4f})")

# ---- Test ----
print(f"\n{'='*50}")
print("Testing best model...")
checkpoint = torch.load(save_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = torch.tensor(labels).long().to(DEVICE)

        logits = model(images)
        predicted = coral_predict(logits)

        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc:.4f}")

print("\nPer-class results:")
for i, name in enumerate(class_names):
    cls_mask = [j for j, l in enumerate(all_labels) if l == i]
    if cls_mask:
        cls_correct = sum(1 for j in cls_mask if all_preds[j] == i)
        cls_total = len(cls_mask)
        print(f"  {name}: {cls_correct}/{cls_total} = {cls_correct/cls_total:.4f}")