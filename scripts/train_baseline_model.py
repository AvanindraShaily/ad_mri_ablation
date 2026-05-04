import os
import sys
import time
import torch
import torch.nn as nn

# Add scripts folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from base_models import get_model
from dataset import load_dataset

# ---- Config ----
DATASET = "OASIS"               # "OASIS" or "Alzheimer (Preprocessed Data)"
MODE = "raw"                    # "raw" for 2D images, "wavelet" for DT-CWT features
MODEL_NAME = "resnet18"         # resnet18, resnet50, efficientnet_b0, mobilenet_v2, deit_tiny
USE_CBAM = False                # Toggle attention modules
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, DATASET)
WAVELET_DATA_DIR = os.path.join(RAW_DATA_DIR, "dtcwt_preprocessed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DATA_DIR = WAVELET_DATA_DIR if MODE == "wavelet" else RAW_DATA_DIR
IN_CHANNELS = 13 if MODE == "wavelet" else 3

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"{'='*50}")
print(f"Mode:        {MODE}")
print(f"Data dir:    {DATA_DIR}")
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

# ---- Model ----
if USE_CBAM:
    from models_with_cbam import get_model_with_attention
    model = get_model_with_attention(MODEL_NAME, num_classes=num_classes).to(DEVICE)
    print(f"Loaded {MODEL_NAME} WITH attention")
else:
    model = get_model(MODEL_NAME, num_classes=num_classes, in_channels=IN_CHANNELS).to(DEVICE)
    print(f"Loaded {MODEL_NAME} baseline (no attention)")

# ---- Class weights for imbalance ----
train_labels = train_loader.dataset.labels
class_counts = [train_labels.count(i) for i in range(num_classes)]
class_weights = torch.tensor([max(class_counts) / c for c in class_counts], dtype=torch.float).to(DEVICE)
print(f"Class counts: {dict(zip(class_names, class_counts))}")
print(f"Class weights: {class_weights.tolist()}")

# ---- Loss, optimizer, scheduler ----
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---- Build save name ----
suffix = f"_{MODE}"
if USE_CBAM:
    suffix += "_cbam"
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
        images, labels = images.to(DEVICE), torch.tensor(labels).to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
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
            images, labels = images.to(DEVICE), torch.tensor(labels).to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss /= val_total
    val_acc = val_correct / val_total
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"Time: {epoch_time:.1f}s")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"  -> Saved best model (val acc: {val_acc:.4f})")

# ---- Test evaluation ----
print(f"\n{'='*50}")
print("Testing best model...")
model.load_state_dict(torch.load(save_path, map_location=DEVICE))
model.eval()

test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), torch.tensor(labels).to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc:.4f}")

# Per-class accuracy
print("\nPer-class results:")
for i, name in enumerate(class_names):
    cls_mask = [j for j, l in enumerate(all_labels) if l == i]
    if cls_mask:
        cls_correct = sum(1 for j in cls_mask if all_preds[j] == i)
        cls_total = len(cls_mask)
        print(f"  {name}: {cls_correct}/{cls_total} = {cls_correct/cls_total:.4f}")