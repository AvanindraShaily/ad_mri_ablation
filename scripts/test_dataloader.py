import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

data_dir = r"C:\Users\avash\OneDrive\Desktop\projects\ad_ablation_study\data\sachin_kumar_ad_dataset"

# Collect all images and labels
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
class_to_idx = {name: i for i, name in enumerate(class_names)}

all_paths = []
all_labels = []
for cls in class_names:
    cls_path = os.path.join(data_dir, cls)
    for img_name in os.listdir(cls_path):
        all_paths.append(os.path.join(cls_path, img_name))
        all_labels.append(class_to_idx[cls])

# Split: 70% train, 10% val, 20% test
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.125, stratify=train_val_labels, random_state=42)

# Print counts
print(f"Classes: {class_names}")
print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
for name, lbls in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
    counts = {class_names[i]: lbls.count(i) for i in range(len(class_names))}
    print(f"  {name}: {counts}")

# Quick batch test
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SimpleDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

loader = DataLoader(SimpleDataset(train_paths, train_labels, transform), batch_size=32, shuffle=True)
images, labels = next(iter(loader))
print(f"\nBatch shape: {images.shape}")
print(f"Labels: {labels.tolist()}")