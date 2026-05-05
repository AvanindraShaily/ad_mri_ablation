import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


class AlzheimerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


class WaveletDataset(Dataset):
    """Dataset for loading DT-CWT preprocessed .npy wavelet features.
    
    Each .npy file has shape (13, H, W) — 12 highpass subbands + 1 lowpass.
    Optionally resizes spatial dims to `target_size` for model compatibility.
    """
    def __init__(self, npy_paths, labels, target_size=224):
        self.npy_paths = npy_paths
        self.labels = labels
        self.target_size = target_size

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        data = np.load(self.npy_paths[idx])  # (13, H, W)
        tensor = torch.from_numpy(data)  # already float32

        # Resize to target_size if needed
        _, h, w = tensor.shape
        if h != self.target_size or w != self.target_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        label = self.labels[idx]
        return tensor, label


def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def load_dataset(data_dir, mode="raw", test_size=0.2, val_size=0.1, seed=42,
                 batch_size=32, target_size=224):
    if mode not in ("raw", "wavelet", "haar"):
        raise ValueError(f"mode must be 'raw', 'wavelet', or 'haar', got '{mode}'")

    print(f"\nLoading dataset in '{mode}' mode from: {data_dir}")

    # Collect all file paths and labels
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and d not in ("dtcwt_preprocessed", "haar_preprocessed")
    ])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_paths = []
    all_labels = []

    ext = ".npy" if mode in ("wavelet", "haar") else None
    for cls_name in class_names:
        cls_path = os.path.join(data_dir, cls_name)
        for fname in sorted(os.listdir(cls_path)):
            if ext and not fname.endswith(ext):
                continue
            all_paths.append(os.path.join(cls_path, fname))
            all_labels.append(class_to_idx[cls_name])

    # First split: separate out test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=seed
    )

    # Second split: separate validation from training
    val_fraction = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=seed
    )

    # Create datasets based on mode
    if mode == "raw":
        train_dataset = AlzheimerDataset(train_paths, train_labels, get_transforms("train"))
        val_dataset   = AlzheimerDataset(val_paths,   val_labels,   get_transforms("val"))
        test_dataset  = AlzheimerDataset(test_paths,  test_labels,  get_transforms("test"))
    else:  # wavelet
        train_dataset = WaveletDataset(train_paths, train_labels, target_size=target_size)
        val_dataset   = WaveletDataset(val_paths,   val_labels,   target_size=target_size)
        test_dataset  = WaveletDataset(test_paths,  test_labels,  target_size=target_size)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Print summary
    in_channels = 13 if mode == "wavelet" else (6 if mode == "haar" else 3)

    print(f"Mode: {mode} | Input channels: {in_channels}")
    print(f"Classes: {class_names}")
    print(f"Class mapping: {class_to_idx}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        counts = [split_labels.count(i) for i in range(len(class_names))]
        print(f"  {split_name} distribution: {dict(zip(class_names, counts))}")

    return train_loader, val_loader, test_loader, class_names