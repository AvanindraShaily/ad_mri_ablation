import os
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


def load_dataset(data_dir, test_size=0.2, val_size=0.1, seed=42, batch_size=32):
    """
    Load images, split into train/val/test with stratification,
    and return DataLoaders.
    """
    # Collect all image paths and labels
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_paths = []
    all_labels = []

    for cls_name in class_names:
        cls_path = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            all_paths.append(os.path.join(cls_path, img_name))
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

    # Create datasets
    train_dataset = AlzheimerDataset(train_paths, train_labels, get_transforms("train"))
    val_dataset = AlzheimerDataset(val_paths, val_labels, get_transforms("val"))
    test_dataset = AlzheimerDataset(test_paths, test_labels, get_transforms("test"))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Print summary
    print(f"Classes: {class_names}")
    print(f"Class mapping: {class_to_idx}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        counts = [split_labels.count(i) for i in range(len(class_names))]
        print(f"  {split_name} distribution: {dict(zip(class_names, counts))}")

    return train_loader, val_loader, test_loader, class_names