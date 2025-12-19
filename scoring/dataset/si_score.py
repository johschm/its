import os
import json
import urllib.request

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, models


class SIScoreDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.samples = []
        # Default transform: Resize to 224x224, ToTensor, Normalize (ImageNet)
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        self.transform = transform

        # Load official ImageNet class index (wnid -> 0-999)
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        class_idx = json.load(urllib.request.urlopen(url))
        self.wnid_to_idx = {v[1]: int(k) for k, v in class_idx.items()}

        # Only include classes actually present in the dataset folder
        self.classes = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d in self.wnid_to_idx
        ]

        self.class_to_idx = {cls_name: self.wnid_to_idx[cls_name] for cls_name in self.classes}

        # Collect (path, class_idx) pairs
        for cls_name in self.classes:
            cls_dir = os.path.join(root, cls_name)
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


from typing import Optional, Callable, Tuple, List
from pathlib import Path


class ImageNetSubset(Dataset):
    """
    PyTorch Dataset for ImageNet subset stored in class-labeled folders.

    Directory structure:
        root/
            0/
                0000.jpg
                0001.jpg
                ...
            1/
                ...
            ...
    """
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        # Default transform: same ImageNet preprocessing (224x224 + Normalize)
        if transform is None:
            transform = models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        self.transform = transform

        # Gather all image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        for class_dir in sorted(self.root.iterdir()):
            if class_dir.is_dir():
                label = int(class_dir.name)
                # Accept common image types, not just .jpg
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label