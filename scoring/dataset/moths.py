import os

from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image

class EUMoths(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        img_list_path = os.path.join(root, 'images.txt')
        label_list_path = os.path.join(root, 'labels.txt')
        split_list_path = os.path.join(root, 'tr_ID.txt')

        exclude_files = [
            "gwk19august2014_176_pareulype_berberea.jpg",
            "gwk28märz2016_069_biston_strataria.jpg",
            "gwk19september2014_316_c_atocala_nupta.jpg"
        ]

        self.samples = []
        with open(img_list_path, 'r') as f_img, \
             open(label_list_path, 'r') as f_label, \
             open(split_list_path, 'r') as f_split:

            for img, label, split in zip(f_img, f_label, f_split):
                img_path = img.strip().split()[1]  # Extract filename from second column
                label_val = int(label.strip())
                split_val = int(split.strip())

                if os.path.basename(img_path) in exclude_files:
                    continue

                # split != 0 means training, split == 0 means testing
                if (split_val != 0 and train) or (split_val == 0 and not train):
                    full_path = os.path.join(root, 'images', img_path)
                    self.samples.append((full_path, label_val))

        # Create label mapping to ensure contiguous labels starting from 0
        self.label_mapping = self.create_label_mapping()

    def create_label_mapping(self):
        labels = [label for _, label in self.samples]
        unique_labels = sorted(set(labels))
        return {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, original_label = self.samples[idx]

        image = PIL.Image.open(img_path).convert('RGB')
        label = self.label_mapping[original_label]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_num_classes(self):
        return len(self.label_mapping)


class Moths(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        img_list_path = os.path.join(root, 'imagelist.txt')
        label_list_path = os.path.join(root, 'labels.txt')
        split_list_path = os.path.join(root, 'tr_ID.txt')

        self.samples = []
        self.class_names = set()
        with open(img_list_path, 'r') as f_img, \
             open(label_list_path, 'r') as f_label, \
             open(split_list_path, 'r') as f_split:

            for img, label, split in zip(f_img, f_label, f_split):
                img_path = img.strip()
                label_val = int(label.strip())
                split_val = int(split.strip())

                # Filter by train/test split
                # split != 0 means training, split == 0 means testing
                if (split_val != 0 and train) or (split_val == 0 and not train):
                    full_path = os.path.join(root, img_path)
                    self.samples.append((full_path, label_val))
                self.class_names.add((label_val, self.get_class_name(img_path)))

        # Create label mapping to ensure contiguous labels starting from 0
        self.label_mapping = self.create_label_mapping()
        self.class_names = {self.label_mapping[i]: name for i, name in self.class_names}
        self.class_names = [self.class_names[i] for i in sorted(self.class_names)]

    def create_label_mapping(self):
        labels = [label for _, label in self.samples]
        return {old_label: new_label for new_label, old_label in enumerate(sorted(set(labels)))}

    def get_class_name(self, filename):
        name = os.path.splitext(filename)[0]
        return ' '.join(word.capitalize() for word in name.split('/')[0].split('_'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, original_label = self.samples[idx]
        image = PIL.Image.open(img_path).convert('RGB')

        label = self.label_mapping[original_label]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_num_classes(self):
        return len(self.label_mapping)


from torch.utils.data import Dataset
import os
import torch

class PersistentCachedDataset(Dataset):
    """
    Wraps a Dataset and persists its __getitem__ outputs to disk.
    On init it loads from cache_file if present, else preloads and saves.
    """
    def __init__(self, dataset, cache_file, preload=True):
        self.dataset = dataset
        self.cache_file = cache_file
        self._cache = {}
        if preload:
            self._load_or_build_cache()

    def _load_or_build_cache(self):
        if os.path.exists(self.cache_file):
            self._cache = torch.load(self.cache_file)
        else:
            for idx in range(len(self.dataset)):
                self._cache[idx] = self.dataset[idx]
            torch.save(self._cache, self.cache_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self._cache[idx]

    def clear_cache(self):
        """Remove in-memory and on-disk cache."""
        self._cache.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)