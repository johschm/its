import os

import torch
from PIL import Image

from utils.transforms.apply import grid_resample
#TODO for some reason using adjusted_grid_resample(only during search) decreases the performance during search but why?

import torch
from utils.transforms.apply import grid_resample

class AdjustedGridResample:
    def __init__(self, background_color):
        # store background as float tensor
        self.bg_tensor = torch.tensor(background_color, dtype=torch.float32)
    def __call__(self, image, transformation):
        # center, resample, then add background back
        bg = self.bg_tensor.to(image.device).view(3, 1, 1)
        centered = image - bg
        out = grid_resample(centered, transformation)
        return out + bg


class COIL100Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for filename in os.listdir(root_dir):
            if filename.endswith('.png'):
                self.image_paths.append(os.path.join(root_dir, filename))
                # Extract label from filename, e.g., "obj23__0.png" -> 23
                label = int(filename.split('__')[0].replace('obj', '')) -1
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        #remove 1 pixel from border as these contain a single yellow pixel which maybe could be used for giving information about position.
        image = image.crop((1, 1, image.width - 1, image.height - 1))  # Crop 1 pixel from each side
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label