import os

import torch
from torch import nn

from model.classifier import Classifier, MyProgressBar
from utils.transforms.apply import grid_resample
import pytorch_lightning as pl



def load_model_mnist(model_path,train_loader, val_loader,overwrite=False):
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: 32x28x28
        nn.BatchNorm2d(32),  # Batch Normalization
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x14x14
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64x14x14
        nn.BatchNorm2d(64),  # Batch Normalization
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64x7x7
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        torch.nn.Dropout(0.0),  # Dropout layer with 50% dropout rate
        nn.GELU(),
        nn.Linear(128, 10)
    )

    # Check if model is already trained
    if os.path.exists(model_path) and not overwrite:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Training model and will save to {model_path}")
        lightning_model = Classifier(model, optimizer_class=torch.optim.AdamW, optimizer_params={"lr": 1e-3})
        progress_bar = MyProgressBar()
        trainer = pl.Trainer(
            accelerator="cuda",
            max_epochs=20,
            precision="16-mixed",
            callbacks=[progress_bar],
        )
        # Train the model
        trainer.fit(lightning_model, train_loader, val_loader)
        # Test the model
        # trainer.test(lightning_model, test_loader)
        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    return model

from torchvision import models
class MNISTResNet(nn.Module):
    def __init__(self, num_classes=10, imagenet_mean=None, imagenet_std=None,pretrained=True):
        super().__init__()
        # Load pretrained ResNet18 (keep conv1 unchanged)
        self.resnet = models.resnet18(pretrained=pretrained)
        # Replace the final fully-connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # ImageNet mean / std (default)
        if imagenet_mean is None:
            imagenet_mean = [0.485, 0.456, 0.406]
        if imagenet_std is None:
            imagenet_std = [0.229, 0.224, 0.225]

        mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(imagenet_std,  dtype=torch.float32).view(1, 3, 1, 1)

        # Register as buffers so they move with the model (cpu <-> gpu)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        """
        Expect x with shape (B, 1, H, W).
        Recommended input: torchvision.transforms_old.ToTensor() which gives float in [0,1].
        If you pass uint8 in [0,255], the model will convert it.
        """
        # If uint8 (0-255), convert to float and scale to [0,1]
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)

        # Duplicate grayscale channel to 3 channels
        # using expand avoids an extra copy when possible; make contiguous if needed
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        elif x.shape[1] == 3:
            pass  # already RGB
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

        # Resize to 224x224 (what the pretrained ResNet expects)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize using ImageNet mean/std
        x = (x - self.mean) / self.std

        return self.resnet(x)


def load_model_mnist_resnet(model_path, train_loader, val_loader,pretrained=True):
    model = MNISTResNet(pretrained=pretrained)

    # Check if model is already trained
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Training model and will save to {model_path}")
        lightning_model = Classifier(model, optimizer_class=torch.optim.AdamW, optimizer_params={"lr": 1e-3})
        progress_bar = MyProgressBar()
        trainer = pl.Trainer(
            accelerator="cuda",
            max_epochs=10,
            precision="16-mixed",
            callbacks=[progress_bar],
        )
        # Train the model
        trainer.fit(lightning_model, train_loader, val_loader)
        # Test the model
        # trainer.test(lightning_model, test_loader)
        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model
