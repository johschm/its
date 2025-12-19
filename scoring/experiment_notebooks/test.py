import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from escnn import gspaces
from escnn import nn as enn


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from escnn import gspaces
from escnn import nn as enn

class EquivariantRotationRegressor(nn.Module):
    """
    Predicts the rotation angle from input images, assuming rotations are in multiples of 360/N.
    Uses an SO(2)_N-equivariant backbone and a classification head over N angles.
    """
    def __init__(self, N=4, in_channels=1, channels=32):
        super().__init__()
        # Discrete rotation group with N elements (e.g. N=4 for 0,90,180,270 degrees)
        self.N = N
        gspace = gspaces.rot2dOnR2(N=N)

        # Input and hidden feature types
        in_type  = enn.FieldType(gspace, in_channels * [gspace.trivial_repr])
        hid_type = enn.FieldType(gspace, channels    * [gspace.regular_repr])

        # Equivariant backbone
        self.net = enn.SequentialModule(
            enn.R2Conv(in_type,  hid_type, kernel_size=7, padding=3, bias=False),
            enn.InnerBatchNorm(hid_type), enn.ReLU(hid_type),
            enn.R2Conv(hid_type, hid_type, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(hid_type), enn.ReLU(hid_type),
            enn.PointwiseAvgPool(hid_type, kernel_size=28),  # global
        )
        # Classification head: invariant map to N logits
        out_type = enn.FieldType(gspace, channels * [gspace.trivial_repr])  # trivial repr outputs invariant channels
        self.classifier = enn.SequentialModule(
            enn.R2Conv(hid_type, out_type, kernel_size=1, bias=True),
            # outputs [B, channels,1,1] trivial features
        )
        # Final linear layer on pooled invariants
        self.fc = nn.Linear(channels, N)

    def forward(self, x):
        # x: [B,1,28,28]
        geom = enn.GeometricTensor(x, self.net.in_type)
        y = self.net(geom)
        y_inv = self.classifier(y)       # GeometricTensor of trivial reps
        # flatten to [B, channels]
        feats = y_inv.tensor.view(y_inv.tensor.size(0), -1)
        logits = self.fc(feats)          # [B, N]
        # predicted angle in degrees
        idx = logits.argmax(dim=1)      # [B]
        angles = idx.to(torch.float) * (360.0 / self.N)
        return angles, logits

# ---- Test on rotated MNIST ----
if __name__ == '__main__':
    # Prepare MNIST and rotations
    base_transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST('.', train=False, download=True, transform=base_transform)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    imgs, _ = next(iter(loader))  # [4,1,28,28]

    # Generate rotations by multiples of 90 degrees
    rotated_imgs = []

    # concatenated batch
    batch = torch.cat([imgs], dim=0)  # [4,1,28,28]


    # Load model (random weights) and predict
    model = EquivariantRotationRegressor(N=4).eval()
    with torch.no_grad():
        pred_angles, logits = model(batch)
        batch_rotated = transforms.functional.rotate(batch, 180, expand=True)
        pred_angles2, logits2 = model(batch_rotated)

    # Compute jumps between successive rotations
    diffs = pred_angles2 - pred_angles


    print("Predicted angles:", pred_angles)
    print("Predicted angles after rotation:", pred_angles2)
    print("Differences between successive rotations:", diffs)