import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def strokes_to_sequence(drawing, max_len=200):
    seq = drawing[:max_len]
    seq += [[0, 0, 0]] * (max_len - len(seq))
    return np.array(seq, dtype=np.float32)

class SketchDataset(Dataset):
    def __init__(self, hf_ds, max_len=200):
        self.ds = hf_ds
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        seq = strokes_to_sequence(item["drawing"], self.max_len)
        label = item["label"]  # adjust field name as needed
        return torch.from_numpy(seq), label


class NormalizeToRangeBatched(nn.Module):
    """
    Option 1: Normalize absolute coordinates (largest coordinate magnitude = 1)
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, stroke_seq: torch.Tensor) -> torch.Tensor:
        # stroke_seq: [B, N, 3] where last dim is (dx, dy, pen_state)
        abs_xy = torch.cumsum(stroke_seq[..., :2], dim=1)           # [B, N, 2]
        center = abs_xy.mean(dim=1, keepdim=True)                   # [B, 1, 2]
        centered = abs_xy - center                                  # [B, N, 2]

        max_abs = centered.abs().amax(dim=(1, 2), keepdim=True)     # [B, 1, 1]
        scale = 1.0 / (max_abs + self.eps)                          # [B, 1, 1]
        scaled = centered * scale
        scaled = scaled*128.0  # Scale to [-128, 128] range


        # Convert scaled absolute coordinates back to deltas
        rel_xy = torch.zeros_like(scaled)
        rel_xy[:, 0] = scaled[:, 0]  # First point becomes the delta from origin
        rel_xy[:, 1:] = scaled[:, 1:] - scaled[:, :-1]              # [B, N, 2]

        pen_state = stroke_seq[..., 2:]                             # [B, N, 1]
        return torch.cat([rel_xy, pen_state], dim=-1)             # [B, N, 3]