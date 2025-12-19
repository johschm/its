import torch
import torch.nn as nn
from torch_uncertainty.post_processing import MatrixScaler, TemperatureScaler,VectorScaler
import torch_uncertainty.post_processing.calibration.matrix_scaler as ms_module

# Override the internal scaling to use elementwise broadcast
def _scale(self, logits):
    # logits: (batch_size, num_classes)
    # temp_w, temp_b: (num_classes,)
    return logits * self.temp_w.unsqueeze(0) + self.temp_b.unsqueeze(0)

ms_module.MatrixScaler._scale = _scale

class MatrixScalingWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int,init_w= 1.0,init_b=0.0, max_iter: int = 100):
        super().__init__()
        self.scaler = MatrixScaler(num_classes=num_classes, model=model,
                                   init_w=init_w, init_b=init_b, max_iter=max_iter)

    def fit(self, calibration_dataset):
        self.scaler.fit(calibration_dataset)

    def forward(self, x):
        return self.scaler(x)


    def to(self, *args, **kwargs):
        """Override to ensure the model and scaler are moved to the same device."""
        super().to(*args, **kwargs)
        self.scaler.device = next(self.scaler.model.parameters()).device



class TemperatureScalingWrapper(nn.Module):
    def __init__(self, model: nn.Module, init_val: float = 1.0, lr: float = 0.1, max_iter: int = 100):
        super().__init__()
        self.scaler = TemperatureScaler(model=model, init_val=init_val, lr=lr, max_iter=max_iter)

    def fit(self, dataloader):
        self.scaler.fit(dataloader)

    def forward(self, x):
        return self.scaler(x)

    def to(self, *args, **kwargs):
        """Override to ensure the model and scaler are moved to the same device."""
        super().to(*args, **kwargs)
        self.scaler.device = next(self.scaler.model.parameters()).device




class VectorScalingWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int, lr: float = 0.1, max_iter: int = 100,init_w: float = 1.0, init_b: float = 0.0):

        super().__init__()
        self.model = model
        self.scaler = VectorScaler(model=model, num_classes=num_classes, lr=lr, max_iter=max_iter,
                                      init_w=init_w, init_b=init_b)

    def fit(self, dataloader):
        self.scaler.fit(dataloader)

    def forward(self, x):
        return self.scaler(x)


    def to(self, *args, **kwargs):
        """Override to ensure the model and scaler are moved to the same device."""
        super().to(*args, **kwargs)
        self.scaler.device = next(self.scaler.model.parameters()).device



if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn import Linear

    # Create dummy data: 100 samples, 10 features, 5 classes
    X = torch.randn(100, 10)
    y = torch.randint(0, 6, (100,))
    loader = DataLoader(TensorDataset(X, y), batch_size=20)

    # Dummy model mapping 10→5
    base_model = Linear(10, 6)

    # Test TemperatureScalingWrapper
    tsw = TemperatureScalingWrapper(model=base_model)
    tsw.fit(loader)
    out = tsw(torch.randn(4, 10))
    assert out.shape == (4, 6), "TemperatureScalingWrapper output shape mismatch"


    # Test VectorScalingWrapper
    vew = VectorScalingWrapper(model=base_model, num_classes=6)
    vew.fit(loader)
    out = vew(torch.randn(2, 10))
    assert out.shape == (2, 6), "VectorScalingWrapper output shape mismatch"


    # Test MatrixScalingWrapper
    msw = MatrixScalingWrapper(model=base_model, num_classes=6)
    msw.fit(loader)
    out = msw(torch.randn(3, 10))
    assert out.shape == (3, 6), "MatrixScalingWrapper output shape mismatch"




    print("All wrappers passed basic shape tests.")
