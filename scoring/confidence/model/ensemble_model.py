import torch
import torch.nn as nn

class SimpleStackEnsemble(nn.Module):
    def __init__(self, modules: list[nn.Module]):
        super().__init__()
        self.subnets = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # run each module on x and stack along dim=1
        outputs = [m(x) for m in self.subnets]
        return torch.stack(outputs, dim=1)
