import copy
from typing import Callable, Optional
import torch
import torch.nn.functional as F
from optree import tree_map
from confidence.model.base_model import ModelBasedConfidence
from confidence.base_confidence import ConfidenceModule

# ----------------------------------------------------------
# Helper: KL to uniform loss
# ----------------------------------------------------------
def kl_to_uniform_loss(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(dim=-1)
    uniform = torch.full_like(p, 1.0 / p.size(-1))
    return F.kl_div(p.log(), uniform, reduction="batchmean")


# ----------------------------------------------------------
# GradNormConfidence (loop version)
# ----------------------------------------------------------
class GradNormConfidence(ModelBasedConfidence):
    """
    Computes per-sample gradient-norm scores as in the GradNorm paper.
    In-distribution → larger gradient norm; OOD → smaller.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        confidence: Optional[Callable] = None,
        param_filter: Optional[Callable[[str], bool]] = None,
        index: Optional[int] = None
    ):
        super().__init__(model, confidence or (lambda scores, y=None: scores[0]), index)
        self.param_filter = param_filter or (lambda name: True)

    def forward(self, x: torch.Tensor, y=None):
        device = next(self.model.parameters()).device
        x = x.to(device)
        x.requires_grad_(True)

        grad_norms, raw_outputs = [], []
        for xi in x:
            with torch.enable_grad():
                self.model.zero_grad()
                out = self.model(xi.unsqueeze(0))
                logits = out[self.index] if self.index is not None else out

                # KL divergence to uniform
                loss = kl_to_uniform_loss(logits)
                loss.backward()

                # compute L2 gradient norm
                total_norm = torch.tensor(0.0, device=device)
                for name, p in self.model.named_parameters():
                    if self.param_filter(name) and p.grad is not None:
                        total_norm += p.grad.detach().pow(2).sum()
                total_norm = total_norm.sqrt()

            grad_norms.append(total_norm)
            raw_outputs.append(out)

        grad_norms = torch.stack(grad_norms)
        batched_outputs = tree_map(lambda *t: torch.cat(t, dim=0), *raw_outputs)

        confidences = self.confidence((grad_norms, batched_outputs), y)
        return confidences, batched_outputs


# ----------------------------------------------------------
# FuncGradNormConfidence (torch.func/vmap version)
# ----------------------------------------------------------
try:
    from torch.func import vmap, grad, functional_call
except ImportError:
    raise ImportError("Requires PyTorch >= 2.0 for torch.func API")

class FuncGradNormConfidence(ConfidenceModule):
    """
    Vectorized GradNorm with torch.func and vmap.
    """
    def __init__(self, model, confidence=None, param_filter=None, index=None):
        super().__init__()
        self.model = model
        self.param_filter = param_filter or (lambda n: True)
        self.index = index
        self.confidence = confidence or (lambda scores, y=None: scores[0])

    def forward(self, x: torch.Tensor, y=None):
        device = next(self.model.parameters()).device
        x = x.to(device)
        x.requires_grad_(True)

        params = {n: p for n, p in self.model.named_parameters() if self.param_filter(n)}
        buffers = dict(self.model.named_buffers())

        def loss_and_output(curr_params, curr_buffers, xi):
            out = functional_call(self.model,
                                  {**curr_params, **curr_buffers},
                                  (xi.unsqueeze(0),))
            logits = out[self.index] if self.index is not None else out
            loss = kl_to_uniform_loss(logits)
            return loss, out.squeeze(0)

        grad_fn = grad(loss_and_output, argnums=0, has_aux=True)

        with torch.enable_grad():
            grads_per_sample, outputs_per_sample = vmap(
                grad_fn, in_dims=(None, None, 0)
            )(params, buffers, x)

        norms = torch.zeros(x.size(0), device=device)
        for name, g in grads_per_sample.items():
            norms += g.flatten(start_dim=1).pow(2).sum(dim=1)
        norms = norms.sqrt()

        raw_outputs = outputs_per_sample
        confidences = self.confidence((norms, raw_outputs), y)
        return confidences, raw_outputs


# ----------------------------------------------------------
# BackpackGradNormConfidence (BackPACK version)
# ----------------------------------------------------------
try:
    from backpack import backpack, extensions, extend
except ImportError:
    raise ImportError("Requires BackPACK library installed")

class BackpackGradNormConfidence(ConfidenceModule):
    """
    Computes per-sample gradient norms via BackPACK.
    """
    def __init__(self, model, confidence=None, param_filter=None, index=None):
        super().__init__()
        self.model = extend(copy.deepcopy(model))
        self.confidence = confidence or (lambda scores, y=None: scores[0])
        self.param_filter = param_filter or (lambda name: True)
        self.index = index

    def forward(self, x: torch.Tensor, y=None):
        device = next(self.model.parameters()).device
        x = x.to(device)
        x.requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(x)
            logits_sel = logits[self.index] if self.index is not None else logits
            p = logits_sel.softmax(dim=-1)
            uniform = torch.full_like(p, 1.0 / p.size(-1))
            loss_per_sample = F.kl_div(p.log(), uniform, reduction="none").sum(dim=-1)

            with backpack(extensions.BatchGrad()):
                loss_per_sample.backward(torch.ones_like(loss_per_sample))

        norms = []
        for name, p in self.model.named_parameters():
            if self.param_filter(name):
                g = p.grad_batch
                norms.append(g.flatten(start_dim=1).pow(2).sum(dim=1))

        if len(norms) == 0:
            norms_tensor = torch.zeros(x.size(0), device=device)
        else:
            norms_tensor = torch.stack(norms, dim=1).sum(dim=1).sqrt()

        raw_outputs = logits
        confidences = self.confidence((norms_tensor, raw_outputs), y)
        return confidences, raw_outputs
