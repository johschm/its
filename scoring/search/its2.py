import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from confidence.direct.logit_based import EnergyConfidence
from confidence.model.single_pass import SinglePassConfidence
from its.search import gaussian_filter1d_channel_wise, gaussian_filter1d, curvature, highlight_subplot
from its.transform import orbit_sampling, identity
from utils.transformation_problem import TransformationProblem
from utils.transforms.rotation import Rotation2D
from utils.transforms.scale import UniformScale2D

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from utils.transformation_problem import TransformationProblem
from utils.transform_sequence_discrete import TransformSequenceDiscrete


class InverseTransformationSearch2:
    """
    Discrete inverse transformation search using TransformSequenceDiscrete.
    Mirrors original ITS logic:
      - Layer 0: evaluate orbit once (B * O_0); spawn K hypotheses (top-k or unique-class masking).
      - Later layers: evaluate each hypothesis' orbit independently (B * K * O_l) and select one index per hypothesis.
      - Scores history structured like original: list over layers of (B,K,O_l); layer 0 replicated over hypotheses.
      - Change-of-mind (score) identical: sum over per-layer max scores.
    Differences:
      - Orbits assumed to have identity at middle index ( (L-1)//2 ).
    """

    def __init__(self,
                 model,
                 n_hypotheses=1,
                 mc_steps=1,
                 change_of_mind='score',
                 en_unique_class_condition=False,
                 labels=None,
                 line_thickness=1,
                 fontsize=16,
                 confidence_module=None,
                 gaussian_filter_channel_wise=False,
                 extend: int = 0,
                 discrete_n_samples: int = 9):  # can also be list of int i think?
        assert change_of_mind in ['score', 'off'], "Only 'score' and 'off' supported."
        self.model = model
        self.n_hypotheses = n_hypotheses
        self.mc_steps = mc_steps
        self.change_of_mind = change_of_mind
        self.en_unique_class_condition = en_unique_class_condition
        self.labels = np.arange(100) if labels is None else labels
        self.line_thickness = line_thickness
        self.fontsize = fontsize
        self.confidence_module = confidence_module
        self.gaussian_filter_channel_wise = gaussian_filter_channel_wise
        self.extend = extend
        self.discrete_n_samples = discrete_n_samples
        # runtime
        self.batch_size = None
        self.history = None

    # ---------- Helper parity functions (copied / adapted from original) ----------
    def _predict(self, x: torch.Tensor):
        if self.mc_steps == 1:
            self.model.eval()
            with torch.no_grad():
                return self.model(x)
        self.model.train()
        preds = []
        for _ in range(self.mc_steps):
            with torch.no_grad():
                preds.append(self.model(x))
        return torch.stack(preds, dim=0).mean(0)

    def _estimate_confidence(self, z: torch.Tensor):
        energies = torch.log(torch.exp(z).sum(dim=-1))
        if self.gaussian_filter_channel_wise:
            energies = gaussian_filter1d_channel_wise(energies, sigma=2, radius=3, mode='replicate') * 3
        else:
            energies = gaussian_filter1d(energies, sigma=2, radius=3, mode='replicate')
        return -curvature(energies.clone().detach().to(device=z.device))

    def _forward_confidence(self, x_batch: torch.Tensor):
        """
        Returns (score, logits) with shapes (N,) and (N,C).
        Accepts module outputs of form (score, logits) or logits only.
        Falls back to energy-based scoring if no confidence module.
        """
        if self.confidence_module is not None:
            out = self.confidence_module(x_batch)
            if isinstance(out, (list, tuple)):
                if len(out) == 2:
                    score, logits = out
                else:
                    score = out[0]
                    logits = None
            else:
                score = out
                logits = None

            # If logits are None, get them from the model
            if logits is None:
                logits = self._predict(x_batch)
        else:
            logits = self._predict(x_batch)
            score = self._estimate_confidence(logits)
        return score, logits

    # ---------- Selection (mirrors original select_candidates exactly) ----------
    def _select_candidates(self, score, pred_class, layer: int):
        """
        score: layer 0 -> (B,O); layer>0 -> (B,K,O)
        pred_class: same shape w/out last dim difference or None.
        Returns n_max: (B,K) chosen orbit indices for this layer.
        If pred_class is None, unique-class condition cannot be enforced and selection
        falls back to score-based top-k / argmax behavior.
        """
        B = score.shape[0]
        K = self.n_hypotheses

        # If unique-class enforcement requested but no pred_class available, fall back
        enforce_unique = self.en_unique_class_condition and (pred_class is not None)

        if not enforce_unique:
            if layer == 0:
                topk = torch.topk(score, k=min(K, score.shape[1]), dim=-1).indices  # (B, <=K)
                if topk.shape[1] < K:  # pad if orbit smaller than K
                    pad = topk[:, :1].expand(-1, K - topk.shape[1])
                    topk = torch.cat([topk, pad], dim=1)
                return topk
            else:
                return score.argmax(dim=2)  # (B,K)

        # unique-class condition (pred_class is available)
        n_max = torch.zeros((B, K), dtype=torch.int64, device=score.device)
        if layer == 0:
            # score: (B,O); pred_class: (B,O)
            sc = score.clone()
            pc = pred_class
            for h in range(K):
                idx = sc.argmax(-1)  # (B,)
                n_max[:, h] = idx
                chosen_cls = pc.gather(1, idx[:, None])  # (B,1)
                # mask all occurrences of chosen class
                mask = (pc == chosen_cls)
                sc = torch.where(mask, torch.full_like(sc, -torch.inf), sc)
        else:
            # score: (B,K,O); pred_class: (B,K,O)
            sc = score.clone()
            pc = pred_class
            for h in range(K):
                idx = sc[:, h].argmax(-1)  # (B,)
                n_max[:, h] = idx
                chosen_cls = pc[:, h].gather(1, idx[:, None])[:, None, :]  # (B,1,1)
                # broadcast across all hypotheses to enforce global uniqueness (match original)
                mask = (pc == chosen_cls)
                sc = torch.where(mask, torch.full_like(sc, -torch.inf), sc)
        return n_max

    @torch.no_grad()
    def optimize(self, transformation_problem, x: torch.Tensor, plot_idx=None, y=None):
        # Ensure discrete sequence
        transformation_problem = transformation_problem.ensure_discrete(self.discrete_n_samples)
        seq: TransformSequenceDiscrete = transformation_problem.transform_sequence

        B = x.shape[0]
        K = self.n_hypotheses
        L = seq.n_transforms()
        device = x.device
        self.batch_size = B
        self.history = {"score": [], "n_max": []}

        # Start with identity indices (middle of each orbit)
        mid = torch.tensor([(m - 1) // 2 for m in seq.orbit_lengths], device=device)
        indices = mid.view(1, 1, -1).expand(B, K, -1).clone()  # (B,K,L)
        e = self.extend

        # -------- Layer 0 --------
        core_len0 = seq.orbit_lengths[0]
        cand0_full = torch.arange(-e, core_len0 + e, device=device)

        # Create candidate indices for layer 0
        idx0 = indices[:, :1, :].expand(B, cand0_full.numel(), L).clone()
        idx0[:, :, 0] = cand0_full
        flat_idx0 = idx0.view(-1, L)

        # Transform images
        x0 = x.unsqueeze(1).expand(-1, cand0_full.numel(), -1, -1, -1).reshape(-1, *x.shape[1:])
        x0_t = seq.transform_with_extend(x0, flat_idx0, extend=e, layer=0)
        x0_t = x0_t.view(B, cand0_full.numel(), *x.shape[1:])

        # Extract core slice
        core_slice = slice(e, e + core_len0) if e > 0 else slice(0, core_len0)

        # Evaluate confidence
        flat_all0 = x0_t.reshape(B * cand0_full.numel(), *x.shape[1:])
        score0_all, logits0_all = self._forward_confidence(flat_all0)
        score0_all = score0_all.view(B, cand0_full.numel())
        logits0_all = logits0_all.view(B, cand0_full.numel(), -1) if logits0_all is not None else None

        # Extract core scores/logits
        score0 = score0_all[:, core_slice]
        logits0 = logits0_all[:, core_slice] if logits0_all is not None else None
        pred0 = logits0.argmax(-1) if logits0 is not None else None

        # Select candidates
        chosen0 = self._select_candidates(score0, pred0, layer=0)
        indices[:, :, 0] = chosen0

        # Store history
        self.history["score"].append(score0.unsqueeze(1).expand(-1, K, -1).detach().cpu().numpy())
        self.history["n_max"].append(chosen0.detach().cpu().numpy())

        # Update current hypotheses
        b_ar = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
        x_h = x0_t[:, core_slice][b_ar, chosen0]

        # Update logits for hypotheses
        if logits0 is not None:
            z_h = torch.gather(
                logits0,  # (B, core_len0, C)
                1,  # gather along core_len0 axis
                chosen0.unsqueeze(-1).expand(-1, -1, logits0.size(-1))  # (B,K,C)
            )
        else:
            z_h = None

        # -------- Subsequent layers --------
        for layer in range(1, L):
            core_len = seq.orbit_lengths[layer]
            cand_full = torch.arange(-e, core_len + e, device=device)
            Ncand = cand_full.numel()

            # Create candidate indices for this layer
            cand_idx = indices.unsqueeze(2).expand(B, K, Ncand, L).clone()
            cand_idx[:, :, :, layer] = cand_full.view(1, 1, -1)
            flat_idx = cand_idx.view(-1, L)

            # Transform images for all candidates
            x_expand = x_h.unsqueeze(2).expand(-1, -1, Ncand, -1, -1, -1).reshape(-1, *x.shape[1:])
            x_t_all = seq.transform_with_extend(x_expand, flat_idx, extend=e, layer=layer)
            x_t_all = x_t_all.view(B, K, Ncand, *x.shape[1:])

            # Extract core slice
            core_slice = slice(e, e + core_len) if e > 0 else slice(0, core_len)

            # Evaluate confidence
            flat_all = x_t_all.view(B * K * Ncand, *x.shape[1:])
            score_all, logits_all = self._forward_confidence(flat_all)
            score_all = score_all.view(B, K, Ncand)
            logits_all = logits_all.view(B, K, Ncand, -1) if logits_all is not None else None

            # Extract core scores/logits
            score_l = score_all[:, :, core_slice]
            logits_l = logits_all[:, :, core_slice] if logits_all is not None else None
            pred_l = logits_l.argmax(-1) if logits_l is not None else None

            # Select candidates
            chosen_l = self._select_candidates(score_l, pred_l, layer=layer)
            indices[:, :, layer] = chosen_l

            # Update current hypotheses
            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
            k_idx = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
            x_h = x_t_all[:, :, core_slice][b_idx, k_idx, chosen_l]

            # Update z_h with the logits from chosen candidates
            if logits_l is not None:
                z_h = logits_l[b_idx, k_idx, chosen_l]

            # Store history
            self.history["score"].append(score_l.detach().cpu().numpy())
            self.history["n_max"].append(chosen_l.detach().cpu().numpy())

        # -------- Final aggregation --------
        score_stack = torch.from_numpy(np.stack(self.history["score"])).to(device)
        agg = score_stack.max(-1).values.sum(0)

        if self.change_of_mind == 'score':
            order = agg.argsort(dim=-1, descending=True)
            gather_x = order[..., None, None, None].expand(-1, -1, *x_h.shape[-3:])
            x_h = torch.gather(x_h, 1, gather_x)
            if z_h is not None:
                gather_z = order[..., None].expand(-1, -1, z_h.shape[-1])
                z_h = torch.gather(z_h, 1, gather_z)
            gather_idx = order.unsqueeze(-1).expand(-1, -1, indices.shape[-1])
            self.final_indices = torch.gather(indices.detach(), dim=1, index=gather_idx)
        else:
            self.final_indices = indices.detach()

        best_param = self.final_indices[:, 0, :].clone()
        best_param = transformation_problem.transform_sequence.indices_to_param(best_param)

        best_error = (-agg[:, 0]).detach()
        best_classes = z_h.argmax(-1)[:, 0].detach() if z_h is not None else None

        return best_param, best_error, best_classes


def main():
    # 1) Define a small random model
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(3 * 8 * 8, 10)

        def forward(self, x):
            return self.fc(self.flatten(x))

    model = SmallModel()
    # instantiate transforms (instances, not classes)
    transforms = [Rotation2D, UniformScale2D]
    domains = [(-torch.pi, torch.pi), (0.5, 1.5)]
    n_samples = 5
    seq = TransformSequenceDiscrete(transforms, domains, n_samples)

    confidence = EnergyConfidence()
    confidence = SinglePassConfidence(model, confidence)
    problem = TransformationProblem(confidence, seq)

    its2 = InverseTransformationSearch2(model,
                                        n_hypotheses=2,
                                        mc_steps=1,
                                        change_of_mind='score',
                                        en_unique_class_condition=False,
                                        confidence_module=confidence,
                                        extend=1)

    x = torch.randn(2, 3, 8, 8)
    # fixed method name and safe print if logits are None
    x_t, z, logits = its2.optimize(problem, x)
    if logits is not None:
        print("ITS2 output:", x_t.shape, logits.shape)
    else:
        print("ITS2 output: images", x_t.shape, "logits: None")


if __name__ == "__main__":
    main()