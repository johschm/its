
from typing import List, TypeVar, Optional, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase

Self = TypeVar("Self")

class FeatureGramTorchConfidence(ClassicConfidenceBase):
    """
    Gram‐matrix feature‐only detector matching pytorch_ood.
    fit(features_list, y) computes per‐class min/max bounds.
    forward(features_list, preds) returns –Δ/50 exactly as in original.
    """

    def __init__(
        self,
        num_classes: int,
        num_poles_list: Optional[List[int]] = None,
        input_transform=None,
    ):
        super().__init__(input_transform=input_transform)
        self.num_classes = num_classes
        self.num_poles_list = num_poles_list
        self.num_layer: Optional[int] = None
        self.feature_min: Optional[Tensor] = None
        self.feature_max: Optional[Tensor] = None
        self.fitted = False

    def _fit(
        self,
        features_list: List[Tensor],
        y: Tensor=None
    ) -> "FeatureGramTorchConfidence":
        # prepare
        if isinstance(features_list, Tensor):
            features_list = [features_list]


        self.num_layer = len(features_list)
        if self.num_poles_list is None:
            self.num_poles_list = list(range(1, self.num_layer + 1))
        num_poles = len(self.num_poles_list)

        # placeholders
        feature_class = [
            [[None for _ in range(num_poles)] for _ in range(self.num_layer)]
            for _ in range(self.num_classes)
        ]
        mins = [
            [[None for _ in range(num_poles)] for _ in range(self.num_layer)]
            for _ in range(self.num_classes)
        ]
        maxs = [
            [[None for _ in range(num_poles)] for _ in range(self.num_layer)]
            for _ in range(self.num_classes)
        ]

        with torch.no_grad():
            labels = y.tolist()
            # collect
            for layer_idx, feats in enumerate(features_list):
                temp_feat = feats.detach()
                for pole_idx, p in enumerate(self.num_poles_list):
                    # compute Gram‐sum vector
                    t = temp_feat**p
                    t = t.reshape(t.shape[0], t.shape[1], -1)
                    t = (t @ t.transpose(1, 2)).sum(dim=2)
                    t = (t.sign() * t.abs() ** (1.0 / p)).reshape(t.shape[0], -1)
                    for vec, lbl in zip(t.tolist(), labels):
                        if feature_class[lbl][layer_idx][pole_idx] is None:
                            feature_class[lbl][layer_idx][pole_idx] = vec
                        else:
                            feature_class[lbl][layer_idx][pole_idx].extend(vec)
            print("Feature collection complete.")
            # compute per‐class min/max
            for lbl in range(self.num_classes):
                for layer_idx in range(self.num_layer):
                    for pole_idx in range(num_poles):
                        arr = torch.tensor(
                            feature_class[lbl][layer_idx][pole_idx]
                        )
                        cur_min = arr.min(dim=0, keepdim=True)[0]
                        cur_max = arr.max(dim=0, keepdim=True)[0]
                        if mins[lbl][layer_idx][pole_idx] is None:
                            mins[lbl][layer_idx][pole_idx] = cur_min
                            maxs[lbl][layer_idx][pole_idx] = cur_max
                        else:
                            mins[lbl][layer_idx][pole_idx] = torch.min(
                                mins[lbl][layer_idx][pole_idx], cur_min
                            )
                            maxs[lbl][layer_idx][pole_idx] = torch.max(
                                maxs[lbl][layer_idx][pole_idx], cur_max
                            )

        # store
        self.feature_min = torch.tensor(mins)
        self.feature_max = torch.tensor(maxs)
        self.fitted = True
        return self

    def _forward(
        self,
        features_list: Union[Tensor, List[Tensor]],
        y: Optional[Tensor] = None
    ) -> Tensor:
        if isinstance(features_list, Tensor):
            features_list = [features_list]

        if not self.fitted:
            raise RuntimeError("Call fit() before forward()")
        if y is None:
            raise ValueError("Predicted labels 'y' required at inference")

        device = y.device
        preds = y
        fm = self.feature_min.to(device)
        fx = self.feature_max.to(device)
        fm_p = (fm + 1e-6).abs()
        fx_p = (fx + 1e-6).abs()

        N = features_list[0].shape[0]
        deviations = torch.zeros(N, device=device)

        for layer_idx, feats in enumerate(features_list):
            tfeat = feats.to(device)
            for pole_idx, p in enumerate(self.num_poles_list):
                t = tfeat**p
                t = t.reshape(t.shape[0], t.shape[1], -1)
                t = (t @ t.transpose(1, 2)).sum(dim=2)
                t = (t.sign() * t.abs() ** (1.0 / p)).reshape(t.shape[0], -1)
                t_sum = t.sum(dim=1)

                min_norm = fm_p[preds, layer_idx, pole_idx]
                max_norm = fx_p[preds, layer_idx, pole_idx]
                fmin = fm[preds, layer_idx, pole_idx]
                fmax = fx[preds, layer_idx, pole_idx]

                deviations += F.relu(fmin - t_sum) / min_norm
                deviations += F.relu(t_sum - fmax) / max_norm

        return -deviations / 50.0


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms
    from torch import nn
    from pytorch_ood.detector import Gram as OODGram

    # For reproducibility
    torch.manual_seed(0)

    # synthetic data
    N, D, C = 16, 8, 3
    X = torch.randn(N, D, 1, 1)
    y = torch.randint(0, C, (N,))

    # head must output C logits
    head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(D, C)
    )
    layers = [nn.Identity()]
    loader = DataLoader(TensorDataset(X, y), batch_size=40000)

    # --- Original Implementation ---
    orig = OODGram(head=head, feature_layers=layers, num_classes=C)
    orig.fit(loader, device="cpu")
    logits, feats = orig._create_feature_list(X)
    # Use the model's predictions for scoring
    preds = torch.argmax(logits, dim=1)

    # --- Your (Optimized) Implementation ---
    feat = FeatureGramTorchConfidence(num_classes=C)
    feat.fit(feats, y)

    print("\n--- Diagnostics ---")
    print("orig.feature_min shape:", orig.feature_min.shape)
    print("feat.feature_min shape:", feat.feature_min.shape)
    print("orig.feature_max shape:", orig.feature_max.shape)
    print("feat.feature_max shape:", feat.feature_max.shape)

    cls, layer_idx, pole_idx = 0, 0, 0
    print(f"\norig.feature_min[{cls},{layer_idx},{pole_idx}]:", orig.feature_min[cls, layer_idx, pole_idx])
    print(f"feat.feature_min[{cls},{layer_idx},{pole_idx}]:", feat.feature_min[cls, layer_idx, pole_idx])

    print("\nfeature_min match per class/layer/pole:",
          torch.allclose(orig.feature_min.float(), feat.feature_min.float(), atol=1e-5))
    print("feature_max match per class/layer/pole:",
          torch.allclose(orig.feature_max.float(), feat.feature_max.float(), atol=1e-5))

    scores_orig = orig.predict_features(logits, feats)
    scores_new = feat(feats, preds)

    print("Feature Gram scores:", scores_new.to(torch.float32))
    print("Original Gram scores:", scores_orig.to(torch.float32))

    print("Batch match:", torch.allclose(scores_new.to(torch.float32), scores_orig.to(torch.float32), atol=1e-5))