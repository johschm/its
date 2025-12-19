import torch
import torch.nn.functional as F
from optree import tree_map
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import pytorch_lightning as pl
from typing import Optional, Union, Callable, List, Tuple, Any, Literal
import random
from confidence.input_transform import InputTransform




def _get_at_path(tree: Any, path: Tuple[Union[int, str], ...]) -> Any:
    for key in path:
        tree = tree[key]
    return tree

def _set_at_path(tree: Any, path: Tuple[Union[int, str], ...], value: Any) -> Any:
    if not path:
        return value
    key, *rest = path
    if isinstance(tree, tuple):
        lst = list(tree)
        lst[key] = _set_at_path(lst[key], rest, value)
        return tuple(lst)
    if isinstance(tree, list):
        tree[key] = _set_at_path(tree[key], rest, value)
        return tree
    if isinstance(tree, dict):
        tree[key] = _set_at_path(tree[key], rest, value)
        return tree
    return tree

class BatchNegativeSampler(torch.nn.Module):
    def __init__(
        self,
        strategy,
        x_index: Union[int, Tuple[Union[int, str], ...]] = 0,
        y_index: Union[int, Tuple[Union[int, str], ...]] = 1,
        negative_value: Union[int, float] = -1,
        transform_true_function=None,
        augment_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decision_strategy=None,
        number_of_negatives: int = 1,
        return_params: bool = False,
    ):
        super().__init__()
        self.strategy = strategy
        self.x_index = (x_index,) if isinstance(x_index, int) else tuple(x_index)
        self.y_index = (y_index,) if isinstance(y_index, int) else tuple(y_index)
        self.negative_value = negative_value
        self.transform_true_function = transform_true_function
        self.augment_function = augment_function
        self.decision_strategy = decision_strategy or None
        self.number_of_negatives = number_of_negatives
        self.return_params = return_params


    @torch.no_grad()
    def forward(self, batch: Any,seed=None):
        if isinstance(batch, torch.Tensor):
            raise ValueError("BatchNegativeSampler expects a structured batch, not a simple tensor.")
        X = _get_at_path(batch, self.x_index)
        batch_size = X.size(0)
        y = _get_at_path(batch, self.y_index)

        # ---- NEW: handle zero negatives early ----
        if self.number_of_negatives == 0:
            X_pos = self.transform_true_function(X) if self.transform_true_function else X
            both = X_pos
            modified_batch = _set_at_path(batch, self.x_index, both)

            if y is not None:
                modified_y = y
                if self.decision_strategy is not None:
                    in_dist = modified_y >= 0
                    ref_y = y
                    modified_y = self.decision_strategy(both, in_dist, ref_y)
                modified_batch = _set_at_path(modified_batch, self.y_index, modified_y)

            if self.return_params:
                T_all = self.strategy.identity_transform(X.size(0))
                return modified_batch, T_all
            return modified_batch
        # ---- END zero-negatives branch ----

        if self.return_params:
            # sample_and_params must return (samples, transform_matrix)
            sampled = [self.strategy.sample_and_params(X,seed=seed) for _ in range(self.number_of_negatives)]
            X_neg_list, T_neg_list = zip(*sampled)  # tuples of tensors
            X_neg = torch.cat(X_neg_list, dim=0)
            # Identity (positive) transform
            T_pos = self.strategy.identity_transform(batch_size)
            # Concatenate transforms_old so we return one tensor, not a tuple
            T_all = torch.cat([T_pos, *T_neg_list], dim=0)
        else:
            X_neg = torch.cat([self.strategy.sample(X,seed=seed) for _ in range(self.number_of_negatives)], dim=0)

        X_pos = self.transform_true_function(X) if self.transform_true_function else X
        both = torch.cat([X_pos, X_neg], dim=0)  # positives first

        if self.augment_function is not None:
            both = self.augment_function(both)

        modified_batch = _set_at_path(batch, self.x_index, both)

        if y is not None:
            y_neg = torch.full(
                (batch_size * self.number_of_negatives, *y.shape[1:]),
                self.negative_value,
                device=y.device,
                dtype=y.dtype,
            )
            modified_y = torch.cat([y, y_neg], dim=0)

            if self.decision_strategy is not None:
                in_dist = modified_y >= 0
                ref_y = torch.cat([y] * (1 + self.number_of_negatives), dim=0)
                modified_y = self.decision_strategy(both, in_dist, ref_y)

            modified_batch = _set_at_path(modified_batch, self.y_index, modified_y)

        if self.return_params:
            # Return transforms_old alongside modified batch
            return modified_batch, T_all
        return modified_batch


class PrecomputedNegativeSamplingDataset(Dataset):
    def __init__(
        self,
        X_pos: torch.Tensor,
        strategy,
    ):
        """
        Samples all negatives once. Then __getitem__ returns (x, y) for positives
        first, then negatives. Default transform is identity.
        """
        self.device = X_pos.device

        # positives
        self.X_pos = X_pos
        self.y_pos = torch.ones(len(X_pos), device=self.device)

        # precomputed negatives
        self.X_neg = strategy.sample(X_pos)
        self.y_neg = torch.zeros(len(X_pos), device=self.device)


        # total length = pos + neg
        self.n_pos = len(self.X_pos)

    def __len__(self):
        return 2 * self.n_pos

    def __getitem__(self, idx):
        if idx < self.n_pos:
            return self.X_pos[idx], self.y_pos[idx]
        else:
            i = idx - self.n_pos
            return self.X_neg[i], self.y_neg[i]



class PreloadedDataset(Dataset):
    """Dataset with preloaded positive and negative samples and their function values"""

    def __init__(self,
                 X_pos: torch.Tensor,
                 X_neg: torch.Tensor,
                 y_pos: Optional[torch.Tensor] = None,
                 y_neg: Optional[torch.Tensor] = None,
                 input_transform: Optional[InputTransform] = None):
        """
        Args:
            X_pos: Tensor of positive examples
            X_neg: Tensor of negative examples
            y_pos: Optional function values for positive examples (default: ones)
            y_neg: Optional function values for negative examples (default: zeros)
            input_transform: Optional transform to apply to inputs
        """
        self.X_pos = X_pos
        self.X_neg = X_neg

        # Use default values if not provided
        if y_pos is None:
            y_pos = torch.ones(len(X_pos), device=X_pos.device)
        if y_neg is None:
            y_neg = torch.zeros(len(X_neg), device=X_neg.device)

        self.y_pos = y_pos
        self.y_neg = y_neg

        # Combine data for easier indexing
        self.X_all = torch.cat([X_pos, X_neg], 0)
        self.y_all = torch.cat([y_pos, y_neg], 0)
        self.is_positive = torch.cat([
            torch.ones(len(X_pos), device=X_pos.device, dtype=torch.bool),
            torch.zeros(len(X_neg), device=X_neg.device, dtype=torch.bool)
        ])

        # Store indices for balanced sampling
        self.pos_indices = list(range(len(X_pos)))
        self.neg_indices = list(range(len(X_pos), len(X_pos) + len(X_neg)))

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        x = self.X_all[idx]
        return x, self.y_all[idx]


