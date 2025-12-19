# file: `confidence/learned/svm.py`
import torch
import numpy as np
from typing import Optional, Union
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from confidence.base_confidence import ConfidenceModule
from confidence.input_transform import InputTransform

class BinarySVMPredictiveConfidence(ConfidenceModule):
    def __init__(
        self,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        degree: int = 3,
        coef0: float = 0.0,
        C: float = 1.0,
        neg_sampling: str = 'gaussian',
        n_neg_samples: Optional[int] = None,
        neg_sampling_margin: float = 0.1,  # fraction to expand uniform range
        input_transform: Optional[InputTransform] = None,
    ):
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.neg_sampling = neg_sampling
        self.n_neg_samples = n_neg_samples
        self.neg_sampling_margin = neg_sampling_margin
        self.input_transform = input_transform
        self.clf: Optional[SVC] = None
        self.fitted = False

    def _extract_data_from_loader(self, data_loader: DataLoader) -> tuple:
        """Extract positive and negative samples from a DataLoader with boolean labels."""
        X_pos_list = []
        X_neg_list = []
        
        for batch in data_loader:
            x, y = batch
            
            # Ensure y is boolean
            if not torch.is_tensor(y):
                y = torch.tensor(y, dtype=torch.bool)
            elif y.dtype != torch.bool:
                y = y.bool()
            
            # Split into positive and negative samples
            pos_mask = y
            neg_mask = ~pos_mask
            
            if pos_mask.any():
                X_pos_list.append(x[pos_mask])
                
            if neg_mask.any():
                X_neg_list.append(x[neg_mask])
        
        # Concatenate all positive samples
        X_pos = torch.cat(X_pos_list) if X_pos_list else None
        
        # Concatenate all negative samples if any exist
        X_neg = torch.cat(X_neg_list) if X_neg_list else None
        
        if X_pos is None:
            raise ValueError("No positive samples found in DataLoader")
            
        return X_pos, X_neg

    def fit(self, data):
        """
        Fit SVM on data. Can accept:
        - A DataLoader with (x, y) batches where y is a boolean tensor
        - A tensor X_pos (and optionally X_neg) for backward compatibility
        """
        if isinstance(data, DataLoader):
            X_pos, X_neg = self._extract_data_from_loader(data)
            # If X_neg was explicitly passed, it overrides the one from DataLoader

        else:
            # Backward compatibility: data is X_pos tensor
            X_pos, X_neg = data
            
        Xp = X_pos.detach().cpu().numpy()
        if self.input_transform:
            Xp = self.input_transform.fit(torch.from_numpy(Xp)).numpy()
        
        # Generate negatives if needed
        if X_neg is None:
            Xn = self._sample_negatives(Xp)
        else:
            Xn = X_neg.detach().cpu().numpy()
            if self.input_transform:
                Xn = self.input_transform.transform(torch.from_numpy(Xn)).numpy()
        
        X = np.vstack([Xp, Xn])
        y = np.hstack([np.ones(len(Xp)), np.zeros(len(Xn))])
        
        self.clf = SVC(kernel=self.kernel, gamma=self.gamma,
                      degree=self.degree, coef0=self.coef0,
                      C=self.C)
        self.clf.fit(X, y)

        # extract and store boundary tensors for torch computation
        sv = torch.from_numpy(self.clf.support_vectors_).float()
        coef = torch.from_numpy(self.clf.dual_coef_[0]).float()
        intercept = float(self.clf.intercept_[0])
        self.register_buffer('support_vectors', sv)
        self.register_buffer('dual_coef', coef)
        self.intercept = intercept
        # compute gamma
        if self.gamma == 'scale':
            self._gamma = 1.0 / (X.shape[1] * X.var() + 1e-12)
        elif self.gamma == 'auto':
            self._gamma = 1.0 / X.shape[1]
        else:
            self._gamma = float(self.gamma)
        self.fitted = True
        return self

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Call fit() before forward().")
        # reshape and compute kernel manually
        orig = x.shape[:-1]
        flat = x.reshape(-1, x.size(-1))
        if self.kernel == 'rbf':
            dif = flat.unsqueeze(1) - self.support_vectors.unsqueeze(0)
            K = torch.exp(-self._gamma * dif.pow(2).sum(-1))
        elif self.kernel == 'linear':
            K = flat @ self.support_vectors.t()
        elif self.kernel == 'poly':
            K = (self._gamma * (flat @ self.support_vectors.t()) + self.coef0).pow(self.degree)
        elif self.kernel == 'sigmoid':
            K = torch.tanh(self._gamma * (flat @ self.support_vectors.t()) + self.coef0)
        else:
            raise NotImplementedError(f"Kernel '{self.kernel}' not supported")
        scores = K.matmul(self.dual_coef) + self.intercept
        return scores.reshape(*orig)
