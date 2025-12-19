import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import time
from typing import Optional, Callable, Union, Any, Literal

# Added import for KMeans
from sklearn.cluster import KMeans

from confidence.base_confidence import ConfidenceModule
from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase
from confidence.unsupervised.classic.nn_pytorch import PerClassKNNConfidence as PerClassKNNConfidencePyTorch
from confidence.unsupervised.classic.nn_pytorch import KNNConfidence as KNNConfidencePyTorch


class FaissKNN(torch.autograd.Function):
    """
    Custom autograd function to wrap Faiss KNN search and provide a backward pass.
    """
    @staticmethod
    def forward(ctx, query_vectors, train_vectors, k, index, metric='l2'):
        # Ensure inputs are contiguous float32 numpy arrays
        query_np = query_vectors.detach().cpu().numpy().astype(np.float32)
        
        # Perform the search
        distances, indices = index.search(query_np, k)

        # Convert to tensors
        distances_tensor = torch.from_numpy(distances).to(query_vectors.device, query_vectors.dtype)
        indices_tensor = torch.from_numpy(indices).to(query_vectors.device)

        # Save necessary tensors for backward pass
        ctx.save_for_backward(query_vectors, train_vectors, indices_tensor)
        ctx.metric = metric
        
        return distances_tensor, indices_tensor

    @staticmethod
    def backward(ctx, grad_distances, grad_indices):
        query_vectors, train_vectors, indices = ctx.saved_tensors
        metric = ctx.metric
        
        grad_query = torch.zeros_like(query_vectors)

        if grad_distances is None:
            return None, None, None, None, None

        # Get the top-k neighbors from the training data
        # Note: This assumes train_vectors contains the original, un-augmented vectors
        neighbors = train_vectors[indices] # Shape: (N, k, D)

        # Expand grad_distances for broadcasting
        grad_distances_expanded = grad_distances.unsqueeze(-1)

        # The query_vectors are augmented, but the gradient should only be for the original features.
        original_dim = neighbors.shape[-1]
        original_query_vectors = query_vectors[:, :original_dim]

        if metric == 'l2':
            # grad_distances is the gradient w.r.t. the squared L2 distance (d_sq).
            # We need dL/dq = dL/d_sq * d(d_sq)/dq
            # d(d_sq)/dq = d/dq (q-n)^2 = 2 * (q-n)
            diff = original_query_vectors.unsqueeze(1) - neighbors
            grad_query_k = 2 * grad_distances_expanded * diff
            grad_query_sum = grad_query_k.sum(dim=1)
            grad_query[:, :original_dim] = grad_query_sum
        
        elif metric == 'cosine':
            # grad_distances is the gradient w.r.t. cosine distance (1 - s).
            # dL/dq = dL/d(dist) * d(1-s)/ds * ds/dq = grad_distances * (-1) * ds/dq
            q_norm = torch.linalg.norm(original_query_vectors, dim=1, keepdim=True)
            n_norm = torch.linalg.norm(neighbors, dim=2, keepdim=True)
            
            # Recalculate similarity
            sim = (original_query_vectors.unsqueeze(1) * neighbors).sum(-1, keepdim=True) / (q_norm.unsqueeze(1) * n_norm + 1e-8)
            
            # Grad of cosine sim w.r.t query_vectors
            grad_sim_q = (neighbors / (q_norm.unsqueeze(1) * n_norm + 1e-8)) - \
                         (sim * original_query_vectors.unsqueeze(1) / (q_norm.pow(2).unsqueeze(1) + 1e-8))
            
            # dL/dq = dL/d(dist) * (-ds/dq)
            grad_query_k = -grad_sim_q * grad_distances_expanded
            grad_query_sum = grad_query_k.sum(dim=1)
            grad_query[:, :original_dim] = grad_query_sum

        return grad_query, None, None, None, None


def calculate_class_prototypes(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Calculate the class prototypes (means) for each class in the input tensor.

    Args:
        x: Input array of shape (num_samples, num_features).
        labels: Array of shape (num_samples,) containing class labels.

    Returns:
        Tensor of shape (num_classes, num_features) containing the mean of each class.
    """
    unique_labels = np.unique(labels)
    prototypes = []
    for label in unique_labels:
        prototypes.append(np.mean(x[labels == label], axis=0))
    return np.stack(prototypes)



def default_confidence_function(x: torch.Tensor) -> torch.Tensor:
    """
    Default confidence function that maps distances to confidence scores.
    This function can be replaced with a custom function if needed.

    Args:
        x: Input tensor of distances.

    Returns:
        Tensor of confidence scores.
    """

    #take mean over last dimension
    x = x.mean(dim=-1)
    #take median over last dimension
    #x = x.median(dim=-1).values
    return 1 / (1 + x)


class KNNConfidence(ClassicConfidenceBase):
    """
    Calculates confidence scores based on nearest neighbor distances using Faiss index.
    Supports 'euclidean' and 'cosine' metrics.
    """

    def __init__(
            self,
            metric: Literal['euclidean', 'cosine'] = 'euclidean',
            function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            number_of_neighbors: int = 3,
            index_type: Literal['flat', 'ivf', 'ivfpq', 'hnsw', 'ivf_hnsw'] = 'flat',
            use_gpu: bool = False,
            nlist: int = 100,
            hnsw_m: int = 32,
            code_size: int = 8,
            nbits: int = 8,
            nprobe: int = 5,
            input_transform: Optional[InputTransform] = None
    ):
        """
        Creates a new instance of the KNNConfidence class.
        """
        super().__init__(input_transform=input_transform)
        self.metric = metric
        self.index: Optional[Any] = None
        self.points: Optional[torch.Tensor] = None
        self.index_type: str = index_type.lower()
        self.use_gpu: bool = use_gpu
        self.nlist: int = nlist
        self.hnsw_m: int = hnsw_m
        self.code_size: int = code_size
        self.nbits: int = nbits
        self.nprobe: int = nprobe
        self.number_of_neighbors: int = number_of_neighbors

        self.confidence_function: Callable[[torch.Tensor], torch.Tensor] = (
            function if function is not None
            else default_confidence_function
        )

    def _fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> "KNNConfidence":
        self.points = x.clone().detach()
        
        if self.metric == 'cosine':
            data_torch = F.normalize(self.points, p=2, dim=1)
        else: # euclidean
            data_torch = self.points
            
        data: np.ndarray = data_torch.cpu().numpy().astype(np.float32)
        dim: int = data.shape[1]

        if self.metric == 'euclidean':
            metric_type = faiss.METRIC_L2
            if self.index_type == 'flat':
                index = faiss.IndexFlatL2(dim)
            elif self.index_type in ['ivf', 'ivfpq', 'ivf_hnsw']:
                quantizer = faiss.IndexFlatL2(dim) if self.index_type != 'ivf_hnsw' else faiss.IndexHNSWFlat(dim, self.hnsw_m)
                if self.index_type == 'ivfpq':
                    index = faiss.IndexIVFPQ(quantizer, dim, self.nlist, self.code_size, self.nbits)
                    index.nprobe = self.nprobe
                else:
                    index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, metric_type)
                if data.shape[0] > self.nlist:
                    index.train(data)
                else:
                    print(f"Warning: Not enough data for {self.index_type}. Falling back to flat index.")
                    index = faiss.IndexFlatL2(dim)
            elif self.index_type == 'hnsw':
                index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
        
        elif self.metric == 'cosine':
            metric_type = faiss.METRIC_INNER_PRODUCT
            if self.index_type == 'flat':
                index = faiss.IndexFlatIP(dim)
            elif self.index_type in ['ivf', 'ivfpq', 'ivf_hnsw']:
                quantizer = faiss.IndexFlatIP(dim) if self.index_type != 'ivf_hnsw' else faiss.IndexHNSWFlat(dim, self.hnsw_m, metric_type)
                if self.index_type == 'ivfpq':
                    index = faiss.IndexIVFPQ(quantizer, dim, self.nlist, self.code_size, self.nbits)
                    index.nprobe = self.nprobe
                else:
                    index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, metric_type)
                if data.shape[0] > self.nlist:
                    index.train(data)
                else:
                    print(f"Warning: Not enough data for {self.index_type}. Falling back to flat index.")
                    index = faiss.IndexFlatIP(dim)
            elif self.index_type == 'hnsw':
                index = faiss.IndexHNSWFlat(dim, self.hnsw_m, metric_type)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")

        if self.use_gpu:
            try:
                if faiss.get_num_gpus() > 0:
                    gpu_res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
                else:
                    print("Warning: GPU requested but not available, using CPU index")
            except Exception as e:
                print(f"Error moving index to GPU: {e}. Using CPU index.")

        self.index = index
        self.index.add(data)
        return self

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.index is None:
            raise ValueError("Index not built. Call fit() before forward().")

        if self.metric == 'euclidean':
            distances_sq, _ = FaissKNN.apply(x, self.points, self.number_of_neighbors, self.index, 'l2')
            distances = torch.sqrt(torch.clamp(distances_sq, min=0.0))
        elif self.metric == 'cosine':
            x_normalized = F.normalize(x, p=2, dim=1)
            similarities, _ = FaissKNN.apply(x_normalized, self.points, self.number_of_neighbors, self.index, 'cosine')
            distances = 1.0 - similarities
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return self.confidence_function(distances)


class PerClassKNNConfidence(KNNConfidence):
    """
    Per-class version of KNNConfidence.
    Finds nearest neighbors only within the same class by augmenting the feature space.
    Always uses an L2 index, converting distances for cosine metric.
    """
    def __init__(self, class_penalty: Optional[float] = None, debug_class_match: bool = False, **kwargs):
        # Force metric to euclidean for parent's index building logic, but we'll handle it
        super().__init__(**kwargs)
        self.class_penalty = class_penalty
        self.labels_unique = None
        self.debug_class_match = debug_class_match

    def _fit(self, x: torch.Tensor, y: torch.Tensor) -> "PerClassKNNConfidence":
        if y is None:
            raise ValueError("PerClassKNNConfidence requires labels 'y' for fitting.")

        self.register_buffer("train_labels", y.clone().detach()) # Store for debugging
        self.points = x.clone().detach() # Store original data for backward pass
        
        if self.metric == 'cosine':
            fit_data = F.normalize(x, p=2, dim=1)
            if self.class_penalty is None:
                self.class_penalty = 4.0 # A value > 2 is sufficient, use a larger margin.
                print(f"Using class penalty: {self.class_penalty}")
        else: # euclidean
            fit_data = x
            if self.class_penalty is None:
                sample_size = min(1000, x.shape[0])
                sample = x[torch.randperm(x.shape[0])[:sample_size]]
                max_dist = torch.cdist(sample, sample).max()
                self.class_penalty = max_dist.item() * 2
                print(f"Estimated class penalty: {self.class_penalty}")

        self.labels_unique, y_idx = torch.unique(y, return_inverse=True)
        
        y_scaled = y_idx.float().unsqueeze(1) * self.class_penalty
        augmented_x = torch.cat([fit_data, y_scaled.to(fit_data.device)], dim=1)
        
        # --- Build L2 Index (logic from parent) ---
        data: np.ndarray = augmented_x.cpu().numpy().astype(np.float32)
        dim: int = data.shape[1]
        
        if self.index_type == 'flat':
            index = faiss.IndexFlatL2(dim)
        elif self.index_type in ['ivf', 'ivfpq', 'ivf_hnsw']:
            quantizer = faiss.IndexFlatL2(dim) if self.index_type != 'ivf_hnsw' else faiss.IndexHNSWFlat(dim, self.hnsw_m)
            if self.index_type == 'ivfpq':
                index = faiss.IndexIVFPQ(quantizer, dim, self.nlist, self.code_size, self.nbits)
                index.nprobe = self.nprobe
            else:
                index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)
            if data.shape[0] > self.nlist:
                index.train(data)
            else:
                print(f"Warning: Not enough data for {self.index_type}. Falling back to flat index.")
                index = faiss.IndexFlatL2(dim)
        elif self.index_type == 'hnsw':
            index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        if self.use_gpu:
            try:
                if faiss.get_num_gpus() > 0:
                    gpu_res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
                else:
                    print("Warning: GPU requested but not available, using CPU index")
            except Exception as e:
                print(f"Error moving index to GPU: {e}. Using CPU index.")
        
        index.add(data)
        self.index = index
        return self

    def to(self, *args, **kwargs):
        # Ensure labels_unique and train_labels are moved to the correct device
        super().to(*args, **kwargs)
        if hasattr(self, 'train_labels') and self.train_labels is not None:
            self.train_labels = self.train_labels.to(*args, **kwargs)
        return self

    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y is None:
            raise ValueError("PerClassKNNConfidence requires labels 'y' for forward pass.")
        if self.index is None:
            raise ValueError("Index not built. Call fit() before forward().")
            
        y_idx = torch.searchsorted(self.labels_unique.to(y.device), y)
        y_scaled = y_idx.float().unsqueeze(1) * self.class_penalty

        if self.metric == 'cosine':
            query_data = F.normalize(x, p=2, dim=1)
            augmented_x = torch.cat([query_data, y_scaled.to(query_data.device)], dim=1)
            # Pass 'cosine' to FaissKNN. The forward pass will still use the L2 index,
            # but the backward pass will use the correct gradient logic for cosine similarity.
            sq_l2_dists, neighbor_indices = FaissKNN.apply(augmented_x, self.points, self.number_of_neighbors, self.index, 'cosine')
            # Convert squared L2 distance on normalized vectors back to cosine distance
            distances = sq_l2_dists / 2.0
        else: # euclidean
            augmented_x = torch.cat([x, y_scaled.to(x.device)], dim=1)
            sq_l2_dists, neighbor_indices = FaissKNN.apply(augmented_x, self.points, self.number_of_neighbors, self.index, 'l2')
            distances = torch.sqrt(torch.clamp(sq_l2_dists, min=0.0))

        if self.debug_class_match:
            # Move tensors to CPU for this debugging check to avoid device mismatches
            neighbor_labels = self.train_labels.cpu()[neighbor_indices.cpu()]
            query_labels_expanded = y.cpu().unsqueeze(1).expand(-1, self.number_of_neighbors)
            all_match = (neighbor_labels == query_labels_expanded).all()
            if not all_match:
                print("Warning: Mismatch found between query and neighbor classes!")
                for i in range(y.shape[0]):
                    if not (neighbor_labels[i] == query_labels_expanded[i]).all():
                        print(f"  Query {i} (label {y[i]}): found neighbor labels {neighbor_labels[i].tolist()}")
            assert all_match, "Neighbor class does not match query class!"
        
        return self.confidence_function(distances)


# ----------------- Added benchmarking helper -----------------
import math

def benchmark_knn(
    train_N: int = 50000,
    D: int = 512,
    batches: int = 100,
    batch_size: int = 128,
    k: int = 10,
    metric: str = 'euclidean',
    use_gpu_if_available: bool = True
):
    """
    Benchmark Faiss KNN (KNNConfidence) vs PyTorch KNN (KNNConfidencePyTorch)
    on synthetic data: train_N x D and `batches` of batch_size x D queries.
    Prints fit time, query time and throughput.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu_if_available else 'cpu')
    use_gpu_for_faiss = True if (faiss.get_num_gpus() if hasattr(faiss, 'get_num_gpus') else 0) > 0 and use_gpu_if_available else False

    print(f"\n--- KNN Benchmark: N={train_N}, D={D}, batches={batches}, batch_size={batch_size}, k={k}, metric={metric} ---")
    print(f"PyTorch device: {device}; Faiss GPU requested: {use_gpu_for_faiss}")

    # Generate data (keep on CPU for Faiss; PyTorch model will be moved to device)
    X_train = torch.randn(train_N, D, dtype=torch.float32)
    query_batches = [torch.randn(batch_size, D, dtype=torch.float32) for _ in range(batches)]

    # --- Faiss (KNNConfidence) ---
    faiss_model = KNNConfidence(
        metric=metric,
        number_of_neighbors=k,
        index_type='ivf',
        use_gpu=True
    )

    t0 = time.perf_counter()
    faiss_model.fit(X_train)  # builds index
    t_fit_faiss = time.perf_counter() - t0

    # Query timing (do not move queries to cuda because FaissKNN.apply converts to numpy CPU)
    #warmup
    _ = faiss_model(query_batches[0])  # returns confidence tensor
    t0 = time.perf_counter()
    for q in query_batches:
        _ = faiss_model(q)  # returns confidence tensor
    t_query_faiss = time.perf_counter() - t0

    # --- PyTorch (KNNConfidencePyTorch) ---
    # Move training data to device for GPU acceleration
    X_train_device = X_train.to(device)
    pyt_model = KNNConfidencePyTorch(k=k, metric=metric)
    pyt_model = pyt_model.to(device)

    t0 = time.perf_counter()
    pyt_model.fit(X_train_device)
    t_fit_pyt = time.perf_counter() - t0

    # Query timing (move each batch to device)
    #warmup
    _ = pyt_model(query_batches[0].to(device))
    t0 = time.perf_counter()
    for q in query_batches:
        q_dev = q.to(device)
        _ = pyt_model(q_dev)
    t_query_pyt = time.perf_counter() - t0

    total_queries = batches * batch_size
    print(f"\nFaiss fit time: {t_fit_faiss:.4f}s; Faiss query time (all batches): {t_query_faiss:.4f}s; throughput: {total_queries / t_query_faiss if t_query_faiss>0 else float('inf'):.1f} q/s")
    print(f"PyTorch fit time: {t_fit_pyt:.4f}s; PyTorch query time (all batches): {t_query_pyt:.4f}s; throughput: {total_queries / t_query_pyt if t_query_pyt>0 else float('inf'):.1f} q/s")

    print("\nSummary (seconds):")
    print(f"  Faiss: fit={t_fit_faiss:.4f}, query={t_query_faiss:.4f}")
    print(f"  PyTorch: fit={t_fit_pyt:.4f}, query={t_query_pyt:.4f}")


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    N_per = 500; D = 80
    mu0 = np.random.randn(D)
    mu1 = np.random.randn(D) + 5
    data0 = np.random.randn(N_per, D) + mu0
    data1 = np.random.randn(N_per, D) + mu1
    X = np.vstack([data0, data1])
    y = np.hstack([np.zeros(N_per), np.ones(N_per)])
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    
    X_test_np = np.vstack([mu0, mu1, (mu0 + mu1) / 2])
    X_test = torch.from_numpy(X_test_np).float()
    y_test = torch.tensor([0, 1, 0])

    print("--- Testing Per-Class Faiss L2 ---")
    per_class_l2 = PerClassKNNConfidence(metric='euclidean', number_of_neighbors=3, index_type='flat', debug_class_match=True)
    per_class_l2.fit(X_t, y_t)
    conf_l2 = per_class_l2(X_test, y_test)
    print("Confidences (L2):", conf_l2.tolist())

    print("\n--- Testing Per-Class Faiss Cosine ---")
    per_class_cos = PerClassKNNConfidence(metric='cosine', number_of_neighbors=3, index_type='flat', debug_class_match=True)
    per_class_cos.fit(X_t, y_t)
    conf_cos = per_class_cos(X_test, y_test)
    print("Confidences (Cosine):", conf_cos.tolist())

    print("\n--- Faiss vs PyTorch Implementation Comparison ---")
    from confidence.unsupervised.classic.nn_pytorch import PerClassKNNConfidence as PerClassKNNConfidencePyTorch

    # Euclidean
    model_torch_l2 = PerClassKNNConfidencePyTorch(k=3, metric='euclidean', computation_mode='per_class')
    model_torch_l2.fit(X_t, y_t)
    conf_torch_l2 = model_torch_l2(X_test, y_test)
    print("Faiss L2 Confidences:   ", conf_l2.tolist())
    print("PyTorch L2 Confidences: ", conf_torch_l2.tolist())
    print("L2 Difference:", torch.linalg.norm(conf_l2 - conf_torch_l2).item())

    # Cosine
    model_torch_cos = PerClassKNNConfidencePyTorch(k=3, metric='cosine', computation_mode='per_class')
    model_torch_cos.fit(X_t, y_t)
    conf_torch_cos = model_torch_cos(X_test, y_test)
    print("\nFaiss Cosine Confidences:   ", conf_cos.tolist())
    print("PyTorch Cosine Confidences: ", conf_torch_cos.tolist())
    print("Cosine Difference:", torch.linalg.norm(conf_cos - conf_torch_cos).item())


    print("\n--- Per-Class Neighbor Match Check ---")
    model_to_test = per_class_l2
    k = model_to_test.number_of_neighbors

    # Augment test data to query the index
    y_idx_test = torch.searchsorted(model_to_test.labels_unique.to(y_test.device), y_test)
    y_scaled_test = y_idx_test.float().unsqueeze(1) * model_to_test.class_penalty
    augmented_x_test = torch.cat([X_test, y_scaled_test.to(X_test.device)], dim=1)

    # Directly query the faiss index to get neighbor indices
    query_np = augmented_x_test.detach().cpu().numpy().astype(np.float32)
    _, neighbor_indices_np = model_to_test.index.search(query_np, k)

    # Get the labels of the found neighbors from the original training labels
    neighbor_labels = y_t[neighbor_indices_np]

    # Check if the neighbor labels match the query labels
    query_labels_expanded = y_test.unsqueeze(1).expand(-1, k)

    all_match = (neighbor_labels == query_labels_expanded).all()

    print(f"Query labels: {y_test.tolist()}")
    print(f"Neighbor labels:\n{neighbor_labels.tolist()}")
    print(f"All neighbors match query class: {all_match.item()}")
    assert all_match, "Neighbor class does not match query class!"
    print("Per-class neighbor match check passed!")

    print("\n--- Gradient Check ---")
    X_test_grad = X_test.clone().requires_grad_(True)
    
    # Faiss with autograd
    model_faiss = PerClassKNNConfidence(metric='euclidean', number_of_neighbors=1, index_type='flat')
    model_faiss.fit(X_t, y_t)
    # The model's forward returns confidence, but for grad check we need the distance.
    # We can't call _forward directly as it applies confidence_function.
    # Let's re-implement the forward logic here for the check.
    y_idx = torch.searchsorted(model_faiss.labels_unique, y_test)
    y_scaled = y_idx.float().unsqueeze(1) * model_faiss.class_penalty
    aug_x_test = torch.cat([X_test_grad, y_scaled], dim=1)
    dist_faiss_sq, _ = FaissKNN.apply(aug_x_test, model_faiss.points, 1, model_faiss.index, 'l2')
    dist_faiss = torch.sqrt(dist_faiss_sq)

    dist_faiss.mean(dim=-1).sum().backward()
    grad_faiss = X_test_grad.grad.clone()
    
    # PyTorch equivalent for comparison
    X_test_grad.grad.zero_()
    model_torch = PerClassKNNConfidencePyTorch(k=1, metric='euclidean', computation_mode='masked')
    model_torch.fit(X_t, y_t)
    # Get raw distance from torch model
    # --- CHANGED: avoid in-place modification error by using a detached clone with requires_grad ---
    X_test_clone = X_test_grad.clone().detach().requires_grad_(True)
    dist_torch = model_torch._compute_distance(X_test_clone, y_test)
    dist_torch.sum().backward()
    # Collect gradient from the clone, then copy it if needed for further use
    grad_torch = X_test_clone.grad.clone()
    # keep X_test_grad.grad consistent (not required but useful)
    X_test_grad.grad = grad_torch.clone()
    
    print("\nFaiss Gradients:", grad_faiss.norm(dim=-1).tolist())
    print("PyTorch Gradients:", grad_torch.norm(dim=-1).tolist())
    print("Faiss Grad Norm:", torch.linalg.norm(grad_faiss).item())
    print("PyTorch Grad Norm:", torch.linalg.norm(grad_torch).item())
    print("Gradient Difference Norm:", torch.linalg.norm(grad_faiss - grad_torch).item())
    assert torch.allclose(grad_faiss, grad_torch, atol=1e-5), "Gradients do not match!"
    print("Gradient check passed!")

    # Run the large benchmark (50000 x 512 and 100 batches of 128 x 512)
    try:
        benchmark_knn(
            train_N=50000,
            D=512,
            batches=100,
            batch_size=128,
            k=3,
            metric='cosine',
            use_gpu_if_available=True
        )
    except Exception as e:
        print(f"Benchmark failed: {e}")