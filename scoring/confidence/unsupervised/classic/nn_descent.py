import torch
import torch.nn.functional as F
import cupy as cp
from cuml.neighbors import NearestNeighbors
from typing import Optional, Callable, Any, Literal

from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase

def default_confidence_function(distances: torch.Tensor) -> torch.Tensor:
    # Simple example: inverse of mean distance
    return 1.0 / (1.0 + distances.mean(dim=1))


class CuMLKNNConfidence(ClassicConfidenceBase):
    """
    Calculates confidence scores based on nearest neighbor distances using cuML GPU-accelerated KNN.
    Supports 'euclidean' and 'cosine' metrics, fully GPU-enabled.
    """

    def __init__(
        self,
        metric: Literal['euclidean', 'cosine'] = 'euclidean',
        function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        number_of_neighbors: int = 3,
        algorithm: str = 'auto',
        verbose: bool = False,
        algo_params: Optional[dict] = None,
        metric_params: Optional[dict] = None,
        input_transform: Optional[Any] = None,
    ):
        super().__init__(input_transform=input_transform)
        self.metric = metric
        self.number_of_neighbors = number_of_neighbors
        self.algorithm = algorithm
        self.verbose = verbose
        self.algo_params = algo_params
        self.metric_params = metric_params

        self.points: Optional[torch.Tensor] = None
        self.nn: Optional[NearestNeighbors] = None

        self.confidence_function: Callable[[torch.Tensor], torch.Tensor] = (
            function if function is not None else default_confidence_function
        )

    def _fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> "CuMLKNNConfidence":
        if not x.is_cuda:
            raise ValueError("CuMLKNNConfidence requires CUDA tensors for full GPU support.")

        self.points = x.clone().detach()

        # Normalize for cosine distance
        if self.metric == 'cosine':
            data_torch = F.normalize(self.points, p=2, dim=1)
            metric = 'cosine'
        elif self.metric == 'euclidean':
            data_torch = self.points
            metric = 'euclidean'
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Move data to CuPy array
        data = cp.asarray(data_torch.detach().contiguous().cpu().numpy(), dtype=cp.float32)

        # Initialize cuML NearestNeighbors
        self.nn = NearestNeighbors(
            n_neighbors=self.number_of_neighbors,
            algorithm=self.algorithm,
            metric=metric,
            verbose=self.verbose,
            algo_params=self.algo_params,
            metric_params=self.metric_params,
            output_type='cupy',
        )
        self.nn.fit(data)
        return self

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.nn is None:
            raise ValueError("Model not fitted. Call fit() before forward().")
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor for GPU acceleration.")

        # Normalize if cosine
        if self.metric == 'cosine':
            x_torch = F.normalize(x, p=2, dim=1)
        else:
            x_torch = x

        # Convert to CuPy
        queries = cp.asarray(x_torch.detach().contiguous().cpu().numpy(), dtype=cp.float32)

        distances, _ = self.nn.kneighbors(queries)

        # Convert back to torch.cuda.Tensor
        distances_torch = torch.as_tensor(distances, device=x.device)

        return self.confidence_function(distances_torch)


import torch
import ggnn
from typing import Optional, Callable, Any, Literal
import sys, os, contextlib


def default_confidence_function(distances: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + distances.mean(dim=1))

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

class GGNNKNNConfidence(ClassicConfidenceBase):
    """
    Calculates confidence scores using GGNN approximate nearest neighbors.
    Supports 'euclidean' and 'cosine' distance measures.
    """

    def __init__(
        self,
        number_of_neighbors: int = 3,
        confidence_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        metric: str = "euclidean",
        return_results_on_gpu: bool = True,
        shard_size: Optional[int] = None,
        gpus: Optional[list[int]] = None,
        log_level: int = 0,
        k_build: int = 24,
        tau_build: float = 0.5,
        refinement_iterations: int = 2,
        tau_query: float = 0.64,
        max_iterations: int = 400,
        input_transform: Optional[Any] = None,
    ):
        super().__init__(input_transform=input_transform)
        self.number_of_neighbors = number_of_neighbors
        self.confidence_function = confidence_function or default_confidence_function
        self.metric = metric.lower()
        self.return_results_on_gpu = return_results_on_gpu
        self.shard_size = shard_size
        self.gpus = gpus
        self.log_level = log_level
        self.k_build = k_build
        self.tau_build = tau_build
        self.refinement_iterations = refinement_iterations
        self.tau_query = tau_query
        self.max_iterations = max_iterations
        self.gg = ggnn.GGNN()
        self.base_set = None
        self.measure = (
            ggnn.DistanceMeasure.Cosine
            if self.metric == "cosine"
            else ggnn.DistanceMeasure.Euclidean
        )

    def _fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> "GGNNKNNConfidence":
        ggnn.set_log_level(self.log_level)

        self.base_set = x.clone().detach()
        self.gg.set_base(self.base_set)
        self.gg.set_return_results_on_gpu(self.return_results_on_gpu)

        if self.shard_size is not None:
            self.gg.set_shard_size(self.shard_size)
        if self.gpus is not None:
            self.gg.set_gpus(self.gpus)

        # Build GGNN graph
        with suppress_output():
            self.gg.build(
                k_build=self.k_build,
                tau_build=self.tau_build,
                refinement_iterations=self.refinement_iterations,
                measure=self.measure
            )

        return self

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.base_set is None:
            raise ValueError("Model not fitted. Call fit() before forward().")

        # Run GGNN query
        indices, dists = self.gg.query(
                x,
                k_query=self.number_of_neighbors,
                tau_query=self.tau_query,
                max_iterations=self.max_iterations,
                measure=self.measure
        )

        dists_tensor = torch.tensor(dists, device=x.device, dtype=x.dtype)
        return self.confidence_function(dists_tensor)

if __name__ == "__main__":
    ggnn.set_log_level(1)
    # Example usage
    model = CuMLKNNConfidence(metric='euclidean', number_of_neighbors=5)
    data = torch.randn(100, 20).cuda()  # Example data on GPU
    model.fit(data)
    test_data = torch.randn(10, 20).cuda()
    scores = model.forward(test_data)
    print(scores)

    model_ggnn = GGNNKNNConfidence(number_of_neighbors=5)
    model_ggnn.fit(data)
    scores_ggnn = model_ggnn.forward(test_data)
    print(scores_ggnn)
    scores_ggnn = model_ggnn.forward(test_data)
    print(scores_ggnn)
    scores_ggnn = model_ggnn.forward(test_data)
    print(scores_ggnn)
    scores_ggnn = model_ggnn.forward(test_data)
    print(scores_ggnn)
    scores_ggnn = model_ggnn.forward(test_data)
    print(scores_ggnn)

