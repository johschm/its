#ignore for now
#untested composed mostly of old code
#TODO test

import torch
from sklearn.cluster import KMeans
import pickle
import torch.nn.functional as F
import pytorch_lightning as pl

def exp_radial_basis_function(r_sq, bandwidth):
    return torch.exp(-bandwidth * r_sq)

def cubic_radial_basis_function(r_sq, bandwidth):
    r = torch.sqrt(r_sq + 1e-8)
    return (bandwidth * r) ** 3

def inverse_quadratic_radial_basis_function(r_sq, bandwidth):
    return 1.0 / (1 + bandwidth * r_sq)

def linear_radial_basis_function(r_sq, bandwidth):
    r = torch.sqrt(r_sq + 1e-8)
    return bandwidth * r

def thin_plate_radial_basis_function(r_sq, bandwidth):
    r = torch.sqrt(r_sq + 1e-8)
    scaled_r = bandwidth * r
    return torch.where(scaled_r < 1e-8, torch.zeros_like(scaled_r), scaled_r ** 2 * torch.log(scaled_r + 1e-8))

class RBF(torch.nn.Module):
    """
    Radial Basis Function Network
    This layer applies a radial basis function to the input data. That it $output_{i} = rbf(||x - c_{i}|| / s_{i})$. Here c are the centers and s is the bandwidth which is a positive number.
    The RBF layer is parameterized by a set of centers and scaling factors.
    After this we apply a linear transformation to the output of the RBF layer using positive weights.

    """
    def __init__(self,
        latent_dim: int,
        number_centers: int,
        output_shape: torch.Size,
        basis_function='cubic',
        bandwidth_positivity_mode: str = 'clamp',
        weight_positivity_mode: str = 'clamp',
        min_bandwidth: float = 1e-6,
        min_weight: float = 1e-7,
        zeta: float = 1e-6):

        super(RBF, self).__init__()
        self.input_dim = latent_dim
        self.number_centers = number_centers
        self.output_shape = output_shape
        self.bandwidth_positivity_mode = bandwidth_positivity_mode
        self.weight_positivity_mode = weight_positivity_mode
        self.min_bandwidth = min_bandwidth
        self.min_weight = min_weight
        self.zeta = zeta
        self.centres = torch.nn.Parameter(torch.Tensor(number_centers, latent_dim))
        self.bandwidth = torch.nn.Parameter(torch.Tensor(number_centers))

        self.output_dim = torch.prod(torch.tensor(output_shape)).item()
        self.linear_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, number_centers))

        # Initialize parameters
        self.initialize_random()

        

        self.basis_function = basis_function
        self.basis_function_map = {
            'exp': exp_radial_basis_function,
            'cubic': cubic_radial_basis_function,
            'inverse_quadratic': inverse_quadratic_radial_basis_function,
            'linear': linear_radial_basis_function,
            'thin_plate': thin_plate_radial_basis_function
        }

    def _ensure_positive(self, x, positivity_type, min_value):
        if positivity_type == 'clamp':
            return torch.clamp(x, min=min_value)
        elif positivity_type == 'softplus':
            return torch.nn.functional.softplus(x)
        elif positivity_type == 'exp':
            return torch.exp(x)
        else:
            raise ValueError(f"Unknown positivity type: {positivity_type}")

    def _inverse_positive(self, target, positivity_type, min_value):
        # New helper: convert initialized positive value to raw parameter.
        if positivity_type == 'clamp':
            return target
        elif positivity_type == 'softplus':
            return torch.log(torch.exp(target) - 1)
        elif positivity_type == 'exp':
            return torch.log(target)
        else:
            raise ValueError(f"Unknown positivity type: {positivity_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        target_shape = (batch_size, self.number_centers, self.input_dim)
        expanded_x  = x.unsqueeze(1).expand(target_shape)
        dif = expanded_x - self.centres.unsqueeze(0)
        dif_sq = torch.sum(dif**2, dim=-1)

        bandwidth = self._ensure_positive(self.bandwidth, self.bandwidth_positivity_mode, self.min_bandwidth)
        weight = self._ensure_positive(self.linear_weight, self.weight_positivity_mode, self.min_weight)

        if self.basis_function not in self.basis_function_map:
            raise ValueError(f"Unknown basis function: {self.basis_function}")
        rbf_function = self.basis_function_map[self.basis_function]

        v_k = rbf_function(dif_sq, bandwidth.unsqueeze(0))

        result = torch.einsum('bo,nc->bn', weight, v_k) + self.zeta
        result = result.reshape(batch_size, *self.output_shape)
        return result

    def initialize_kmeans(self, init_data, bandwidth_multiplier=2, target_scale=1000):
        """
        K-means based initialization.
        bandwidth_multiplier: scales the average distances to compute the bandwidth.
        target_scale: scaling factor applied to the square root of the mean target outputs.
                      It is used to compute the bias adjustment (zeta) such that the network output
                      is well-calibrated relative to the average target magnitude.
        """
        all_zs = []
        all_targets = []
        for batch in init_data:
            zs, targets = batch
            all_zs.append(zs)
            all_targets.append(targets)
        all_zs = torch.cat(all_zs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        kmeans = KMeans(n_clusters=self.number_centers).fit(all_zs.cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, device=all_zs.device, dtype=all_zs.dtype)
        self.centres.data.copy_(centers)
        labels = kmeans.labels_
        Sigmas = torch.zeros((self.number_centers), device=all_zs.device, dtype=all_zs.dtype)
        for k in range(self.number_centers):
            inds = (labels == k)
            if inds.sum() > 0:
                points = all_zs[inds]
                avg_distance = torch.norm(points - centers[k], dim=1).mean()
                Sigmas[k] = avg_distance * bandwidth_multiplier
            else:
                Sigmas[k] = 1.0
        desired_bandwidth = 0.5 / (Sigmas ** 2)
        self.bandwidth.data.copy_(self._inverse_positive(desired_bandwidth, self.bandwidth_positivity_mode, self.min_bandwidth))
        torch.nn.init.normal_(self.linear_weight)
        self.linear_weight.data.copy_(self._inverse_positive(self.linear_weight.data, self.weight_positivity_mode, self.min_weight))
        alpha_rbf = target_scale * torch.sqrt(all_targets.mean()).detach().to(all_zs.device)
        self.zeta = 1 / (alpha_rbf ** 2)

    def initialize_random_centres(self, init_data, bandwidth_multiplier=2, target_scale=1000):
        """
        Random initialization with local bandwidth estimation.
        
        This method selects random points from the data as centers, but computes
        bandwidths based on local neighborhoods rather than global statistics.
        
        Args:
            init_data: Data used for initialization
            bandwidth_multiplier: Scales the computed bandwidths
            target_scale: Scaling factor for output normalization
        """
        all_zs = []
        all_targets = []
        for batch in init_data:
            zs, targets = batch
            all_zs.append(zs)
            all_targets.append(targets)
        all_zs = torch.cat(all_zs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Randomly select centers from input data
        indices = torch.randperm(all_zs.shape[0])[:self.number_centers]
        centers = all_zs[indices]
        self.centres.data.copy_(centers)
        
        # Compute distances from all points to each center
        dists = torch.cdist(all_zs, centers)
        
        # For each center, find the k nearest neighbors and compute bandwidth
        k = min(20, all_zs.shape[0] // self.number_centers)  # Reasonable neighborhood size
        
        # Vectorized computation using topk instead of loop
        nearest_dists, _ = torch.topk(dists, k, dim=0, largest=False)
        local_scales = nearest_dists.mean(dim=0) * bandwidth_multiplier
        local_scales = local_scales.clamp(min=1e-5)
        bandwidths = 0.5 / (local_scales ** 2)
        
        desired_bandwidth = torch.tensor(bandwidths, device=self.bandwidth.device)
        self.bandwidth.data.copy_(self._inverse_positive(desired_bandwidth, self.bandwidth_positivity_mode, self.min_bandwidth))
        
        torch.nn.init.normal_(self.linear_weight)
        self.linear_weight.data.copy_(self._inverse_positive(self.linear_weight.data, self.weight_positivity_mode, self.min_weight))
        alpha_rbf = target_scale * torch.sqrt(all_targets.mean()).detach().to(all_zs.device)
        self.zeta = 1 / (alpha_rbf ** 2)

    def initialize_variance(self, init_data, bandwidth_multiplier=2, target_scale=1000):
        """
        Variance-based initialization with data-driven center placement.
        
        This method places centers using a combination of grid placement and density-based
        adjustments, with adaptive bandwidths for each center.
        
        Args:
            init_data: Data used for initialization
            bandwidth_multiplier: Scales the computed bandwidths
            target_scale: Scaling factor for output normalization
        """
        all_zs = []
        all_targets = []
        for batch in init_data:
            zs, targets = batch
            all_zs.append(zs)
            all_targets.append(targets)
        all_zs = torch.cat(all_zs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Find data bounds
        min_vals, _ = torch.min(all_zs, dim=0)
        max_vals, _ = torch.max(all_zs, dim=0)
        
        # Get a sense of data density by dividing into regions
        if self.input_dim == 1:
            # For 1D, create a histogram and place more centers in high-density regions
            num_bins = min(50, all_zs.shape[0] // 10)
            hist = torch.histc(all_zs[:, 0], bins=num_bins, min=min_vals[0], max=max_vals[0])
            bin_edges = torch.linspace(min_vals[0], max_vals[0], num_bins+1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Sample centers with probability proportional to histogram count
            probs = hist / hist.sum()
            sample_indices = torch.multinomial(probs, self.number_centers, replacement=True)
            centers = bin_centers[sample_indices].unsqueeze(1)
        else:
            # For higher dimensions, use stratified sampling but place centers near data
            points_per_dim = max(2, int(self.number_centers ** (1/self.input_dim)))
            
            # Create initial grid centers
            grid_points = []
            for dim in range(self.input_dim):
                grid_points.append(torch.linspace(min_vals[dim], max_vals[dim], points_per_dim))
            
            grid = torch.meshgrid(*grid_points, indexing='ij')
            grid_centers = torch.stack([g.flatten() for g in grid], dim=1)
            
            # Select subset of grid points if needed
            if grid_centers.shape[0] > self.number_centers:
                idx = torch.randperm(grid_centers.shape[0])[:self.number_centers]
                centers = grid_centers[idx]
            else:
                centers = grid_centers
                
            # For each center, move it toward the nearest data point
            for i in range(centers.shape[0]):
                dists = torch.norm(all_zs - centers[i], dim=1)
                closest_idx = torch.argmin(dists)
                # Move center 50% toward nearest data point
                centers[i] = centers[i] * 0.5 + all_zs[closest_idx] * 0.5
        
        self.centres.data.copy_(centers.to(self.centres.device))
        
        # Compute adaptive bandwidths for each center
        dists = torch.cdist(all_zs, centers)
        
        # For each center, compute bandwidth based on local neighborhood
        k = min(20, all_zs.shape[0] // self.number_centers)
        
        # Vectorized computation using topk instead of loop
        nearest_dists, _ = torch.topk(dists, k, dim=0, largest=False)
        local_scales = nearest_dists.mean(dim=0) * bandwidth_multiplier
        local_scales = local_scales.clamp(min=1e-5)
        bandwidths = 0.5 / (local_scales ** 2)
        
        desired_bandwidth = torch.tensor(bandwidths, device=self.bandwidth.device)
        self.bandwidth.data.copy_(self._inverse_positive(desired_bandwidth, self.bandwidth_positivity_mode, self.min_bandwidth))
        
        torch.nn.init.normal_(self.linear_weight)
        self.linear_weight.data.copy_(self._inverse_positive(self.linear_weight.data, self.weight_positivity_mode, self.min_weight))
        alpha_rbf = target_scale * torch.sqrt(all_targets.mean()).detach().to(all_zs.device)
        self.zeta = 1 / (alpha_rbf ** 2)

    def initialize_random(self):
        """
        Random initialization without data.
        Sets centres, bandwidth, and linear_weight with random values.
        """
        with torch.no_grad():
            # Randomly initialize centres from standard normal
            self.centres.data.copy_(torch.randn(self.number_centers, self.input_dim, device=self.centres.device))
            # Create random positive bandwidth targets and compute raw parameter using _inverse_positive
            random_bandwidth = torch.rand(self.number_centers, device=self.bandwidth.device) * 0.1 + 0.1  # ensure > min_bandwidth
            self.bandwidth.data.copy_(self._inverse_positive(random_bandwidth, self.bandwidth_positivity_mode, self.min_bandwidth))
            # Initialize weights using normal initializer and adjust with _inverse_positive
            torch.nn.init.normal_(self.linear_weight)
            self.linear_weight.data.copy_(self._inverse_positive(self.linear_weight.data, self.weight_positivity_mode, self.min_weight))

class RBFPostfit(pl.LightningModule):
    def __init__(self, latent_dim: int, number_centers: int, output_shape: torch.Size, init_data=None, 
                 optimize_centers=False, lr=1e-1, bandwidth_multiplier=2, target_scale=1000, init_method="kmeans"):
        """
        Initialization of the RBFPostfit module.

        latent_dim, number_centers, output_shape: [parameters unchanged]
        init_data: data used for initializing the RBF parameters.
        optimize_centers: whether to optimize the RBF centers during training.
        lr: learning rate.
        bandwidth_multiplier: scaling factor for bandwidth computation from the mean distances.
        target_scale: scaling factor used to adjust the bias parameter (zeta). Specifically, it is multiplied
                      by the square root of the mean target outputs to obtain an intermediate scaling factor,
                      ensuring the network's output is appropriately scaled relative to the targets.
        init_method: initialization method ("kmeans", "random", or "variance").
        """
        super().__init__()
        self.rbf = RBF(latent_dim, number_centers, output_shape)
        self.lr = lr
        self.optimize_centers = optimize_centers

        if init_data is not None:
            with torch.no_grad():
                if init_method == "kmeans":
                    self.rbf.initialize_kmeans(init_data, bandwidth_multiplier, target_scale)
                elif init_method == "random":
                    self.rbf.initialize_random(init_data, bandwidth_multiplier, target_scale)
                elif init_method == "variance":
                    self.rbf.initialize_variance(init_data, bandwidth_multiplier, target_scale)
                else:
                    raise ValueError(f"Unknown init_method: {init_method}")
                self.rbf.to(self.rbf.centres.device)

    def forward(self, x):
        return self.rbf(x)

    def training_step(self, batch, batch_idx):
        z, target = batch
        predicted_value = self.rbf(z)
        loss = F.mse_loss(predicted_value, target)
        self.log("loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        if self.optimize_centers:
            return torch.optim.Adam(self.rbf.parameters(), lr=self.lr)
        return torch.optim.Adam([self.rbf.linear_weight], lr=self.lr)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        model_state = state['_modules'].copy()
        del model_state["decoder"]
        state['_modules'] = model_state
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.decoder = None

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))

