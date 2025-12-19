import torch

from utils.transforms.apply import grid_resample
from utils.transform_sequence import create_sampler, create_parameter_sampler, TransformSequence


class TransformationProblem:
    def __init__(self, confidence_module, transform_sequence:TransformSequence, consolidate_method="consolidate_simple",max_batch_size=None):
        """
        :param confidence_module: Module to calculate confidence for transformed inputs.
        :param transform_sequence: TransformSequence object that handles transformations.
        :param consolidate_method: Method to consolidate parallel runs. Default is "consolidate_simple".
        """
        self.transform_sequence = transform_sequence
        self.confidence_module = confidence_module
        self.consolidate_method = consolidate_method
        self.max_batch_size = max_batch_size

    def __call__(self, param):
        """
        Apply the transformation sequence to get a transformation matrix.
        :param param: Parameters for the transformations.
        :return: Transformation matrix.
        """
        return self.transform_sequence(param)

    def normalize(self, param):
        """
        Normalize parameters using the underlying transform sequence's normalization rules.
        """
        return self.transform_sequence.normalize(param)

    def transform(self, x, param):
        """
        Transform the input x using the transformation matrix T.
        :param x: Input x
        :param param: Parameter to transform.
        :return: Transformed tensor.
        """
        return self.transform_sequence.transform(x, param)

    def sample_neighbor(self, param, neighboor_hood_size=None):
        """
        Sample a random point in the neighborhood of the parameter.
        :param param: Parameter to sample around.
        :param neighboor_hood_size: Optional scale factor for the neighborhood size.
        :return: Random parameter in the neighborhood.
        """
        return self.transform_sequence.sample_neighbor(param, neighboor_hood_size)

    def initial_param(self, batch_size=1, n_samples=None):
        """
        Create an initial parameter for the transformation.
        :param batch_size: Batch size.
        :param n_samples: Number of samples per batch.
        :return: Initial parameter.
        """
        return self.transform_sequence.initial_param(batch_size, n_samples)

    def correct_param(self, param, reflect=None):
        """
        Corrects the parameter to be within the bounds.
        :param param: Parameter to correct.
        :param reflect: Whether to use reflection for correction (if None, uses transform_sequence default).
        :return: Corrected parameter.
        """
        reflect = self.transform_sequence.reflect if reflect is None else reflect
        return self.transform_sequence.correct_param(param, reflect)

    def boundary_violation(self, param):
        """
        Compute a "violation distance" for each coordinate in the parameter.
        Wraps the boundary_violation method from transform_sequence.
        
        Args:
            param: Parameter tensor to check for boundary violations
            
        Returns:
            Tensor containing violation distances for each coordinate
        """
        return self.transform_sequence.boundary_violation(param)

    def normalization_violation(self, param):
        """
        Compute a "violation distance" for each coordinate in the parameter
        based on normalization constraints.

        Args:
            param: Parameter tensor to check for normalization violations

        Returns:
            Single value tensor representing total normalization violation
            """
        return self.transform_sequence.normalization_violation(param)
        
    def distance(self, param1, param2):
        """
        Calculate the distance between two parameter vectors using the 
        distance metric defined by the transform sequence.
        
        Args:
            param1: First parameter tensor
            param2: Second parameter tensor
            
        Returns:
            Distance between the parameters
        """
        return self.transform_sequence.distance(param1, param2)
        
    def extract_param_sizes(self):
        """
        Returns the parameter sizes for each transformation in the sequence.
        
        Returns:
            List of parameter sizes
        """
        return self.transform_sequence.extract_param_sizes()
        
    def calc_complete_size(self):
        """
        Calculates the total number of parameters for all transformations.
        
        Returns:
            Total parameter size
        """
        return self.transform_sequence.calc_complete_size()

    def matrix_dim(self) -> int:
        """
        Infers the dimension of the transformation matrix by generating one.
        """
        # Create a zero parameter vector for a single sample
        param_dim = self.get_dim()
        dummy_param = torch.zeros(1, param_dim, device=self.transform_sequence.dummy_param.device, dtype=self.transform_sequence.dummy_param.dtype)
        # Generate a transformation matrix
        T = self.transform_sequence(dummy_param)
        # Return its dimension
        return T.shape[-1]

    def calculate_error(self, x, param, y=None):
        if self.max_batch_size is None or x.size(0) <= self.max_batch_size:
            return self._calculate_error(x, param, y)
        return self._calculate_error_batched(x, param, y)

    def _calculate_error(self, x, param,y=None):
        """
        Calculate the error as the negative confidence. In addition outputs the predicted class.
        :param x: Input tensor to transform.
        :param param: Transformation parameter.
        :return: Error value and logits
        """
        # Apply the transformations
        # Calculate the error using the confidence module
        x = self.transform_sequence.transform(x,param)
        res = self.confidence_module(x, y)
        if isinstance(res, tuple):
            conf, logits = res
            error = -conf
            if logits is None:
                return error, torch.empty(x.size(0), device=x.device).unsqueeze(-1)
            return error, logits
        else:
            return -res,torch.empty(x.size(0), device=x.device).unsqueeze(-1)

    def _calculate_error_batched(self, x, param, y=None):
        """
        Batched error calculation respecting max_batch_size.
        :param x: Tensor of shape (B, ...).
        :param param: Tensor of shape (B, ...).
        :param y: Optional tensor of shape (B, ...).
        :return: error of shape (B, ...) and logits of shape (B, ...).
        """
        B = x.size(0)
        max_bs = self.max_batch_size or B
        errors = []
        classes = []
        for start in range(0, B, max_bs):
            end = start + max_bs
            xi = x[start:end]
            pi = param[start:end]
            yi = y[start:end] if y is not None else None
            err_chunk, cls_chunk = self._calculate_error(xi, pi, yi)
            errors.append(err_chunk)
            classes.append(cls_chunk)
        error = torch.cat(errors, dim=0)
        if classes[0] is None:
            return error, torch.empty(x.size(0), device=x.device).unsqueeze(-1)
        clss = torch.cat(classes, dim=0)
        return error, clss

    def consolidate_simple(self, x, best_param, best_error, classes_best):
        """
        Consolidate parallel runs by selecting for each sample the run with the minimum error.
        Assumes:
          best_param: Tensor of shape (batch_size, parallel_runs, param_dim)
          best_error: Tensor of shape (batch_size, parallel_runs)
          best_other_data: Tensor of shape (batch_size, parallel_runs, class_dim)
        :return: Consolidated best_param, best_error, best_other_data (one per sample)
        """
        # Get indices of minimum error per sample
        best_indices = torch.argmin(best_error, dim=1, keepdim=True)  # shape: (batch_size, 1)
        # Gather best parameters: best_param has shape (batch_size, parallel_runs, param_dim)
        best_param_selected = best_param.gather(
            dim=1, 
            index=best_indices.unsqueeze(-1).expand(-1, -1, best_param.size(-1))
        ).squeeze(1)
        # Gather best errors: best_error has shape (batch_size, parallel_runs)
        best_error_selected = best_error.gather(dim=1, index=best_indices).squeeze(1)
        # Gather best classes: classes_best has shape (batch_size, parallel_runs, class_dim)
        best_classes = classes_best.gather(
            dim=1, 
            index=best_indices.unsqueeze(-1).expand(-1, -1, classes_best.size(-1))
        ).squeeze(1)
        return best_param_selected, best_error_selected, best_classes

    def consolidate_average(self, x, best_param, best_error, classes_best, top_fraction=0.9):
            """
            Consolidate parallel runs by averaging over top-performing runs.
            Top-performing runs are defined per sample as those with error
            within the top_fraction of the best-to-worst range.

            Args:
                best_param: Tensor (batch_size, parallel_runs, param_dim)
                best_error: Tensor (batch_size, parallel_runs)
                classes_best: Tensor (batch_size, parallel_runs, class_dim)
                top_fraction: float, fraction of the best-to-worst range to include

            Returns:
                Averaged best_param, best_error, best_classes (one per sample)
            """
            # Find min and max error per sample
            min_error, _ = best_error.min(dim=1, keepdim=True)  # shape: (batch_size, 1)
            max_error, _ = best_error.max(dim=1, keepdim=True)  # shape: (batch_size, 1)

            # Threshold: include runs with error <= min_error + top_fraction * (max_error - min_error)
            threshold = min_error + top_fraction * (max_error - min_error)

            # Create mask for top-performing runs
            mask = best_error <= threshold  # shape: (batch_size, parallel_runs), bool

            # Convert mask to float for averaging
            mask_float = mask.float()

            # Avoid division by zero
            counts = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)

            # Weighted average over selected runs
            best_param_avg = (best_param * mask_float.unsqueeze(-1)).sum(dim=1) / counts
            best_error_avg = (best_error * mask_float).sum(dim=1) / counts.squeeze(1)
            best_classes_avg = (classes_best * mask_float.unsqueeze(-1)).sum(dim=1) / counts

            return best_param_avg, best_error_avg, best_classes_avg

    def consolidate_robust(self, x, best_param, best_error, classes_best):
        """
        Consolidate by evaluating robustness of solutions through neighboring samples.
        """
        robust_samples = 8
        B, P, D = best_param.shape
        error_samples = []
        x_expanded = x.unsqueeze(1).expand(-1, P, -1, -1, -1).contiguous().view(B * P, *x.shape[1:])
        for _ in range(robust_samples):
            # Sample neighbors vectorized over candidates.
            best_param_flat = best_param.reshape(B * P, D)
            neighbor_sample = self.sample_neighbor(best_param_flat)  # shape: [B*P, D]
            # Compute error for each neighbor sample.
            error, _ = self.calculate_error(x_expanded, neighbor_sample)  # error shape: [B*P]
            error = error.reshape(B, P)
            error_samples.append(error)
        # Average errors over robust samples (looped only over robust_samples)
        error_stack = torch.stack(error_samples, dim=0)  # shape: [robust_samples, B, P]
        avg_error = error_stack.mean(dim=0) + 0* error_stack.var(dim=0)  # shape: [B, P]
        # Select candidate with minimal average error per batch.
        best_indices = avg_error.argmin(dim=1)  # shape: [B]
        best_param_selected = best_param.gather(1, best_indices.view(-1,1,1).expand(-1, 1, D)).squeeze(1)
        robust_error_selected = avg_error.gather(1, best_indices.view(-1,1)).squeeze(1)
        if classes_best.dim() == 3 and classes_best.size(-1) == 1:
            classes_best = classes_best.squeeze(-1)
        selected_class = classes_best.gather(1, best_indices.view(-1,1)).squeeze(1)
        return best_param_selected, robust_error_selected, selected_class

    def consolidate_class(self, x, best_param, best_error, classes_best):
        """
        Consolidate parallel runs by selecting the best run for the class with the highest combined score of confidence and frequency.
        Returns a dictionary with the selected parameters, error, and class for each sample.
        """
        if classes_best.dim() == 3 and classes_best.size(-1) == 1:
            classes_best = classes_best.squeeze(-1)

        confidence = -best_error  # Convert error to confidence
        batch_size, parallel_runs = best_error.shape

        num_classes = classes_best.max() + 1

        # Calculate counts and sum of confidences for each class
        counts = torch.zeros(batch_size, num_classes, device=best_error.device)
        sum_conf = torch.zeros_like(counts)
        counts.scatter_add_(1, classes_best, torch.ones_like(confidence))
        sum_conf.scatter_add_(1, classes_best, confidence)
        best_per_class = torch.full((batch_size, num_classes), 0.0, device=best_error.device)
        best_per_class = best_per_class.scatter_reduce(1, classes_best, confidence, reduce='amax', include_self=False)

        # Compute score and select the class with the highest score
        percentages = counts / parallel_runs
        score = best_per_class * (percentages**0.1)  # * parallel_runs**0.25
        selected_class = score.argmax(dim=1)

        # Mask to find runs of the selected class
        mask = (classes_best == selected_class.unsqueeze(1))
        masked_confidence = confidence * mask.float()

        # Find the best run within the selected class
        _, max_indices = masked_confidence.max(dim=1)

        # Gather results
        best_param_selected = best_param.gather(
            1, max_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, best_param.size(-1))
        ).squeeze(1)
        best_error_selected = best_error.gather(1, max_indices.unsqueeze(1)).squeeze(1)

        return best_param_selected, best_error_selected, selected_class

    def consolidate(self, x, best_param, best_error, classes_best):
        if self.consolidate_method == "consolidate_simple":
            return self.consolidate_simple(x, best_param, best_error, classes_best)
        elif self.consolidate_method == "consolidate_average":
            return self.consolidate_average(x, best_param, best_error, classes_best)
        elif self.consolidate_method == "consolidate_class":
            return self.consolidate_class(x, best_param, best_error, classes_best)
        elif self.consolidate_method == "consolidate_robust":
            return self.consolidate_robust(x, best_param, best_error, classes_best)
        else:
            raise ValueError(f"Unknown consolidation method: {self.consolidate_method}")

    def to(self, device):
        """
        Moves all internal tensors to the specified device.
        """
        self.transform_sequence.to(device)
        return self

    def params_to_matrix(self,param):
        """
        Converts the parameter tensor to a transformation matrix.
        :param param: Parameter tensor.
        :return: Transformation matrix.
        """
        return self.transform_sequence(param)

    def get_dim(self):
        return self.transform_sequence.calc_complete_size()

    def get_identity_parameters(self, batch_size=1):
        """
        Create a parameter that represents the identity transformation.
        :param batch_size: Batch size.
        :return: Identity parameter.
        """
        return self.transform_sequence.get_identity_parameters(batch_size)




    # -------- New: conversions between continuous/discrete problems --------
    def to_discrete(self, n_samples, application_method=None):
        """
        Return a new TransformationProblem with a discrete TransformSequence.
        If the underlying sequence is already discrete, returns self.
        """
        # local import avoids hard coupling at module import time
        try:
            from utils.transform_sequence_discrete import TransformSequenceDiscrete
            is_discrete = isinstance(self.transform_sequence, TransformSequenceDiscrete)
        except Exception:
            is_discrete = False

        if is_discrete:
            return self
        if not hasattr(self.transform_sequence, "to_discrete"):
            raise NotImplementedError("Underlying transform sequence cannot be discretized (missing to_discrete).")
        new_seq = self.transform_sequence.to_discrete(n_samples, application_method=application_method)
        return TransformationProblem(self.confidence_module, new_seq, self.consolidate_method, self.max_batch_size)




    def to_continuous(self, neighbour_hood_size=None, application_method=None, init_method="individual", reflect=False):
        """
        Return a new TransformationProblem with a continuous TransformSequence.
        If the underlying sequence is already continuous, returns self.
        """
        # Best-effort: prefer a method on the sequence; otherwise, attempt discrete->continuous
        if hasattr(self.transform_sequence, "to_continuous"):
            new_seq = self.transform_sequence.to_continuous(neighbour_hood_size, application_method, init_method, reflect)
            return TransformationProblem(self.confidence_module, new_seq, self.consolidate_method, self.max_batch_size)
        return self  # assume already continuous

    def ensure_discrete(self, n_samples, application_method=None):
        """
        Ensure this problem uses a discrete sequence. If not, convert and return a new problem.
        """
        try:
            from utils.transform_sequence_discrete import TransformSequenceDiscrete
            if isinstance(self.transform_sequence, TransformSequenceDiscrete):
                return self
        except Exception:
            pass
        return self.to_discrete(n_samples, application_method=application_method)


if __name__ == "__main__":
    import torch
    from utils.transform_sequence import TransformSequence
    from utils.affine_transforms import AffineTransformation2D

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a transform sequence
    transforms_2d = [
        AffineTransformation2D.ROTATION.value,
        AffineTransformation2D.TRANSLATION.value
    ]
    domains_2d = [
        (-torch.pi, torch.pi),  # Rotation domain
        ((-1.0, 1.0), (-2.0, 2.0))  # Translation domain
    ]
    
    transform_seq = TransformSequence(transforms_2d, domains_2d, device=device)
    
    # Create a dummy confidence module for testing
    class DummyConfidence:
        def __call__(self, x):
            return torch.sum(x**2, dim=(1,2,3)), torch.ones(x.shape[0], 10).to(x.device)
    
    confidence_module = DummyConfidence()
    
    # Create transformation problem
    problem = TransformationProblem(confidence_module, transform_seq)
    
    # Test initial_param
    param = problem.initial_param(batch_size=2)
    print(f"Initial param shape: {param.shape}")
    
    # Test transformation
    x = torch.randn(2, 3, 28, 28, device=device)
    T = problem(param)
    print(f"Transformation matrix shape: {T.shape}")
    
    # Test transform
    transformed_x = problem.transform(x, param)
    print(f"Transformed x shape: {transformed_x.shape}")
    
    # Test error calculation
    error, classes = problem.calculate_error(x, param)
    print(f"Error: {error}, Classes: {classes}")
    
    # Test new functions
    boundary_violation = problem.boundary_violation(param)
    print(f"Boundary violation: {boundary_violation.shape}")
    
    param2 = problem.initial_param(batch_size=2)
    dist = problem.distance(param, param2)
    print(f"Distance: {dist}")
    
    param_sizes = problem.extract_param_sizes()
    print(f"Parameter sizes: {param_sizes}")
    
    total_size = problem.calc_complete_size()
    print(f"Total parameter size: {total_size}")