import math
from abc import ABC, abstractmethod
from typing import Optional

import torch

class Transform(ABC):
    """


    Subclasses must implement:
      - matrix(self, param: Tensor) -> Tensor  # the (batch, D+1, D+1) matrix
      - param_size(self) -> int  # number of parameters
      - if support_calc_bounds() is True:
            calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> (Tensor, Tensor)  # lower, upper bounds for each parameter

    """

    #Methods that have to be implemented by subclasses:

    def __init__(self):
        """
        Initializes the Transform class.
        :param log: If True, orbits will be in log-space, otherwise in linear space.
        """
        pass

    @abstractmethod
    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Function that creates a transformation matrix treating the last dimension as the translation vector.
        Args:
            param: The parameter tensor. The output will use param's dtype and device.

        Returns:

        """
        ...

    @abstractmethod
    def param_size(self) -> int:
        """
        Returns the size of the parameter vector.
        :return: The size of the parameter vector.
        """
        ...


    @abstractmethod
    def sample_param(self,batch_size,domain,device="cpu",dtype=torch.float32) -> torch.Tensor:
        """
        Sample a parameter for the transformation.
        :return: The sampled parameter.
        """
        ...





    def project_parameters(self, param: torch.Tensor,domain, reflect=True) -> torch.Tensor:
        """
        Projects the parameters into the domain.
        :param param: The parameters to project.
        :param domain: The domain to project into.
        :return: The projected parameters.

        Args:
            reflect:  If True, reflect the parameters into the domain. If False, clip the parameters to the domain.
        """
        return param

    def normalize_parameters(self, param: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the parameters. Used for quaternions to keep at unit norm.
        :param param: The parameters to project.
        :return: The projected parameters.
        """
        return param



    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Returns a nonnegative tensor of the same shape as `param`, where each entry
        is how far that coordinate lies outside [lower_i, upper_i]. If inside, returns 0.
        """
        return torch.tensor(0.0, dtype=param.dtype, device=param.device)

    def normalization_violation(self, param: torch.Tensor) -> torch.Tensor:
        # Initialize result tensor with zero
        res = torch.tensor(0.0, dtype=param.dtype, device=param.device)

        # Reshape to batch shapes...,1 if needed
        if len(param.shape) == 1:
            # Expand res to have a shape compatible with param
            res = res.view(1)  # makes it shape [1], can broadcast with param of shape [N]
        else:
            # For higher dimensions, expand res to match batch dimensions
            batch_shape = param.shape[:-1]  # all dims except the last
            res = res.expand(*batch_shape, 1)  # shape becomes batch_shape + [1]

        return res




    @abstractmethod
    def supports_sobol(self) -> bool:
        """
        Indicates if the transform supports sobol sampling.
        :return: True if the transform supports sobol sampling, False otherwise.
        """
        ...


    def sample_space_param_size(self):
        """
        Tells special sample methods like sobol how many parameters they should sample.
        Only required if the transform supports sobol sampling.
        Returns:

        """
        return self.param_size()



    #methods that have to be potentially overridden by subclasses
    def sobol_to_param(self, sparam: torch.Tensor,domain) -> torch.Tensor:
        """
        Converts parameters from the range [0,1] to actual parameters.
        """
        pass

    @abstractmethod
    def supports_orbit(self) -> bool:
        """
        Indicates if the transform supports orbit sampling.
        :return: True if the transform supports orbit sampling, False otherwise.
        """


    def orbit(self,
              n_samples: int,
              domain,
              dim=0,
              extend: int = 0,
              shift: int = 0) -> Optional[torch.Tensor]:
        """
        Generates a set of samples along the orbit of a discrete transformation. Only works for Transforms that use a single parameter.
        Other transforms_old return None.
        Args:
            n_samples:
            domain:
            dim:
            extend:
            shift:

        Returns:

        """

        # get per-dimension bounds and determine parameter dimension

        low_p = 0
        high_p = 1 #assume sample space is always 0..1

        # total samples including padding
        total = n_samples + 2 * extend


        # linear spacing in [low_p, high_p]
        rng = high_p - low_p
        spacing = rng / (n_samples - 1) if n_samples > 1 else 0
        start = low_p - extend * spacing
        values = torch.linspace(start,
                                    high_p + extend * spacing,
                                    total) + shift * spacing
        # project values back into [0,1] via modulo (finish previous TODO)
        if total > 0:
            values = torch.remainder(values, 1.0)
        sample_dim = self.sample_space_param_size()
        params = torch.zeros((total, sample_dim), dtype=torch.float32)
        params[:, dim] = values

        #convert params from sample space to parameter space
        params = self.sobol_to_param(params)
        return params




    #methods that should not be overridden by subclasses
    def as_dict(self):
        """
        Provide a dictionary interface for backward compatibility.
        """
        return {
            "matrix": self.apply,
            "param_size": self.param_size(),
            "project_parameters": self.project_parameters,
            "calc_bounds": self.calc_bounds,
            "orbit": [self.orbit,],
            "param": [self.orbit_member,],
            "interval": self.interval(),
        }


    def __call__(self, param: torch.Tensor) -> torch.Tensor:
        """
        Calls the matrix method with the given parameter.
        :param param: The parameter to create the transformation matrix for.
        :return: The transformation matrix.
        """
        return self.matrix(param)



    def __getitem__(self, key: str):
        """
        Allow backward-compatible dict-style access, e.g. transform['matrix'].
        """
        return self.as_dict()[key]


    def apply(self,T: torch.Tensor,param: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation to the given tensor T using the provided parameters.
        :return: The transformed data.
        """
        if T is None:
            return self.matrix(param)
        #matrix multiplication
        return torch.matmul(self.matrix(param), T)

    def __eq__(self, other):
        # Only compare if they are exactly the same class
        if self.__class__ is not other.__class__:
            return NotImplemented
        # Compare each attribute in __dict__; both keys and values must match
        return self.__dict__ == other.__dict__

    def __hash__(self):
        # Build a hash from (class, sorted(__dict__.items()))
        #   • sorting ensures a consistent order
        #   • all values must themselves be hashable (ints, bools, tuples, etc.)
        items = tuple(sorted(self.__dict__.items()))
        return hash((self.__class__, items))



    #methods that are reserved for very specific subclasses that are used in conversions.
    def convert_forward(self, param: torch.Tensor) -> torch.Tensor:
        """
        This is a placeholder for any conversion logic that might be needed.
        This is used for converting transform that sample in the sample space of one represenation and then use a conversion between different represenations to get their result.
        :param param: The parameter to convert.
        :return: The converted parameter.
        """
        return param

    def convert_backward(self, param: torch.Tensor) -> torch.Tensor:
        """
        This is a placeholder for any conversion logic that might be needed.
        Convert the parameter back to the original format. See `convert_forward`.

        :param param: The parameter to convert.
        :return: The converted parameter.
        """
        return param


    def orbit_member(self, n: int, n_samples: int, domain,dim=0) -> torch.Tensor:
        res =  self.orbit(n_samples=n_samples, domain=domain,dim=dim, extend=0, shift=0)
        return res[n]







    #consider removing and only implementing this for bounded transforms_old
    @abstractmethod
    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the bounds given a domain parameter. This is a fallback for search methods that do not support projecting into the bounds.
        Not all transforms_old support this.
        Args:
            domain: The domain to calculate bounds for
            dtype: Optional dtype for the output tensors (defaults to torch.float32 if None)
            device: Optional device for the output tensors (defaults to 'cpu' if None)

        Returns:
            The bounds of the transformation.
        """
        ...

    def support_calc_bounds(self) -> bool:
        """
        Indicates if the transform implements calc_bounds_fallback.
        """
        return False



    #only needed for periodic transforms_old do we realy need this here? Can this be merged with normalize_parameters?
    def interval(self) -> Optional[tuple[float, float]]:
        """
        For periodic functions this can specify an interval to project into.
        :return: The interval of the transformation.
        """
        return None


    def reproject_to_interval(self, sparam: torch.Tensor) -> torch.Tensor:
        """
        Reprojects the parameters into the interval defined by the type of transformation.
        Does nothing for non-periodic transformations.
        :param sparam: The sample space parameters to reproject.
        :return: The reprojected parameters.
        """
        return sparam


    #TODO consider removing.
    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        """
        #NOT Tested Experimental. Likely not needed.

        Calculate the Euclidean distance between two parameter tensors.
        Args:
            param1: First parameter tensor
            param2: Second parameter tensor

        Returns:
            Distance between the parameters (scalar tensor)
        """
        # For standard bounded transforms_old, use regular Euclidean distance
        return torch.norm(param1 - param2, p=2, dim=-1)

    def distance_bounded(self, param1: torch.Tensor, param2: torch.Tensor, domain=None) -> torch.Tensor:
        """
        #NOT Tested Experimental. Likely not needed.
        Calculate the distance between two parameters, taking into account the bounds of the transformation.
        """
        return self.distance(param1, param2)

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Return a default neighborhood size for this transform in parameter space.
        By default returns 1 for each parameter.
        Args:
            domain: optional domain info to pick sensible scale
            dtype, device: tensor creation options
        Returns:
            1D tensor of length self.param_size() with nonnegative neighborhood sizes.
        """
        # default conservative small neighborhood
        size = self.param_size()
        return torch.ones(size, dtype=dtype, device=device)

    def num_discrete_values(self) -> Optional[int]:
        """
        Returns the number of discrete values this transformation can take.
        If infinite (continuous), returns None.
        Subclasses should override if discrete.
        """
        return None

    def identity_param(self,batch_size: Optional[int] = 1,dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Returns the parameter that corresponds to the identity transformation.
        By default, returns a zero tensor.
        Returns:
            A tensor of shape (param_size,) corresponding to the identity transformation.
        """
        size = self.param_size()
        return torch.zeros((batch_size, size), dtype=dtype, device=device)
