import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
#TODO this seriosly needs a rethink. THe problem is that for a lot of problems we want to sample in a different space than we use for parameterization.
#TODO this mainly applies to roations. For example Quaternions are 4 parametes but all have unit norm so for sampling we need to be clever. This makes application of sampling methods like sobol sequences difficult.
#But specifing this makes setting domains diffuclt.Maybe removing the domain requirment for 3d roations would help as the domain here is mostly diifucult to interpret anyway and does nothing meaningfull for most anyway.
#Workaround is using conversion functions as the convertransform stores a second domain.
#TODO at least for normal rotation and quaternion we should specify a sample space transform forward so that we can sample using sobol and use the transform forward to interpret it.
class Transform(ABC):
    """
    Base class for all transforms_old.
    Generally speaking each transform has two spaces:
    - sample space: the space in which the parameters are sampled. For example rotations in 2d can be angeles between 0 and 2*pi.
    - parameter space: the space in which the parameters are projected. For example rotations in 2d can be represented by a 2x2 rotation matrix. This dimension is 4.
    For most represenation we simply assume that the sample space is the same as the parameter space.
    The domain is per default given by the sample space.


    Subclasses must implement:
      - matrix(self, param: Tensor) -> Tensor  # the (batch, D+1, D+1) matrix
    """
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


    def apply(self,T: torch.Tensor,param: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation to the given tensor T using the provided parameters.
        :return: The transformed data.
        """
        if T is None:
            return self.matrix(param)
        #matrix multiplication
        return torch.matmul(self.matrix(param), T)

    @abstractmethod
    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the bounds given a domain parameter. This is a fallback for search methods that do not support projecting into the bounds.

        Args:
            domain: The domain to calculate bounds for
            dtype: Optional dtype for the output tensors (defaults to torch.float32 if None)
            device: Optional device for the output tensors (defaults to 'cpu' if None)

        Returns:
            The bounds of the transformation.
        """
        ...


    def calc_bounds_sample_space(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:#
        """

        Args:
            domain:
            dtype:
            device:

        Returns:

        """
        return self.calc_bounds(domain, dtype=dtype, device=device)


    def reproject_to_interval(self, sparam: torch.Tensor) -> torch.Tensor:
        """
        Reprojects the parameters into the interval defined by the type of transformation.
        Does nothing for non-periodic transformations.
        :param sparam: The sample space parameters to reproject.
        :return: The reprojected parameters.
        """
        return sparam



    def project_parameters(self, param: torch.Tensor,domain, reflect=True) -> torch.Tensor:
        """
        Projects the parameters into the domain.
        :param param: The parameters to project.
        :param domain: The domain to project into.
        :return: The projected parameters.

        Args:
            reflect:  If True, reflect the parameters into the domain. If False, clip the parameters to the domain.
        """
        ...

    def sample_space_reproject_to_interval(self, sparam: torch.Tensor) -> torch.Tensor:
        """
        Reprojects the parameters into the sample space interval defined by the type of transformation.
        Does nothing for non-periodic transformations.
        :param param: The parameters to reproject.
        :return: The reprojected parameters.
        """
        return self.reproject_to_interval(sparam)

    def sample_space_project_parameters(self, sparam: torch.Tensor, domain, reflect=True) -> torch.Tensor:
        """
        Projects the parameters into the sample space domain.
        :param param: The parameters to project.
        :param domain: The domain to project into.
        :return: The projected parameters.
        """
        return self.project_parameters(sparam, domain, reflect=reflect)

    def interval(self) -> Optional[tuple[float, float]]:
        """
        For periodic functions this can specify an interval to project into.
        :return: The interval of the transformation.
        """
        return None

    def sample_space_interval(self) -> Optional[tuple[float, float]]:
        """
        For periodic functions this can specify an interval to sample from.
        :return: The interval of the transformation.
        """
        return self.interval()

    @abstractmethod
    def param_size(self) -> int:
        """
        Returns the size of the parameter vector.
        :return: The size of the parameter vector.
        """
        ...


    def sample_space_param_size(self):
        return self.param_size()



    def boundary_violation_without_interval(self, param: torch.Tensor, domain) -> torch.Tensor:
        return self.boundary_violation(param, domain)

    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Returns a nonnegative tensor of the same shape as `param`, where each entry
        is how far that coordinate lies outside [lower_i, upper_i]. If inside, returns 0.
        """
        dtype = param.dtype
        device = param.device
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)

        # violation_below = max(0, lower - param)
        violation_below = torch.relu(lower - param)
        # violation_above = max(0, param - upper)
        violation_above = torch.relu(param - upper)

        return violation_below + violation_above

    def distance_bounded(self, param1: torch.Tensor, param2: torch.Tensor, domain=None) -> torch.Tensor:
        """
        #NOT Tested Experimental. Likely not needed.
        Calculate the distance between two parameters, taking into account the bounds of the transformation.
        """
        return self.distance(param1, param2)
    #TODO REMOVE
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





    def orbit_member(self, n: int, n_samples: int, domain,dim=0) -> torch.Tensor:
        res =  self.orbit(n_samples=n_samples, domain=domain,dim=dim, extend=0, shift=0)
        return res[n]


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

    def sample_param(self,batch_size,domain,device="cpu",dtype=torch.float32) -> torch.Tensor:
        """
        Sample a parameter for the transformation.
        :return: The sampled parameter.
        """
        low, up = self.calc_bounds(domain, dtype=dtype, device=device)
        return torch.rand(batch_size,self.param_size(), device=device, dtype=dtype) * (up - low) + low

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

    def sample_space_convert_forward(self, param: torch.Tensor) -> torch.Tensor:
        """
        Converts parameters from the sample space to the parameter space.
        :param param: The parameter to convert.
        :return: The converted parameter.
        """
        return param

    def sample_space_convert_backward(self, param: torch.Tensor) -> torch.Tensor:
        """
        Converts parameters from the parameter space to the sample space.
        This is a placeholder for any conversion logic that might be needed.
        :param param: The parameter to convert.
        :return: The converted parameter.
        """
        return param

    def parameter_space_not_equal_sample_space(self) -> bool:
        """
        This indiciates that the parameter space is not the sample space.
        Thus we require a conversion from the sampled parametes to the parameter space.
        Returns:

        """
        return False

    def domain_given_in_sample_space(self) -> bool:
        """
        Indicates that parameter parsing requires backtransforming the parameters in the sample space to the parameter space.
        This may introduce additional errors.
        Returns:

        """
        return False


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
        low_all, high_all = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        param_dim = low_all.numel()
        low_p, high_p = low_all[dim].item(), high_all[dim].item()

        # total samples including padding
        total = n_samples + 2 * extend


        # linear spacing in [low_p, high_p]
        rng = high_p - low_p
        spacing = rng / (n_samples - 1) if n_samples > 1 else 0
        start = low_p - extend * spacing
        values = torch.linspace(start,
                                    high_p + extend * spacing,
                                    total) + shift * spacing

        # embed into full parameter vectors
        params = torch.zeros((total, param_dim), dtype=torch.float32)
        params[:, dim] = values
        #convert params from sample space to parameter space
        params = self.sample_space_convert_forward(params)
        return params
