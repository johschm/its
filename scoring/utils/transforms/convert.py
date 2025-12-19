from abc import abstractmethod
from typing import Optional

import torch

from utils.transforms.base import Transform

class ConvertTransform(Transform):
    """
    Transforms that does boundary violations, sampling etc. in the original domain but converts the parameters to a new representation.
    """

    def __init__(self, transform_start, transform_end):
        self.transform_start = transform_start
        self.transform_end = transform_end
        self.warned = False  # Flag to warn only once about using calc_bounds as a fallback

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        return self.transform_end.matrix(param)

    def support_calc_bounds(self) -> bool:
        """
        Indicates if the transform implements calc_bounds_fallback.
        """
        return False

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
        #print a warning on the first call that bounds are only fallback as we can not simply convert the original parameters bounds.
        if self.warned is False:
            print("Warning: Using ConvertTransform with calc_bounds as a fallback. This produces only as many params as start transform. So conversion must be done afterward..")
        self.warned = True
        return self.transform_start.calc_bounds(domain, dtype=dtype, device=device)


    def supports_sobol(self) -> bool:
        # Delegate to start (sampling domain)
        return self.transform_start.supports_sobol()

    def supports_orbit(self) -> bool:
        # Orbits (discrete param traversal) logically belong to sampling domain
        return self.transform_start.supports_orbit()

    def sobol_to_param(self, sparam: torch.Tensor,domain) -> torch.Tensor:
        """
        Map Sobol sample(s) in [0,1]^k for the start transform into final (end) representation.
        1. Decode start-space parameters via its own sobol_to_param.
        2. Convert forward to end representation.
        """
        start_param = self.transform_start.sobol_to_param(sparam,domain)
        return self.convert_forward(start_param)

    def normalize_parameters(self, param: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the parameters to a standard range.
        :param param: The parameters to normalize.
        :return: The normalized parameters.
        """
        return self.transform_end.normalize_parameters(param)


    def project_parameters(self, param: torch.Tensor, domain, reflect=True) -> torch.Tensor:
        """
        Projects the parameters into the domain.
        :param param: The parameters to project.
        :param domain: The domain to project into.
        :return: The projected parameters

        Args:
            reflect:  If True, reflect the parameters into the domain. If False, clip the parameters to the domain.
        """
        #first convert back to the start transform, project there and convert forward again.
        param_start = self.convert_backward(param)
        param_start = self.transform_start.project_parameters(param_start, domain, reflect=reflect)
        return self.convert_forward(param_start)





    def param_size(self) -> int:
        """
        Returns the size of the parameter vector.
        :return: The size of the parameter vector.
        """
        return self.transform_end.param_size()


    def sample_space_param_size(self):
        return self.transform_start.sample_space_param_size()




    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        sampling_param = self.convert_backward(param)
        return self.transform_start.boundary_violation(sampling_param, domain)


    def distance_bounded(self, param1: torch.Tensor, param2: torch.Tensor, domain=None) -> torch.Tensor:
        """
        Calculate the distance between two parameters, taking into account the bounds of the transformation.
        This is a fallback for search methods that do not support projecting into the bounds.
        """
        return self.transform_start.distance_bounded(self.convert_backward(param1), self.convert_backward(param2), domain=domain)

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
            dim:
            n_samples:
            domain:
            extend:
            shift:

        Returns:

        """

        return self.convert_forward(self.transform_start.orbit(n_samples=n_samples, domain=domain,dim=dim, extend=extend, shift=shift))

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

    def sample_param(self,batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """
        Sample a parameter for the transformation.
        :return: The sampled parameter.
        """
        return self.convert_forward(self.transform_start.sample_param(batch_size,domain, device=device, dtype=dtype))





    def sample_space_interval(self) -> Optional[tuple[float, float]]:
        return self.transform_start.sample_space_interval()

    def sample_space_project_parameters(self, sample_space_params: torch.Tensor, domain, reflect=True) -> torch.Tensor:
        return self.transform_start.sample_space_project_parameters(sample_space_params, domain, reflect=reflect)

    def sample_space_reproject_to_interval(self, param: torch.Tensor) -> torch.Tensor:
        return self.transform_start.sample_space_reproject_to_interval(param)

    def sample_space_convert_forward(self, param: torch.Tensor) -> torch.Tensor:
        param_start = self.transform_start.sample_space_convert_forward(param)
        return self.convert_forward(param_start)

    def sample_space_convert_backward(self, param: torch.Tensor) -> torch.Tensor:
        param_end = self.convert_backward(param)
        return self.transform_start.sample_space_convert_backward(param_end)

    def convert_forward(self, param: torch.Tensor) -> torch.Tensor:
        from utils.transforms.conversions import DynamicConvertTransform

        return DynamicConvertTransform.forward(param, self.transform_start, self.transform_end)

    def convert_backward(self, param: torch.Tensor) -> torch.Tensor:
        from utils.transforms.conversions import DynamicConvertTransform

        return DynamicConvertTransform.back(param, self.transform_start, self.transform_end)






if __name__ == "__main__":
    import torch
    import math
    from utils.transforms.rotation import Rotation2D, RotationComplex2D
    from utils.transforms.conversions import DynamicConvertTransform

    # Test conversion between Rotation2D and RotationComplex2D
    print("Testing ConvertTransform with Rotation2D → RotationComplex2D")

    # Create test parameters
    angle = torch.tensor([[math.pi / 4]])  # 45 degrees in radians

    # Expected complex representation [cos(θ), sin(θ)]
    expected_complex = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4)]])


    # Define forward and backward conversion functions for testing
    def angle_to_complex(p, _, __):
        """Convert angle to complex number [cos(θ), sin(θ)]"""
        return torch.cat([torch.cos(p), torch.sin(p)], dim=-1)


    def complex_to_angle(p, _, __):
        """Convert complex number to angle"""
        return torch.atan2(p[..., 1:], p[..., :1])


    # Override the conversion functions for testing
    DynamicConvertTransform.forward = staticmethod(angle_to_complex)
    DynamicConvertTransform.back = staticmethod(complex_to_angle)

    # Create a ConvertTransform
    ct = ConvertTransform(Rotation2D, RotationComplex2D)

    # Test conversions
    complex_param = ct.convert_forward(angle)
    angle_back = ct.convert_backward(complex_param)

    print(f"Angle → Complex: {angle.item():.4f} → {complex_param.squeeze().tolist()}")
    print(f"Complex → Angle: {complex_param.squeeze().tolist()} → {angle_back.item():.4f}")

    assert torch.allclose(complex_param, expected_complex, atol=1e-6), "Forward conversion failed"
    assert torch.allclose(angle_back, angle, atol=1e-6), "Backward conversion failed"

    # Test matrix method delegation
    matrix_start = Rotation2D.matrix(angle)
    matrix_end = RotationComplex2D.matrix(complex_param)
    matrix_convert = ct.matrix(complex_param)

    assert torch.allclose(matrix_end, matrix_convert, atol=1e-6), "Matrix method delegation failed"
    print("✓ Matrix method delegation works correctly")

    # Test other delegated methods
    assert ct.param_size() == RotationComplex2D.param_size(), "Param size delegation failed"
    print("✓ Param size delegation works correctly")


    # Test orbit delegation with conversion
    orig_orbit = Rotation2D.orbit(4, math.pi)
    converted_orbit = ct.orbit(4, math.pi)
    expected_orbit = torch.stack([torch.cos(orig_orbit.squeeze(-1)),
                                  torch.sin(orig_orbit.squeeze(-1))], dim=-1)
    assert torch.allclose(converted_orbit, expected_orbit, atol=1e-6), "Orbit conversion failed"
    print("✓ Orbit delegation with conversion works correctly")

    # Test sampling with conversion
    sample_orig = Rotation2D.sample_param(math.pi)
    sample_converted = ct.sample_param(math.pi)
    # Test dict interface
    d = ct.as_dict()
    for key in ("matrix", "param_size", "project_parameters", "calc_bounds", "orbit", "interval"):
        assert key in d, f"Key {key} missing in as_dict()"
    print("✓ Dict interface works correctly")

    print("\nAll ConvertTransform tests passed!")