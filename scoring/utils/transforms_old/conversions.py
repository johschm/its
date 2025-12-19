# utils/transforms_old/conversions.py
import collections
import functools
import math
from typing import Callable, Dict, Tuple

import torch
from utils.transforms_old.base import Transform

from utils.transforms_old.rotation import (
    Rotation3DEuler,
    RotationQuaternion3D,
    Rotation2D,
    RotationComplex2D, RotationSkew3D, _RotationSkewGeneral, RoationSkewGeneral3D, RotationVector3D, Rotation3DEulerIn,
    _RotationEulerGenereal3D, euler_to_quaternion, quaternion_to_euler, angle_to_complex, complex_to_angle,
    quaternion_to_skew_3d, skew_3d_to_quaternion, skew_3d_to_skew_general, skew_general_to_skew_3d
)
from utils.transforms_old.convert import ConvertTransform



class DynamicConvertTransform:
    """
    Maintains a registry of “direct” conversion functions between specific Transform instances.
    If a direct (start→end) converter is not found, it will try to find (end→start) and call it in reverse.
    """
    # The registry maps (start_instance, end_instance) → Callable[[Tensor], Tensor]
    _conversion_registry: Dict[Tuple[Transform, Transform], Callable[[torch.Tensor], torch.Tensor]] = {}
    @classmethod
    def register(
        cls,
        source: Transform,
        target: Transform,
        func: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """
        Register a conversion function that goes from the *specific* instance `source` → `target`.
        Example:
            euler_inst = Rotation3DEuler()
            quat_inst  = RotationQuaternion3D()
            DynamicConvertTransform.register(euler_inst, quat_inst, euler_to_quaternion)
        """
        DynamicConvertTransform.register_with_transitive(source, target, func)
        #cls._conversion_registry[(source, target)] = func
        #DynamicConvertTransform.add_transitive_conversions()

    @classmethod
    def forward_func(cls, start: Transform, end: Transform) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Returns the conversion function that goes from `start` → `end`.
        """
        while isinstance(start, ConvertTransform):
            # If start is a ConvertTransform, we can use its `convert_forward` method
            start = start.transform_end
        while isinstance(end, ConvertTransform):
            # If start is a ConvertTransform, we can use its `convert_forward` method
            end = end.transform_end

        # Check if both are the same
        if start is end:
            return lambda x: x

        direct_key = (start, end)
        if direct_key in cls._conversion_registry:
            return cls._conversion_registry[direct_key]

        raise ValueError(
            f"No conversion registered for instances: {start.__class__.__name__} → {end.__class__.__name__}"
        )

    def backward_func(cls, start: Transform, end: Transform) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Returns the conversion function that goes from `end` → `start`.
        This is simply the forward function with `start` and `end` swapped.
        """
        return cls.forward_func(end, start)


    @classmethod
    def forward(
        cls,
        param: torch.Tensor,
        start: Transform,
        end: Transform
    ) -> torch.Tensor:
        """
        Convert `param` from `start` → `end`. If no direct converter (start→end) exists,
        try (end→start) and call that function (assuming it is the exact inverse).
        """
        return cls.forward_func(start, end)(param)

    @classmethod
    def back(
        cls,
        param: torch.Tensor,
        start: Transform,
        end: Transform
    ) -> torch.Tensor:
        """
        “Backward” conversion is simply forward with `start` and `end` swapped:
            back(param, A, B) = forward(param, B, A)
        """
        return cls.forward(param, end, start)

    @classmethod
    def register_with_transitive(
        cls,
        source: object,
        target: object,
        func: Callable[[object], object]
    ) -> None:
        """
        Register a conversion source→target, then *incrementally* add any
        new transitive conversions that arise because of this single new edge.

        PRE-CONDITION:  before you call register(u, v, f_uv), cls._conversion_registry
        must already include *every* transitive conversion built so far.

        POST-CONDITION:  after calling register(u, v, f_uv), the registry includes
        (u→v) plus exactly those new (x→y) pairs for which there is now a path
        x → … → u → v → … → y that did NOT exist before.  Each new function is
        composed on the fly from the already-registered pieces.
        """
        # 1) Snapshot the old keys so we don’t “see” any of our own inserts in the loops below.
        old_keys = list(cls._conversion_registry.keys())

        # 2) Insert the new DIRECT edge (u→v) itself.
        u, v = source, target
        cls._conversion_registry[(u, v)] = func

        # 3) Collect all old “predecessors of u”: P = { x | (x, u) in old_keys }, EXCLUDING x=u.
        preds: list[object] = []
        for (x, y) in old_keys:
            if y is u and x is not u:
                preds.append(x)

        # 4) Collect all old “successors of v”: S = { y | (v, y) in old_keys }, EXCLUDING y=v.
        succs: list[object] = []
        for (x, y) in old_keys:
            if x is v and y is not v:
                succs.append(y)

        # 5) For convenience, we’ll want to refer to the newly-added (u→v) function many times:
        f_u_v = func

        # 6) 1st: For every x in P, create (x → v) by composing (x→u) with (u→v).
        for x in preds:
            # grab the old f_x_u from the snapshot
            f_x_u = cls._conversion_registry[(x, u)]
            # If (x, v) does not already exist, register it now.
            if (x, v) not in cls._conversion_registry and x is not v:
                def make_fxv(fxu, fuv):
                    def x_to_v(x_in):
                        return fuv(fxu(x_in))
                    return x_to_v

                f_x_v = make_fxv(f_x_u, f_u_v)
                cls._conversion_registry[(x, v)] = f_x_v

        # 7) 2nd: For every y in S, create (u → y) by composing (u→v) with (v→y).
        for y in succs:
            f_v_y = cls._conversion_registry[(v, y)]
            # If (u, y) does not already exist, register it now.
            if (u, y) not in cls._conversion_registry and u is not y:
                def make_fuy(fuv, fvy):
                    def u_to_y(u_in):
                        return fvy(fuv(u_in))
                    return u_to_y

                f_u_y = make_fuy(f_u_v, f_v_y)
                cls._conversion_registry[(u, y)] = f_u_y

        # 8) 3rd: For each combination (x in P) and (y in S), compose (x→u), (u→v), (v→y) to get (x → y).
        for x in preds:
            f_x_u = cls._conversion_registry[(x, u)]
            # We *re-grab* f_u_v = func
            for y in succs:
                f_v_y = cls._conversion_registry[(v, y)]
                if (x, y) not in cls._conversion_registry and x is not y:
                    def make_fxy(fxu, fuv, fvy):
                        def x_to_y(x_in):
                            inter_u = fxu(x_in)      # x→u
                            inter_v = fuv(inter_u)   # u→v
                            return fvy(inter_v)      # v→y
                        return x_to_y

                    f_x_y = make_fxy(f_x_u, f_u_v, f_v_y)
                    cls._conversion_registry[(x, y)] = f_x_y

    optimized_mapping = {
        Rotation3DEuler: RotationQuaternion3D,
        _RotationEulerGenereal3D: RotationQuaternion3D,
        Rotation2D: RotationComplex2D,
    }



    @classmethod
    def add_transitive_conversions(cls) -> None:
        """
        Add transitive conversions to the registry more efficiently.
        Instead of checking every triple (a, b, c) in O(n³), we do a BFS
        from each source node to find all reachable targets and compose functions incrementally.
        """
        # Snapshot the current registry so we only treat direct edges as the "graph".
        # We will read from this adjacency list during the BFS.
        adjacency: dict[type, list[tuple[type, Callable]]] = {}
        for (src, dst), func in cls._conversion_registry.items():
            adjacency.setdefault(src, []).append((dst, func))

        # For each source 'a', do a BFS over the DIRECT adjacency list
        # and compose functions along each path.
        for a in list(adjacency.keys()):
            # visited_targets[c] will hold the composed function for a -> c
            visited_targets: dict[type, Callable] = {}
            queue: collections.deque[tuple[type, Callable]] = collections.deque()

            # 1) Initialize BFS queue with all direct neighbors of `a`
            for (b, f_ab) in adjacency.get(a, []):
                if b == a:
                    # Skip self-loops if any (you could register or ignore)
                    continue
                visited_targets[b] = f_ab
                queue.append((b, f_ab))

            # 2) Process the queue: whenever we pop (current, f_a_to_current),
            #    look at each direct neighbor (current -> next) to build a -> next.
            while queue:
                current, f_a_to_current = queue.popleft()
                for (nxt, f_current_to_nxt) in adjacency.get(current, []):
                    if nxt == a:
                        # avoid cycles back to 'a'
                        continue
                    if nxt in visited_targets:
                        # already found a→nxt by a shorter/previous path
                        continue

                    # Compose a→current with current→nxt to get a→nxt
                    def make_composed(f1, f2):
                        # f1: a→current, f2: current→nxt
                        def composed(x):
                            return f2(f1(x))

                        return composed

                    new_func = make_composed(f_a_to_current, f_current_to_nxt)
                    visited_targets[nxt] = new_func

                    # Register the new transitive conversion a→nxt
                    cls.register(a, nxt, new_func)

                    # Enqueue (nxt, a→nxt) to continue BFS from `nxt`
                    queue.append((nxt, new_func))


# --- Register the rotation conversions ---------------------------------------

DynamicConvertTransform.register(
    Rotation3DEuler,
    RotationQuaternion3D,
    euler_to_quaternion
)
DynamicConvertTransform.register(
    RotationQuaternion3D,
    Rotation3DEuler,
    quaternion_to_euler
)
DynamicConvertTransform.register(
    Rotation2D,
    RotationComplex2D,
    angle_to_complex
)
DynamicConvertTransform.register(
    RotationComplex2D,
    Rotation2D,
    complex_to_angle
)

DynamicConvertTransform.register(
    RotationQuaternion3D,
    RotationSkew3D,
    quaternion_to_skew_3d
)

DynamicConvertTransform.register(
    RotationSkew3D,
    RotationQuaternion3D,
    skew_3d_to_quaternion
)
DynamicConvertTransform.register(
    RotationSkew3D,
    RoationSkewGeneral3D,
    skew_3d_to_skew_general
)
DynamicConvertTransform.register(
    RoationSkewGeneral3D,
    RotationSkew3D,
    skew_general_to_skew_3d
)

DynamicConvertTransform.register(
    RotationVector3D, RotationSkew3D,
    lambda x: x
)
DynamicConvertTransform.register(
    RotationSkew3D, RotationVector3D,
    lambda x: x
)


DynamicConvertTransform.register(
    RotationQuaternion3D, RotationVector3D,
    quaternion_to_skew_3d
)
DynamicConvertTransform.register(
    RotationVector3D, RotationQuaternion3D,
    skew_3d_to_quaternion
)
def euler3d_to_euler_general(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 3D Euler angles (roll, pitch, yaw) to general Euler angles.
    Expects `param` to have shape (..., 3) and returns a tensor of shape (..., 3).
    """
    y, p, r = param.unbind(-1)
    # Convert to general Euler angles (yaw, pitch, roll)
    return torch.stack([y, -p, r], dim=-1)

DynamicConvertTransform.register(
    Rotation3DEuler, _RotationEulerGenereal3D,
    euler3d_to_euler_general
)

DynamicConvertTransform.register(
    _RotationEulerGenereal3D, Rotation3DEuler,
    euler3d_to_euler_general
)
#TODO directed transform
#TODO a way to convert individual directed transforms_old to quaternion rotation. Or at least multiple complex ones.
#maybe using user defined transform. Or simpoy reinterpreting it.




#now for intrinsic 3d to enxtrinsic 3d
def _euler_extrinsic_to_intrinsic(param: torch.Tensor) -> torch.Tensor:
    """
    Convert extrinsic Euler angles (yaw, pitch, roll) to intrinsic angles.
    Expects `param` to have shape (..., 3) and returns a tensor of shape (..., 3).
    """
    yaw, pitch, roll = param.unbind(-1)
    # Convert extrinsic (yaw, pitch, roll) to intrinsic (roll, pitch, yaw)
    return torch.stack([roll, pitch, yaw], dim=-1)

DynamicConvertTransform.register(
    Rotation3DEuler, Rotation3DEulerIn,
    _euler_extrinsic_to_intrinsic
)

def _intrinsic_to_extrinsic_euler(param: torch.Tensor) -> torch.Tensor:
    """
    Convert intrinsic Euler angles (roll, pitch, yaw) to extrinsic angles.
    Expects `param` to have shape (..., 3) and returns a tensor of shape (..., 3).
    """
    roll, pitch, yaw = param.unbind(-1)
    # Convert intrinsic (roll, pitch, yaw) to extrinsic (yaw, pitch, roll)
    return torch.stack([yaw, pitch, roll], dim=-1)
DynamicConvertTransform.register(
    Rotation3DEulerIn, Rotation3DEuler,
    _intrinsic_to_extrinsic_euler
)

import roma

DynamicConvertTransform.register(
    Rotation3DEuler,
    RotationSkew3D,
    lambda x: roma.euler.euler_to_rotvec(("zyx",x)
    ))

DynamicConvertTransform.register(
    RotationSkew3D,
    Rotation3DEuler,
    lambda x: roma.euler.rotvec_to_euler(("zyx",x)
))







def _test_rotvector_skew_roundtrip():
    """
    Test that RotationVector3D ↔ RotationSkew3D roundtrip works correctly.
    """
    # Generate random axis-angle vectors
    rotvectors = torch.randn(1000, 3)
    skew = DynamicConvertTransform.forward(rotvectors, RotationVector3D, RotationSkew3D)
    rotvectors_back = DynamicConvertTransform.back(skew, RotationVector3D, RotationSkew3D)
    # Check that the values are in domain
    matrix1 = RotationVector3D(rotvectors)
    matrix2 = RotationSkew3D(skew)
    matrix3 = RotationVector3D(rotvectors_back)
    argmax = (matrix1 - matrix2).abs().mean(dim=(1, 2)).argmax()
    print(matrix1[argmax])
    print(matrix2[argmax])
    print((matrix2 - matrix1).abs()[argmax])
    assert torch.allclose(matrix1, matrix2, atol=1e-4), "Matrix representations do not match after conversion."
    assert torch.allclose(matrix1, matrix3, atol=1e-4), "Matrix representations do not match after back-conversion."

def _test_dynamic_euler_quaternion_roundtrip():
    angles = torch.randn(1000,3)
    angles = torch.clamp(angles, -torch.pi + 0.0001, torch.pi - 0.0001)
    angles_tt   = DynamicConvertTransform.forward(param=angles, start=Rotation3DEuler, end=RotationQuaternion3D)
    angles_back = DynamicConvertTransform.back(angles_tt, Rotation3DEuler, RotationQuaternion3D)
    #print angle at max

    #now print roation matrices
    matrix1 = Rotation3DEuler(angles)
    matrix2 = RotationQuaternion3D(angles_tt)
    matrix3 = Rotation3DEuler(angles_back)
    #check that the values are in domain

    argmax = (matrix1-matrix3).abs().mean(dim=(1, 2)).argmax()
    print(matrix1[argmax])
    print(matrix3[argmax])
    print((matrix3-matrix1).abs()[argmax])



    assert torch.allclose(matrix1, matrix2, atol=1e-6), "Matrix representations do not match after conversion."
    assert torch.allclose(matrix1, matrix3, atol=1e-3), "Matrix representations do not match after back-conversion."


    #assert torch.allclose(angles, angles_back, atol=1e-6), "Angles do not match after conversion."


def _test_dynamic_angle_complex_roundtrip():
    angles = torch.randn(1000, 1)
    angles = torch.clamp(angles, -torch.pi+0.0001, torch.pi-0.0001)  # Ensure angles are within [-π, π]
    c = DynamicConvertTransform.forward(angles, Rotation2D, RotationComplex2D)
    angles_back = DynamicConvertTransform.back(c, Rotation2D, RotationComplex2D)
    argmax = (angles - angles_back).abs().mean(dim=1).argmax()
    print(f"Max angle difference at index {argmax.item()}: {angles[argmax].item()} vs {angles_back[argmax].item()}")
    assert torch.allclose(angles, angles_back, atol=1e-5)
    matrix1 = Rotation2D(angles)
    matrix2 = RotationComplex2D(c)
    matrix3 = Rotation2D(angles_back)
    assert torch.allclose(matrix1, matrix2, atol=1e-6), "Matrix representations do not match after conversion."
    assert torch.allclose(matrix1, matrix3, atol=1e-5), "Matrix representations do not match after back-conversion."

def _test_dynamic_skew_quaternion_roundtrip():
    quats = torch.randn(1000, 4)
    quats = torch.nn.functional.normalize(quats, dim=-1)  # Normalize to unit quaternions
    skew = DynamicConvertTransform.forward(quats, RotationQuaternion3D, RotationSkew3D)
    quats_back = DynamicConvertTransform.back(skew, RotationQuaternion3D, RotationSkew3D)

    matrix1 = RotationQuaternion3D(quats)
    matrix2 = RotationSkew3D(skew)
    matrix3 = RotationQuaternion3D(quats_back)

    argmax = (matrix1 - matrix2).abs().mean(dim=(1, 2)).argmax()
    print(matrix1[argmax]-matrix2[argmax])
    assert torch.allclose(matrix1, matrix2, atol=1e-4), "Matrix representations do not match after conversion."
    assert torch.allclose(matrix1, matrix3, atol=1e-4), "Matrix representations do not match after back-conversion."

if __name__ == "__main__":
    _test_dynamic_euler_quaternion_roundtrip()
    _test_dynamic_angle_complex_roundtrip()
    print("All conversion tests passed!")
    _test_dynamic_skew_quaternion_roundtrip()
