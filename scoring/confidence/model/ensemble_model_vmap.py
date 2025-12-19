import copy
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, stack_module_state
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map  # Ensure tree_map is imported
from ensemble_model import SimpleStackEnsemble  # noqa

class VmapEnsemble(nn.Module):
    def __init__(self, modules: list[nn.Module]):
        super().__init__()
        self.n_estimators = len(modules)
        if not modules:
            raise ValueError("Input 'modules' list cannot be empty.")

        # Store one of the original modules as a template for functional_call.
        # It's used by functional_call to define the operations, but its state is ignored.
        self.func_mod_template = copy.deepcopy(modules[0])  # Or just modules[0] if no side effects

        # Get (stacked_params_pytree, stacked_buffers_pytree) from stack_module_state
        # Each leaf tensor in these PyTrees will have a new leading dimension (size n_estimators).
        stacked_params_pytree, stacked_buffers_pytree = stack_module_state(modules)

        # Process and register stacked parameters
        flat_params, self._p_spec = tree_flatten(stacked_params_pytree)
        self.param_leaves = nn.ParameterList([nn.Parameter(p) for p in flat_params])

        # Process and register stacked buffers
        flat_buffers, self._b_spec = tree_flatten(stacked_buffers_pytree)
        self.buffer_leaf_names = []
        for i, b in enumerate(flat_buffers):
            name = f"_buffer_leaf_{i}"
            self.register_buffer(name, b)
            self.buffer_leaf_names.append(name)

        # Create PyTree-structured in_dims for vmap.
        self._p_in_dims_spec = tree_map(lambda _: 0, stacked_params_pytree)
        self._b_in_dims_spec = tree_map(lambda _: 0, stacked_buffers_pytree)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct the PyTree of parameters from stored ParameterList
        current_flat_params = list(self.param_leaves)
        current_params_pytree = tree_unflatten(current_flat_params, self._p_spec)

        # Reconstruct the PyTree of buffers from registered buffers
        current_flat_buffers = [getattr(self, name) for name in self.buffer_leaf_names]
        current_buffers_pytree = tree_unflatten(current_flat_buffers, self._b_spec)

        # Define the function to be vmapped.
        # It takes per-estimator params and buffers (slices from the stacked PyTrees)
        # and the shared input x.
        def single_call_fn(params_one_estimator, buffers_one_estimator, current_x):
            # functional_call uses self.func_mod_template as the stateless module structure
            # and applies the provided params_one_estimator and buffers_one_estimator.
            return functional_call(
                self.func_mod_template,
                (params_one_estimator, buffers_one_estimator),
                current_x
            )


        out = vmap(
            single_call_fn,
            in_dims=(self._p_in_dims_spec, self._b_in_dims_spec, None),
            out_dims=0  # Output will have estimators as the first dimension
        )(current_params_pytree, current_buffers_pytree, x)
        
        return out.permute(1, 0, *range(2, out.ndim))

if __name__ == "__main__":
    # simple smoke test
    torch.manual_seed(0)
    batch, feat, hidden = 4, 8, 5
    base = nn.Sequential(nn.Linear(feat, hidden), nn.ReLU())
    modules = [copy.deepcopy(base) for _ in range(3)]
    # SimpleStackEnsemble
    simple = SimpleStackEnsemble(modules)
    # VmapEnsemble
    vm = VmapEnsemble(modules)
    x = torch.randn(batch, feat)
    out1 = simple(x)
    out2 = vm(x)
    print("SimpleStackEnsemble output shape:", out1.shape)  # (batch, 3, hidden)
    print("VmapEnsemble output shape:      ", out2.shape)  # (batch, 3, hidden)
    assert out1.shape == out2.shape
    print("Shapes match, test passed.")
