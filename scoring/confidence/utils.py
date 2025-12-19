"""
This module contains wrappers for PyTorch models to output both the last layer and a specified intermediate layer.
This is used for SplitConfidence which can redirect the different outputs to different confidence estimation modules.
"""
import weakref  # NEW

import torch
import torch.nn as nn

class ModelInputOutputWrapper(nn.Module):
    """
    Wrapper for a PyTorch model to output both the last layer and multiple intermediate layers input, output, or both.
    """
    def __init__(self,
                 model,
                 target_layer_names,
                 flatten=False,
                 concat=False,
                 capture_modes='output',
                 entry_indices=0,
                 return_final: bool = True,
                 return_y: bool = False,
                 feature_reducers: dict = None  # NEW: {(layer_name, mode)->nn.Module}
                 ):
        """
        Wrapper for a PyTorch model to output both the last layer and specified intermediate layers' inputs, outputs, or both.

        Args:
            model (nn.Module)
            target_layer_names (str|list)
            flatten (bool)
            concat (bool)
            capture_modes (str|list) 'input'|'output'|'both'
            entry_indices (int|list)
            return_final (bool): include model final output in returned tuple (default True)
            return_y (bool): include provided y (passed to forward) in returned tuple (default False)
            feature_reducers (dict): optional per-(layer,mode) reducers applied after flattening
        """
        super(ModelInputOutputWrapper, self).__init__()
        self.model = model
        self.flatten = flatten
        self.concat= concat
        self.output_tuple = True
        self.return_final = return_final
        self.return_y = return_y
        self.feature_reducers = feature_reducers or {}  # NEW

        # Handle target_layer_names as single string or list
        if isinstance(target_layer_names, str):
            self.target_layer_names = [target_layer_names]
            self.output_tuple = False
        else:
            self.target_layer_names = target_layer_names

        # Handle capture_modes and entry_indices
        if isinstance(capture_modes, str):
            self.capture_modes = [capture_modes] * len(self.target_layer_names)
        else:
            if len(capture_modes) != len(self.target_layer_names):
                raise ValueError("capture_modes must have the same length as target_layer_names")
            self.capture_modes = capture_modes

        if isinstance(entry_indices, int):
            self.entry_indices = [entry_indices] * len(self.target_layer_names)
        else:
            if len(entry_indices) != len(self.target_layer_names):
                raise ValueError("entry_indices must have the same length as target_layer_names")
            self.entry_indices = entry_indices

        self.layer_data = list(zip(self.target_layer_names, self.capture_modes, self.entry_indices))
        self.target_layer_outputs = {}
        self.target_layer_inputs = {}
        self._hook_handles = []  # NEW: store handles for cleanup
        self._register_hooks()

    def _register_hooks(self):
        # NEW: use weak ref to break cycles: module -> hook fn -> self -> model -> module
        weak_self = weakref.ref(self)

        def create_hook_input(weak_self, layername, entry_idx):
            def hook_fn(_, inputs, outputs):
                s = weak_self()
                if s is None:
                    return
                s.target_layer_inputs[layername] = inputs[entry_idx]
            return hook_fn

        def create_hook_output(weak_self, layername):
            def hook_fn(_, inputs, outputs):
                s = weak_self()
                if s is None:
                    return
                s.target_layer_outputs[layername] = outputs
            return hook_fn

        found_layers = set()
        for layer_name, mode, entry_idx in self.layer_data:
            # Find target module
            module = None
            for name, mod in self.model.named_modules():
                if name == layer_name:
                    module = mod
                    break

            if not module:
                raise ValueError(f"Layer '{layer_name}' not found in model")

            if mode == 'input' or mode == 'both':
                h = module.register_forward_hook(create_hook_input(weak_self, layer_name, entry_idx))
                self._hook_handles.append(h)
            if mode == 'output' or mode == 'both':
                h = module.register_forward_hook(create_hook_output(weak_self, layer_name))
                self._hook_handles.append(h)
            found_layers.add(layer_name)

        if len(set(self.target_layer_names)) != len(set(found_layers)):
            missing = set(self.target_layer_names) - found_layers
            raise ValueError(f"Layers {missing} not found in model")


    def forward(self, x, y=None):
        """
        Forward pass.
        For each (layer, mode):
          mode 'input'  -> append input tensor
          mode 'output' -> append output tensor
          mode 'both'   -> append input THEN output
        Ordering matches the order of target_layer_names / capture_modes with 'both' expanded.
        """
        try:
            self.target_layer_inputs.clear()
            self.target_layer_outputs.clear()

            final_output = self.model(x)

            # Collect in order (cannot rely on dict because of duplicate layer names or 'both')
            outputs_list = []
            for layer_name, mode, _entry_idx in self.layer_data:
                if mode in ('input', 'both'):
                    tensor_in = self.target_layer_inputs[layer_name]
                    # Some layers' inputs are tuples; conventionally the first element is the tensor.
                    if isinstance(tensor_in, tuple):
                        tensor_in = tensor_in[0]
                    key = (layer_name, 'input')
                    if key in self.feature_reducers:
                        red = self.feature_reducers[key]
                        # NEW: do NOT flatten before reducer; reducer handles shape
                        tensor_in = red(tensor_in.float())
                    elif self.flatten:
                        b = tensor_in.shape[0]
                        tensor_in = tensor_in.reshape(b, -1)
                    outputs_list.append(tensor_in)

                if mode in ('output', 'both'):
                    tensor_out = self.target_layer_outputs[layer_name]
                    # Some layers return a tuple; conventionally the first element is the feature tensor.
                    if isinstance(tensor_out, tuple):
                        tensor_out = tensor_out[0]
                    key = (layer_name, 'output')
                    if key in self.feature_reducers:
                        red = self.feature_reducers[key]
                        # NEW: do NOT flatten before reducer; reducer handles shape
                        tensor_out = red(tensor_out.float())
                    elif self.flatten:
                        b = tensor_out.shape[0]
                        tensor_out = tensor_out.reshape(b, -1)
                    outputs_list.append(tensor_out)

            # Flatten if requested (applies after reducers)
            if self.flatten:
                for i, t in enumerate(outputs_list):
                    b = t.shape[0]
                    outputs_list[i] = t.reshape(b, -1)

            if self.concat:
                embeddings = torch.cat(outputs_list, dim=-1)
            else:
                if not self.output_tuple and len(outputs_list) == 1:
                    embeddings = outputs_list[0]
                else:
                    embeddings = outputs_list

            # Build return structure according to flags
            # Cases (priority order):
            # return_final & return_y -> (embeddings, final, y)
            # return_final only       -> (embeddings, final)
            # return_y only           -> (embeddings, y)
            # neither                 -> embeddings
            if self.return_final and self.return_y:
                ret = (embeddings, final_output, y)
            elif self.return_final:
                ret = (embeddings, final_output)
            elif self.return_y:
                ret = (embeddings, y)
            else:
                ret = embeddings
        except Exception as e:
            raise e
        finally:
            self.target_layer_inputs.clear()
            self.target_layer_outputs.clear()


        return ret

    def clear(self):
        """Explicitly clear hooks and feature reducers to break reference cycles."""
        self.remove_hooks()
        if self.feature_reducers:
            for key in list(self.feature_reducers.keys()):
                # The value is a nn.Module, so we delete it to remove references
                del self.feature_reducers[key]
            self.feature_reducers.clear()
        self.feature_reducers = None

    #on delete unregister hooks
    def remove_hooks(self):
        """NEW: explicitly detach all hooks."""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    def __del__(self):
        # NEW: ensure hooks are removed to avoid dangling references
        self.remove_hooks()

    def __enter__(self):
        return self  # NEW: allow context manager usage

    def __exit__(self, exc_type, exc, tb):
        self.remove_hooks()
        return False  # do not suppress exceptions



import torch
import torch.nn as nn

class ModelInputWrapper(nn.Module):
    """
    Wrapper for a PyTorch model to output both the final output and the input to a specified intermediate layer.
    """
    def __init__(self, model, target_layer_name, entry_index=0, flatten=False):
        """
        Args:
            model (nn.Module): The original PyTorch model.
            target_layer_name (str): The name of the intermediate layer to capture input from.
            entry_index (int): Index of the input tuple to capture (default is 0).
            flatten (bool): Whether to flatten the intermediate input to shape (batch_size, -1).
        """
        super().__init__()
        self.model = model
        self.target_layer_name = target_layer_name
        self.entry_index = entry_index
        self.flatten = flatten
        self._captured_input = None
        self._hook_handles = []  # NEW: store handles for cleanup

        # Register a forward hook with closure
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                h = module.register_forward_hook(self._make_hook())  # NEW
                self._hook_handles.append(h)  # NEW

    def _make_hook(self):
        # NEW: weak ref to avoid reference cycle
        weak_self = weakref.ref(self)
        entry_index = self.entry_index
        def hook_fn(module, input, output):
            s = weak_self()
            if s is None:
                return
            s._captured_input = input[entry_index]
        return hook_fn

    def forward(self, x):
        self._captured_input = None  # Reset before forward
        final_output = self.model(x)

        if self._captured_input is None:
            raise RuntimeError(f"Input for layer '{self.target_layer_name}' not captured.")

        intermediate = self._captured_input
        if self.flatten:
            batch_size = intermediate.shape[0]
            intermediate = intermediate.reshape(batch_size, -1)

        # NEW: clear captured ref after use
        self._captured_input = None
        return intermediate, final_output

    # NEW: explicit hook cleanup
    def remove_hooks(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    def __del__(self):
        self.remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove_hooks()
        return False


class ModelOutputWrapper(nn.Module):
    """
    Wrapper for a PyTorch model to output both the final output and the output from a specified intermediate layer.
    """
    def __init__(self, model, target_layer_name, flatten=False):
        """
        Args:
            model (nn.Module): The original PyTorch model.
            target_layer_name (str): The name of the intermediate layer to capture output from.
            flatten (bool): Whether to flatten the intermediate output to shape (batch_size, -1).
        """
        super().__init__()
        self.model = model
        self.target_layer_name = target_layer_name
        self.flatten = flatten
        self._captured_output = None
        self._hook_handles = []  # NEW: store handles for cleanup

        # Register a forward hook with closure
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                h = module.register_forward_hook(self._make_hook())  # NEW
                self._hook_handles.append(h)  # NEW

    def _make_hook(self):
        # NEW: weak ref to avoid reference cycle
        weak_self = weakref.ref(self)
        def hook_fn(module, input, output):
            s = weak_self()
            if s is None:
                return
            s._captured_output = output
        return hook_fn

    def forward(self, x):
        self._captured_output = None  # Reset before forward
        final_output = self.model(x)

        if self._captured_output is None:
            raise RuntimeError(f"Output for layer '{self.target_layer_name}' not captured.")

        intermediate = self._captured_output
        if self.flatten:
            batch_size = intermediate.shape[0]
            intermediate = intermediate.reshape(batch_size, -1)

        # NEW: clear captured ref after use
        self._captured_output = None
        return intermediate, final_output

    # NEW: explicit hook cleanup
    def remove_hooks(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    def __del__(self):
        self.remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove_hooks()
        return False


class SequentialModelWrapper(nn.Module):
    """
    Wrapper for a Sequential PyTorch model to output both the last layer and a specified intermediate layer.
    #TODO test
    """
    def __init__(self, model, split_indexs, flatten=False, concat=False):
        """
        Wrapper for a Sequential PyTorch model to output both the last layer and a specified intermediate layer.

        Args:
            model (nn.Sequential): The original PyTorch Sequential model.
            split_index (int): The index at which to split the model.
            flatten (bool): Whether to flatten the intermediate output to (batch_size, -1).
        """
        super(SequentialModelWrapper, self).__init__()
        if not isinstance(model, nn.Sequential):
            raise TypeError("The model must be an instance of torch.nn.Sequential.")
        self.output_tuple = True
        if isinstance(split_indexs, int):
            split_index = [split_indexs]
            self.output_tuple = False

        self.concat = concat

        #for negative idices counting from the end
        for i in range(len(split_indexs)):
            if split_indexs[i] < 0:
                split_indexs[i] = len(model) + split_indexs[i]

            if not (0 <= split_indexs[i] < len(model)):
                raise ValueError("split_index must be a valid index within the model layers.")

        # Split the model into two parts
        self.parts = []
        for i in range(len(split_indexs)):
            if i == 0:
                self.parts.append(nn.Sequential(*list(model.children())[:split_indexs[i] + 1]))
            else:
                self.parts.append(nn.Sequential(*list(model.children())[split_indexs[i - 1] + 1:split_indexs[i] + 1]))

        self.parts.append(nn.Sequential(*list(model.children())[split_indexs[-1] + 1:]))
        self.flatten = flatten

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (output of the intermediate layer, output of the final layer)
        """
        intermediate_outputs = []
        for part in self.parts[:-1]:
            x = part(x)
            if self.flatten:
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1)
            intermediate_outputs.append(x)
        final_output = self.parts[-1](x)

        if self.concat:
            intermediate_outputs = torch.cat(intermediate_outputs, dim=-1)
            return intermediate_outputs, final_output

        if self.output_tuple:
            return intermediate_outputs, final_output
        else:
            return intermediate_outputs[-1], final_output

class ModelInputOutputWrapperOnDemand(nn.Module):
    """
    On-demand wrapper for a PyTorch model to output both the last layer and multiple intermediate layers input, output, or both.
    Hooks are registered only during forward and removed immediately after.
    """
    def __init__(self,
                 model,
                 target_layer_names,
                 flatten=False,
                 concat=False,
                 capture_modes='output',
                 entry_indices=0,
                 return_final: bool = True,
                 return_y: bool = False,
                 feature_reducers: dict = None
                 ):
        """
        Same as ModelInputOutputWrapper but hooks are not persistent.
        """
        super(ModelInputOutputWrapperOnDemand, self).__init__()
        self.model = model
        self.flatten = flatten
        self.concat = concat
        self.output_tuple = True
        self.return_final = return_final
        self.return_y = return_y
        self.feature_reducers = feature_reducers or {}

        if isinstance(target_layer_names, str):
            self.target_layer_names = [target_layer_names]
            self.output_tuple = False
        else:
            self.target_layer_names = target_layer_names

        if isinstance(capture_modes, str):
            self.capture_modes = [capture_modes] * len(self.target_layer_names)
        else:
            if len(capture_modes) != len(self.target_layer_names):
                raise ValueError("capture_modes must have the same length as target_layer_names")
            self.capture_modes = capture_modes

        if isinstance(entry_indices, int):
            self.entry_indices = [entry_indices] * len(self.target_layer_names)
        else:
            if len(entry_indices) != len(self.target_layer_names):
                raise ValueError("entry_indices must have the same length as target_layer_names")
            self.entry_indices = entry_indices

        self.layer_data = list(zip(self.target_layer_names, self.capture_modes, self.entry_indices))

    def forward(self, x, y=None):
        # Register hooks on-demand
        target_layer_outputs = {}
        target_layer_inputs = {}
        hook_handles = []

        weak_self = weakref.ref(self)

        def create_hook_input(weak_self, layername, entry_idx):
            def hook_fn(_, inputs, outputs):
                s = weak_self()
                if s is None:
                    return
                target_layer_inputs[layername] = inputs[entry_idx]
            return hook_fn

        def create_hook_output(weak_self, layername):
            def hook_fn(_, inputs, outputs):
                s = weak_self()
                if s is None:
                    return
                target_layer_outputs[layername] = outputs
            return hook_fn

        found_layers = set()
        for layer_name, mode, entry_idx in self.layer_data:
            module = None
            for name, mod in self.model.named_modules():
                if name == layer_name:
                    module = mod
                    break
            if not module:
                raise ValueError(f"Layer '{layer_name}' not found in model")

            if mode == 'input' or mode == 'both':
                h = module.register_forward_hook(create_hook_input(weak_self, layer_name, entry_idx))
                hook_handles.append(h)
            if mode == 'output' or mode == 'both':
                h = module.register_forward_hook(create_hook_output(weak_self, layer_name))
                hook_handles.append(h)
            found_layers.add(layer_name)

        if len(set(self.target_layer_names)) != len(set(found_layers)):
            missing = set(self.target_layer_names) - found_layers
            raise ValueError(f"Layers {missing} not found in model")

        try:
            # Forward pass
            final_output = self.model(x)

            # Collect outputs (same logic as original)
            outputs_list = []
            for layer_name, mode, _entry_idx in self.layer_data:
                if mode in ('input', 'both'):
                    tensor_in = target_layer_inputs[layer_name]
                    if isinstance(tensor_in, tuple):
                        tensor_in = tensor_in[0]
                    key = (layer_name, 'input')
                    if key in self.feature_reducers:
                        red = self.feature_reducers[key]
                        tensor_in = red(tensor_in.float())
                    elif self.flatten:
                        b = tensor_in.shape[0]
                        tensor_in = tensor_in.reshape(b, -1)
                    outputs_list.append(tensor_in)

                if mode in ('output', 'both'):
                    tensor_out = target_layer_outputs[layer_name]
                    if isinstance(tensor_out, tuple):
                        tensor_out = tensor_out[0]
                    key = (layer_name, 'output')
                    if key in self.feature_reducers:
                        red = self.feature_reducers[key]
                        tensor_out = red(tensor_out.float())
                    elif self.flatten:
                        b = tensor_out.shape[0]
                        tensor_out = tensor_out.reshape(b, -1)
                    outputs_list.append(tensor_out)

            if self.flatten:
                for i, t in enumerate(outputs_list):
                    b = t.shape[0]
                    outputs_list[i] = t.reshape(b, -1)

            if self.concat:
                embeddings = torch.cat(outputs_list, dim=-1)
            else:
                if not self.output_tuple and len(outputs_list) == 1:
                    embeddings = outputs_list[0]
                else:
                    embeddings = outputs_list

            # Build return structure according to flags
            # Cases (priority order):
            # return_final & return_y -> (embeddings, final, y)
            # return_final only       -> (embeddings, final)
            # return_y only           -> (embeddings, y)
            # neither                 -> embeddings
            if self.return_final and self.return_y:
                ret = (embeddings, final_output, y)
            elif self.return_final:
                ret = (embeddings, final_output)
            elif self.return_y:
                ret = (embeddings, y)
            else:
                ret = embeddings

        finally:
            # Remove hooks immediately, always
            for h in hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass

        return ret

if __name__ == '__main__':
    # Define a simple Sequential model: Linear → ReLU → Linear
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )

    # Dummy input
    x = torch.randn(2, 10)

    # 1. ModelInputOutputWrapper: capture layer '0' input and layer '2' output
    wrapper_io = ModelInputOutputWrapper(
        model,
        target_layer_names=['0', '2'],
        flatten=False,
        concat=False,
        capture_modes=['input', 'output'],
        entry_indices=[0, 0],
    )
    inter_io, out_io = wrapper_io(x)
    print("ModelInputOutputWrapper:")
    for i, t in enumerate(inter_io):
        print(f"  intermediate[{i}] shape: {t.shape}")
    print(f"  final output shape: {out_io.shape}\n")

    # 2. ModelInputWrapper: capture '0' input
    wrapper_i = ModelInputWrapper(
        model,
        target_layer_name='0',
        entry_index=0,
        flatten=True,
    )
    inputs_i, out_i = wrapper_i(x)
    print("ModelInputWrapper:")
    print(f"  input shape: {inputs_i.shape}")
    print(f"  final output shape: {out_i.shape}\n")

    # 3. ModelOutputWrapper: capture '2' output
    wrapper_o = ModelOutputWrapper(
        model,
        target_layer_name='2',
        flatten=True,
    )
    outputs_o, out_o = wrapper_o(x)
    print("ModelOutputWrapper:")
    print(f"  output shape: {outputs_o.shape}")
    print(f"  final output shape: {out_o.shape}\n")

    # 4. SequentialModelWrapper: split after layer 0 and 1
    seq_wrapper = SequentialModelWrapper(
        model,
        split_indexs=[0, 2],
        flatten=True,
        concat=False,
    )
    inter_seq, out_seq = seq_wrapper(x)
    print("SequentialModelWrapper:")
    for i, t in enumerate(inter_seq):
        print(f"  intermediate part[{i}] shape: {t.shape}")
    print(f"  final output shape: {out_seq.shape}")

    # 5. ModelInputOutputWrapper with concat=True
    wrapper_io_concat = ModelInputOutputWrapper(
        model,
        target_layer_names=['0', '2'],
        flatten=True,
        concat=True,
        capture_modes=['input', 'output'],
        entry_indices=[0, 0],
    )
    combined_io, out_io_concat = wrapper_io_concat(x)
    print("ModelInputOutputWrapper (concat=True):")
    print(f"  combined intermediate shape: {combined_io.shape}")
    print(f"  final output shape: {out_io_concat.shape}\n")

    # 6. SequentialModelWrapper with concat=True
    seq_wrapper_concat = SequentialModelWrapper(
        model,
        split_indexs=[0, 2],
        flatten=True,
        concat=True,
    )
    combined_seq, out_seq_concat = seq_wrapper_concat(x)
    print("SequentialModelWrapper (concat=True):")
    print(f"  combined intermediate parts shape: {combined_seq.shape}")
    print(f"  final output shape: {out_seq_concat.shape}")

    # Benchmarking with a bigger model (ResNet18) and CUDA
    import torchvision.models as models
    import time

    num_runs = 100  # Reduced for larger model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available; benchmarking on CPU.")
    
    # Bigger model: ResNet18
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    )
    
    # Bigger batch size: 32, with input shape for ResNet (3, 224, 224)
    x = torch.randn(32, 3, 224, 224).to(device)

    # Wrapper 1: Persistent hooks
    wrapper1 = ModelInputOutputWrapper(
        model,
        target_layer_names=['0', '2'],  # conv1 and layer1.0.conv1 in ResNet18
        flatten=False,
        concat=False,
        capture_modes=['output', 'output'],  # Adjusted for output capture
        entry_indices=[0, 0],
    ).to(device)

    # Wrapper 2: On-demand hooks
    wrapper2 = ModelInputOutputWrapperOnDemand(
        model,
        target_layer_names=['0', '2'],
        flatten=False,
        concat=False,
        capture_modes=['output', 'output'],
        entry_indices=[0, 0],
    ).to(device)

    # Benchmark wrapper1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = wrapper1(x)
        end.record()
        torch.cuda.synchronize()
        time1 = start.elapsed_time(end) / num_runs
    else:
        start = time.time()
        for _ in range(num_runs):
            _ = wrapper1(x)
        time1 = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark wrapper2
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = wrapper2(x)
        end.record()
        torch.cuda.synchronize()
        time2 = start.elapsed_time(end) / num_runs
    else:
        start = time.time()
        for _ in range(num_runs):
            _ = wrapper2(x)
        time2 = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark wrapper1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = wrapper1(x)
        end.record()
        torch.cuda.synchronize()
        time1 = start.elapsed_time(end) / num_runs
    else:
        start = time.time()
        for _ in range(num_runs):
            _ = wrapper1(x)
        time1 = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark wrapper2
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = wrapper2(x)
        end.record()
        torch.cuda.synchronize()
        time2 = start.elapsed_time(end) / num_runs
    else:
        start = time.time()
        for _ in range(num_runs):
            _ = wrapper2(x)
        time2 = (time.time() - start) / num_runs * 1000  # ms

    print(f"Benchmark results ({num_runs} runs on {device}):")
    print(f"  ModelInputOutputWrapper (persistent hooks): {time1:.3f} ms per forward")
    print(f"  ModelInputOutputWrapperOnDemand (on-demand hooks): {time2:.3f} ms per forward")