import threading
import types
from copy import deepcopy, copy
from typing import MutableMapping, Any, List
import hashlib
import time

import torch
from laplace import Laplace, LLLaplace
from laplace.utils import FeatureExtractor
from torch.nn.utils import vector_to_parameters

from confidence.model.base_model import ModelBasedConfidence

# Always import the robust model fingerprint from embedding_cache
from embedding_cache import LayerEmbeddingCache as _LayerEmbeddingCacheHelper

# --- In-memory cache with timeout and background cleanup ---
_LAPLACE_FIT_CACHE: dict = {}  # key -> (laplace_obj, last_access_time)
_LAPLACE_CACHE_TIMEOUT = 120  # seconds (5 minutes)
_LAPLACE_CACHE_SWEEP_INTERVAL = 60  # seconds
_LAPLACE_CACHE_SWEEPER_THREAD = None
_LAPLACE_CACHE_SWEEPER_STOP = threading.Event()
_LAPLACE_CACHE_LOCK = threading.Lock()


def _laplace_cache_sweeper():
    global _LAPLACE_CACHE_SWEEPER_THREAD
    while not _LAPLACE_CACHE_SWEEPER_STOP.is_set():
        time.sleep(_LAPLACE_CACHE_SWEEP_INTERVAL)
        now = time.time()
        with _LAPLACE_CACHE_LOCK:
            keys_to_del = [k for k, (_, t) in _LAPLACE_FIT_CACHE.items() if now - t > _LAPLACE_CACHE_TIMEOUT]
            for k in keys_to_del:
                # Get the laplace object before deletion
                laplace_obj, _ = _LAPLACE_FIT_CACHE[k]
                # Clear any cached gradients or intermediate states
                if hasattr(laplace_obj, 'model'):
                    for param in laplace_obj.model.parameters():
                        param.grad = None
                # Delete the entry
                del _LAPLACE_FIT_CACHE[k]
                del laplace_obj
                print(f"[LaplaceCache] Removed expired cache entry: {k}")

            # Force garbage collection after cleanup
            if keys_to_del:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # If cache is empty, stop the thread
            if not _LAPLACE_FIT_CACHE:
                _LAPLACE_CACHE_SWEEPER_STOP.set()
                _LAPLACE_CACHE_SWEEPER_THREAD = None
                print("[LaplaceCache] Cache empty, sweeper thread exiting.")
                return


def _ensure_laplace_cache_sweeper():
    global _LAPLACE_CACHE_SWEEPER_THREAD
    if _LAPLACE_CACHE_SWEEPER_THREAD is None or not _LAPLACE_CACHE_SWEEPER_THREAD.is_alive():
        _LAPLACE_CACHE_SWEEPER_STOP.clear()
        t = threading.Thread(target=_laplace_cache_sweeper, daemon=True)
        t.start()
        _LAPLACE_CACHE_SWEEPER_THREAD = t


def _maybe_stop_laplace_cache_sweeper():
    global _LAPLACE_CACHE_SWEEPER_THREAD
    if not _LAPLACE_FIT_CACHE and _LAPLACE_CACHE_SWEEPER_THREAD is not None:
        _LAPLACE_CACHE_SWEEPER_STOP.set()
        _LAPLACE_CACHE_SWEEPER_THREAD = None


# function to inject. Not sure why it was missing.
def _nn_functional_samples(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        n_samples: int = 100,
        generator: torch.Generator | None = None,
        **model_kwargs,
) -> torch.Tensor:

    fs = list()

    feats = None
    for sample in self.sample(n_samples, generator):
        vector_to_parameters(sample, self.model.last_layer.parameters())

        if feats is None:
            # Cache features at the first iteration
            f, feats = self.model.forward_with_features(
                X.to(self._device), **model_kwargs
            )
        else:
            # Used the cached features for the rest iterations
            f = self.model.last_layer(feats)

        fs.append(f.detach() if not self.enable_backprop else f)

    vector_to_parameters(self.mean, self.model.last_layer.parameters())
    fs = torch.stack(fs)

    return fs


class OutputCacheWrapper(torch.nn.Module):
    """
    Wrapper that enables model that have multiple ouputs like intermediate features to work with torch-laplace.
    This wrapper causes non logit output the be saved which can later be gotten using get_cached_output.
    """
    def __init__(self, base_model:torch.nn.Module, index: int):
        super().__init__()
        self.base_model = base_model
        self.index = index
        self._cache: List[Any] = []
        self.call_count = 0
        self.output_was_tuple = False
        self.caching_enabled = False

    def enable_caching(self):
        """
        Enable or disable caching of outputs.
        """
        self.caching_enabled = True

    def disable_caching(self):
        """
        Disable caching of outputs and clear cache properly.
        """
        self.caching_enabled = False
        # Delete tensors explicitly before clearing list
        for cached_item in self._cache:
            if isinstance(cached_item, torch.Tensor):
                del cached_item
            elif isinstance(cached_item, (tuple, list)):
                for item in cached_item:
                    if isinstance(item, torch.Tensor):
                        del item
        self._cache.clear()
        self.call_count = 0


    def forward(self, x):
        outputs = self.base_model(x)
        if isinstance(outputs, torch.Tensor):
            raise ValueError(
                "The base model must return a tuple or list of outputs. "
                "If it returns a single tensor, wit should not have to be wrapped with this class."
            )

        # Convert to list for uniform caching
        indexed_outputs = outputs[self.index]
        if not self.caching_enabled:
            # If caching is disabled, return the indexed output directly
            return indexed_outputs

        if isinstance(outputs, tuple):
            # If the output is a tuple, take all elemetns but the indexed one
            outputs = outputs[:self.index] + outputs[self.index + 1:]
        else:
            outputs = outputs.pop(self.index)  # Remove the indexed output
        self._cache.append(outputs)
        self.call_count += 1
        return indexed_outputs

    def get_cached_output(self,stack=True,mean=True,stack_dim=0):
        if self.call_count == 0:
            raise ValueError("No outputs cached yet. Call forward() first.")
        elif self.call_count ==1:
            out = self._cache[0]
            self._cache.clear()
            self.call_count = 0
            return out

        #use _pytree to stack outputs
        if mean:
            out = torch.utils._pytree.tree_map(
                lambda *args: torch.stack(args, dim=stack_dim), *self._cache
            )
        else:
            out = copy(self._cache)

        if mean and stack:
            out = torch.utils._pytree.tree_map(
                lambda t: t.mean(dim = stack_dim), out
            )

        # Clear cache and explicitly delete references
        for cached_item in self._cache:
            if isinstance(cached_item, torch.Tensor):
                del cached_item


        #clear cache
        self._cache.clear()
        self.call_count = 0
        return out



#LLLaplace._nn_predictive_samples = LaplaceModelSamplingConfidence._nn_predictive_samples

from contextlib import contextmanager

@contextmanager
def temporarily_enable_grads(model: torch.nn.Module):
    """
    Temporarily sets all parameters' requires_grad=True,
    then restores their original state afterwards — no side effects.
    """
    requires_grad_backup = {p: p.requires_grad for p in model.parameters()}
    try:
        for p in model.parameters():
            p.requires_grad = True
        yield model
    finally:
        for p, req_grad in requires_grad_backup.items():
            p.requires_grad = req_grad


class LaplaceModelSamplingConfidence(ModelBasedConfidence):
    """
    Wraps torch-laplace Laplace with:
    - full Laplace init signature
    - fit(...) that records kwargs for fine-tuning
    - pred_type modes ('probit', 'bridge', 'bridge_norm', 'mc')
    - set_pred_type to switch sampling strategy
    """
    def __init__(
        self,
        base_model: torch.nn.Module,
        index = None,  # index of the output
        confidence=None,

        #sampling settings
        samples: int = 10, #only used for mc pred type
        pred_type: str = "glm",  # glm or nn for classification. nn only supports mc carle link aprx
        link_approx: str = "probit",  # 'bridge', 'bridge_norm', or 'mc' for classification In addition we supoprt none in which case we simply return the distribution.
        # probit approximates the expected sigmoid of the logits directly from the distribution. So the output is always a softmax. Bridge similary transform the gaussian to a dirichlet distribution.
        diagonal_output =False,  # diagonal glm output, only for 'glm' pred_type

        #return type settings
        index_laplace: int = None, # index for the laplace fitting will use the same as normal index if None


        softmax: bool = True, #only used for mc pred type, if true we apply softmax to the output, if false we return the raw logits.
        average: bool = True, #unly used for mc pred type, if true we average over samples, if false we return the samples directly.

        #laplace settings
        hessian_structure: str = "kron",# {'diag', 'kron', 'full', 'lowrank', 'gp'}
        subset_of_weights: str = "last_layer",

        # fit settings
        method="marglik",  # does not need a val loader "gridsearch" is the alternative
        kwargs_opt_prior: dict = None,  # see torch-laplace optimize_prior_precision for all kwargs.

        # keep the same
        subnetwork_indices = None,
        backend: str = None, #see BaseLaplace for details on backend

        # caching
        enable_fit_cache: bool = True,  # enable in-memory caching of fitted Laplace objects keyed by base model hash
        # laplace only adds paramaters that requires grad, so we need to unfreeze them.(This also must be done for loading from cache from state dict)
        set_all_parameters_trainable: bool = True,
        prior_precision: float = 1.0, #note not supported for cahcing
    ):
        super().__init__(base_model, confidence=confidence, index=index)
        print("prior precision at init:", prior_precision)
        # instantiate Laplace with same API as torch-laplace
        self.temperature = 1.0
        # Save a reference to the original base model for hashing
        self._base_model_for_hash = base_model
        self.enable_fit_cache = enable_fit_cache

        # Always use the robust fingerprint from embedding_cache
        self._base_model_init_hash = _LayerEmbeddingCacheHelper._model_fingerprint(base_model)

        self.index = index  # index of the output to cache, if None, no caching is done
        self.index_laplace = index_laplace or index  # index for the laplace fitting, if None, use the same as index

        if self.index_laplace is not None:
            # wrap the base model to cache the output at the index
            model = OutputCacheWrapper(base_model, self.index_laplace)
        else:
            model = base_model

        #set all parameters to require grad
        self.set_all_parameters_trainable = set_all_parameters_trainable

        if set_all_parameters_trainable:
            grad_context = temporarily_enable_grads(base_model)
        else:
            grad_context = nullcontext = contextmanager(lambda: (yield base_model))

        with grad_context:
            if subnetwork_indices is not None and subset_of_weights != "subnetwork":
                self.laplace = Laplace(
                    model,
                    likelihood="classification",
                    hessian_structure=hessian_structure,
                    prior_precision=prior_precision,
                    subset_of_weights=subset_of_weights,
                    subnetwork_indices=subnetwork_indices,
                    backend=backend,
                    enable_backprop=True,
                    temperature=self.temperature
                )
            else:
                self.laplace = Laplace(
                    model,
                    likelihood="classification",
                    hessian_structure=hessian_structure,
                    prior_precision=prior_precision,
                    subset_of_weights=subset_of_weights,
                    backend=backend,
                    enable_backprop=True,
                    temperature=self.temperature
                )
        #print precision after init
        print("prior precision after init:", self.laplace.prior_precision)

        #if type of laplace is Lllaplace inject function
        if isinstance(self.laplace, LLLaplace):
            print("Warning: Injecting _nn_predictive_samples into LLLaplace instance. This is a workaround for compatibility with torch-laplace. Not sure why it is not reimplemented even though parent implemnts it.")
            self.laplace._nn_functional_samples = types.MethodType(
                _nn_functional_samples, self.laplace
            ) #not working method of base class is still called

        self.pred_type = pred_type
        self.link_approx = link_approx
        self.samples = samples
        self.average = average
        self._fit_params = {}
        self.fit_other_kwargs = kwargs_opt_prior if kwargs_opt_prior is not None else {}
        self.diagonal_output = diagonal_output

        self.softmax = softmax  # only used for mc pred type, if true we apply softmax to the output, if false we return the raw logits.
        self.method = method
        self.experimental_last_layer_opt =False #TODO implement a way to cache intermediate values and only replace last layer weights if this is set to true.
        self.fitted = False
        self._cache_key = None
        self._last_cache_refresh = 0.0

    @torch.no_grad()
    def _debug_hessian_stats(self):
        """
        Compute and print eigenvalue statistics for a Laplace Hessian.
        Handles KronDecomposed, dense tensors, and small convertible matrices.
        """
        H = getattr(self.laplace, "H", None)
        if H is None:
            print("[DEBUG] No Hessian found.")
            return

        try:
            # Case 1: KronDecomposed - extract eigenvalues without dense materialization
            if hasattr(H, "eigenvalues"):
                all_eigs = []
                # Loop over each block (e.g., weights, bias)
                for block_idx, block in enumerate(H.eigenvalues):
                    if isinstance(block, (list, tuple)):  # Handle both list and tuple
                        if len(block) == 2:
                            # Kronecker-factored (weights)
                            eigA, eigB = [t.detach().cpu().flatten() for t in block]
                            eig_products = torch.outer(eigA, eigB).flatten()
                            all_eigs.append(eig_products)
                            print(f"[DEBUG] Block {block_idx}: weights (A⊗B), shapes {eigA.shape} × {eigB.shape}")
                        elif len(block) == 1:
                            # Bias or single-factor block
                            eig_single = block[0].detach().cpu().flatten()
                            all_eigs.append(eig_single)
                            print(f"[DEBUG] Block {block_idx}: single (bias), shape {eig_single.shape}")
                        else:
                            print(f"[DEBUG] Block {block_idx}: unexpected length {len(block)}")
                    elif isinstance(block, torch.Tensor):
                        all_eigs.append(block.detach().cpu().flatten())
                        print(f"[DEBUG] Block {block_idx}: tensor eigenvalues, shape {block.shape}")
                    else:
                        print(f"[DEBUG] Block {block_idx}: unexpected type {type(block)}")

                if not all_eigs:
                    print("[DEBUG] No eigenvalues extracted from KronDecomposed structure.")
                    return

                eigs = torch.cat(all_eigs)
                eig_min = eigs.min().item()
                eig_max = eigs.max().item()
                eig_mean = eigs.mean().item()
                neg_count = (eigs < 0).sum().item()

                print("\n=== Hessian Eigenvalue Diagnostics ===")
                print(f"Hessian type: KronDecomposed")
                print(f"Total eigenvalues: {eigs.numel():,}")
                print(f"Range: [{eig_min:.3e}, {eig_max:.3e}]")
                print(f"Mean: {eig_mean:.3e}")
                print(f"Negatives: {neg_count}")
                print("======================================\n")

                del eigs, all_eigs
                return

            # Case 2: Dense tensor (rare)
            if isinstance(H, torch.Tensor):
                H_mat = H.detach().cpu()
                eigs = torch.linalg.eigvalsh(H_mat)
                eig_min = eigs.min().item()
                eig_max = eigs.max().item()
                eig_mean = eigs.mean().item()
                neg_count = (eigs < 0).sum().item()

                print("\n=== Hessian Eigenvalue Diagnostics ===")
                print(f"Hessian type: Dense tensor")
                print(f"Total eigenvalues: {eigs.numel():,}")
                print(f"Range: [{eig_min:.3e}, {eig_max:.3e}]")
                print(f"Mean: {eig_mean:.3e}")
                print(f"Negatives: {neg_count}")
                print("======================================\n")

                del H_mat, eigs
                return

            # Case 3: Fallback — try to convert only if small
            if hasattr(H, "to_matrix"):
                H_mat = H.to_matrix()
                if H_mat.numel() < 1e6:  # only safe for small matrices
                    eigs = torch.linalg.eigvalsh(H_mat.detach().cpu())
                    eig_min = eigs.min().item()
                    eig_max = eigs.max().item()
                    eig_mean = eigs.mean().item()
                    neg_count = (eigs < 0).sum().item()

                    print("\n=== Hessian Eigenvalue Diagnostics ===")
                    print(f"Hessian type: Converted small matrix")
                    print(f"Total eigenvalues: {eigs.numel():,}")
                    print(f"Range: [{eig_min:.3e}, {eig_max:.3e}]")
                    print(f"Mean: {eig_mean:.3e}")
                    print(f"Negatives: {neg_count}")
                    print("======================================\n")

                    del H_mat, eigs
                else:
                    print("[DEBUG] Hessian too large to convert safely to dense matrix.")
            else:
                print("[DEBUG] Unknown Hessian structure; cannot extract stats safely.")

        except Exception as e:
            print(f"[DEBUG] Could not compute Hessian stats: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @property
    def model(self):
        """
        Return the base model, which is either the original model or the wrapped one.
        """
        if isinstance(self.laplace.model, OutputCacheWrapper):
            return self.laplace.model
        return self.laplace.model.model

    def _make_cache_key(self) -> str:
        # Always use the stable initial base-model hash from embedding_cache
        base_hash = self._base_model_init_hash

        # Make fit_other_kwargs deterministic (sort items if dict)
        if isinstance(self.fit_other_kwargs, dict):
            try:
                extras_items = tuple(sorted(self.fit_other_kwargs.items()))
            except Exception:
                extras_items = repr(self.fit_other_kwargs)
        else:
            extras_items = repr(self.fit_other_kwargs)

        extras = "|".join(
            [
                str(getattr(self.laplace, "hessian_structure", "")),
                str(getattr(self.laplace, "subset_of_weights", "")),
                str(self.method),
                str(extras_items),
                str(self.pred_type),
                str(self.link_approx),
            ]
        )
        return f"{base_hash}|{extras}"
    @torch.no_grad()
    def fit(self, train_loader, validation_loader=None):
        """
        Fine-tune the Laplace posterior.
        """
        if self.enable_fit_cache:
            _ensure_laplace_cache_sweeper()
            try:
                key = self._make_cache_key()
                self._cache_key = key  # Save for later refresh
                with _LAPLACE_CACHE_LOCK:
                    cached = _LAPLACE_FIT_CACHE.get(key, None)
                    if cached is not None:
                        print("Precision before loading from cache:", self.laplace.prior_precision)
                        precision_before = self.laplace.prior_precision
                        self.laplace = cached[0]
                        _LAPLACE_FIT_CACHE[key] = (self.laplace, time.time())
                        if isinstance(self.laplace, LLLaplace):
                            self.laplace._nn_functional_samples = types.MethodType(
                                _nn_functional_samples, self.laplace
                            )
                        self.fitted = True
                        print(f"Loaded fitted Laplace from in-memory cache (key={key})")
                        # Overwrite prior_precision if link_approx is "none"
                        if self.method == "none":
                            self.laplace.prior_precision = precision_before
                        print("Using cached Laplace fit, skipping re-fitting. Precision:", self.laplace.prior_precision)
                        return
            except Exception:
                pass

        if self.index_laplace is not None:
            self.model.disable_caching()
        with torch.enable_grad():
            if self.set_all_parameters_trainable:
                grad_context = temporarily_enable_grads(self._base_model_for_hash)
            else:
                grad_context = nullcontext = contextmanager(lambda: (yield self._base_model_for_hash))
            with grad_context:
                self.laplace.fit(train_loader)
                if self.method != "none":
                    self.laplace.optimize_prior_precision(
                        method=self.method, pred_type=self.pred_type, val_loader=validation_loader,
                        link_approx=self.link_approx, **self.fit_other_kwargs, init_prior_prec=self.laplace.prior_precision
                    )
            self.fitted = True
            self._debug_hessian_stats()

        if self.enable_fit_cache:
            key = self._make_cache_key()
            self._cache_key = key  # Save for later refresh
            with _LAPLACE_CACHE_LOCK:
                _LAPLACE_FIT_CACHE[key] = (self.laplace, time.time())
                _ensure_laplace_cache_sweeper()
            print(f"Stored fitted Laplace into in-memory cache (key={key})")

    def set_pred_type(self, pred_type = None,link_approx = None,diagonal_output = None):
        """
        Update sampling mode: 'probit', 'bridge', 'bridge_norm', or 'mc'.
        """
        if self.method != "marglik":
            raise ValueError(
                "Cannot change pred_type after fitting with method != 'marglik'. "
                "Please re-initialize the LaplaceModelSamplingConfidence instance."
            )

        raise ValueError("set_pred_type is deprecated. Please re-initialize the LaplaceModelSamplingConfidence instance with the desired pred_type and link_approx.")

        if pred_type is not None:
            self.pred_type = pred_type
        if link_approx is not None:
            self.link_approx = link_approx
        if diagonal_output is not None:
            self.diagonal_output = diagonal_output

        #try optimizing prior precision again if pred_type is changed
        self.laplace.optimize_prior_precision(
            method=self.method, pred_type=self.pred_type, link_approx=self.link_approx, **self.fit_other_kwargs
        )



    def forward_no_link_approx(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Forward pass without link approximation. Always assumes glm pred_type.
        """
        logits,logits_var = self.laplace._glm_predictive_distribution(x,diagonal_output=self.diagonal_output)
        tup = (logits,logits_var)


        if self.index_laplace is not None:
            output = self.model.get_cached_output(stack=False, mean=False, stack_dim=1)
            if isinstance(output, tuple):
                output = list(output)
                output.insert(self.index_laplace, tup)
                output = tuple(output)
            else:
                if isinstance(output, list):
                    output.insert(self.index_laplace, tup)
                else:
                    output[self.index_laplace] = tup
        else:
            output = tup

        confidence = self.confidence(output, y)
        if self.index is None:
            return confidence, output
        self.model.disable_caching()
        return confidence, output[self.index]

    def set_backprop(self, backprop: bool):
        """
        Set whether to backpropagate through the Laplace posterior.
        """
        self.laplace.enable_backprop = backprop
        if type(self.laplace.model) == FeatureExtractor:
            self.laplace.model.enable_backprop = backprop






    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        # Refresh cache timeout only when forward is called, and only if enough time has passed
        if self.enable_fit_cache and self._cache_key is not None:
            now = time.time()
            if now - self._last_cache_refresh > 1.0:  # avoid excessive locking
                with _LAPLACE_CACHE_LOCK:
                    entry = _LAPLACE_FIT_CACHE.get(self._cache_key, None)
                    if entry is not None:
                        _LAPLACE_FIT_CACHE[self._cache_key] = (entry[0], now)
                self._last_cache_refresh = now

        if not self.fitted:
            raise RuntimeError("The Laplace model has not been fitted yet. Call .fit() before forward.")
        if self.index is not None:
            self.model.enable_caching()  # enable caching for the model
        if self.pred_type == "nn" and self.link_approx != "mc":
            self.model.disable_caching()  # disable caching for the model if not using mc
            raise ValueError(
                "For pred_type 'nn', link_approx must be 'mc'. Use forward_average for other link approximations."
            )
        if self.pred_type == "glm" and self.link_approx == "none":
            return self.forward_no_link_approx(x, y)

        if self.link_approx == "mc":
            return self.forward_monte_carlo(x, y)
        elif self.link_approx in ["probit", "bridge", "bridge_norm"] and self.pred_type == "glm":
            return self.forward_average(x, y)
        else:
            self.model.disable_caching()  # disable caching for the model if not using mc
            raise ValueError(
                f"Unsupported link approximation: {self.link_approx} for pred_type {self.pred_type}. "
                "Use 'mc' for NN or 'probit', 'bridge', 'bridge_norm' for GLM."
            )


    def forward_monte_carlo(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Forward pass with Monte Carlo sampling.
        """
        if self.link_approx != "mc":
            raise ValueError(
                "For pred_type 'nn', link_approx must be 'mc'. Use forward_average for other link approximations."
            )

        # Ensure caching is enabled if needed
        if self.index_laplace is not None:
            self.model.enable_caching()

        #in this case simply call forward from laplace
        if self.softmax:
            mc_output = self.laplace.predictive_samples(
                x,
                pred_type=self.pred_type,
                diagonal_output=self.diagonal_output,
                n_samples=self.samples,
            )
        else:
            mc_output = self.laplace.functional_samples(
                x,
                pred_type=self.pred_type,
                diagonal_output=self.diagonal_output,
                n_samples=self.samples,
            )

        if self.average:
            mc_output = mc_output.mean(dim=0)
        else:
            mc_output = mc_output.permute(1, 0, *range(2, mc_output.dim()))

        if self.index_laplace is not None:
            output = self.model.get_cached_output(stack=True, mean=self.average, stack_dim=1)
            if isinstance(output, tuple):
                output = list(output)
                output.insert(self.index_laplace, mc_output)
                output = tuple(output)
            else:
                if isinstance(output, list):
                    output.insert(self.index_laplace, mc_output)
                else:
                    output[self.index_laplace] = mc_output
        else:
            output = mc_output



        confidence = self.confidence(output, y)
        #always average output for output
        if not self.average:
            #average here for ouput
            output = output.mean(dim=1)


        if self.index is None:
            return confidence, output

        self.model.disable_caching()
        return confidence, output[self.index]


    def forward_average(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Forward pass with averaging over samples.
        """
        #in this case simply call forward from laplace
        if not self.softmax:
            raise ValueError(
                "Non Softmax output is only supported for Monte Carlo sampling. "
            )


        average_output= self.laplace(x,pred_type=self.pred_type,link_approx=self.link_approx,n_samples=self.samples,
                                         diagonal_output=self.diagonal_output)

        if self.index_laplace is not None:
            outputs = self.model.get_cached_output(
                stack=False, mean=False, stack_dim=1
            )
            if isinstance(outputs, tuple):
                outputs = list(outputs)
                outputs.insert(self.index_laplace, average_output)
                outputs = tuple(outputs)
            else:
                if isinstance(outputs, list):
                    outputs.insert(self.index_laplace, average_output)
                else:
                    outputs[self.index_laplace] = average_output
        else:
            outputs = average_output



        confidence = self.confidence(outputs, y)

        if self.index is None:
            return confidence, outputs
        self.model.disable_caching()
        return confidence, outputs[self.index]

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # --- dual-output head module ---
    class DualOutputModule(torch.nn.Module):
        def __init__(self, in_features: int, hidden: int, num_classes: int):
            super().__init__()
            self.shared = torch.nn.Sequential(
                torch.nn.Linear(in_features, hidden),
                torch.nn.ReLU(),
            )
            self.classifier = torch.nn.Linear(hidden, num_classes)
            self.regressor = torch.nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor):
            h = self.shared(x)
            logits = self.classifier(h)
            return logits, h

    # --- larger dummy dataset for testing caching ---
    # make dataset large so fit is noticeably expensive
    X = torch.randn(5000, 64)
    y = torch.randint(0, 10, (5000,))
    train_loader = DataLoader(TensorDataset(X, y), batch_size=128)

    # use a larger model
    base_model = DualOutputModule(in_features=64, hidden=512, num_classes=10)
    conf = LaplaceModelSamplingConfidence(base_model, index=0, enable_fit_cache=True)

    # first fit (should perform full fit)
    t0 = time.time()
    conf.fit(train_loader)
    t1 = time.time() - t0
    print(f"First fit time: {t1:.3f}s")
    time.sleep(0.1)  # wait a bit to ensure cache sweeper can run if needed

    # second fit (should hit cache and be fast)
    t0 = time.time()
    conf.fit(train_loader)
    t2 = time.time() - t0
    print(f"Second fit (cached) time: {t2:.6f}s")

    conf.confidence = lambda out, y=None: out[0][0].softmax(-1).max(-1).values

    # test glm no link
    conf.set_pred_type(pred_type="glm", link_approx="none")
    x_test = torch.randn(5, 64)
    h,(mu_logits, var)= conf.forward(x_test)
    print("GLM no link → class mean:", mu_logits.shape,)
    print("GLM no link → class var: ", var.shape)

    # test Monte Carlo on dual output
    conf.set_pred_type(pred_type="nn", link_approx="mc")  # Always set link_approx explicitly
    conf.set_backprop(False)
    conf.samples = 10
    conf.average = True
    conf.softmax = True
    conf.confidence = lambda out, y=None: out[0].softmax(-1).max(-1).values
    conf.index = None

    conf_vals, outputs = conf.forward(x_test)

    print("MC sampling → confidence:", conf_vals.shape)
