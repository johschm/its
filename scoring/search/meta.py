
import torch
from typing import List, Optional, Tuple, Any, Dict
from utils.transformation_problem import TransformationProblem

import torch
from typing import List, Any, Optional, Tuple, Dict

class AdaptiveBatchMetaOptimizer:
    """
    Streams a large dataset through internal optimizers using a limited working batch.
    Tracks per-sample cycles and stops when confidence threshold or max_cycles is reached.
    """

    def __init__(
        self,
        internal_optimizers: List[Any],
        internal_batch_size: int = 128,
        confidence_threshold: float = 0.9,
        max_cycles: int = 1000,          # per-sample max optimizer calls
        device: Optional[torch.device] = None,
        retain_history: bool = False,
        global_call_cap: Optional[int] = None  # optional safety cap over ALL calls
    ):
        self.internal_optimizers = internal_optimizers
        self.internal_batch_size = internal_batch_size
        if isinstance(confidence_threshold, (float, int)):
            self.confidence_threshold = torch.tensor([confidence_threshold])
        else:
            self.confidence_threshold = torch.tensor(confidence_threshold, dtype=torch.float32)
        self.max_cycles = max_cycles
        self.device = device
        self.retain_history = retain_history
        self.global_call_cap = global_call_cap

        # Internal cumulative history
        if retain_history:
            self._history: Dict[str, list] = {
                "n_samples": [],
                "percentage_max_calls": []
            }
        else:
            self._history = None

    @staticmethod
    def _compute_confidence(error: torch.Tensor) -> torch.Tensor:
        return -error

    def _refill_active(self, active: torch.Tensor, queue: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        need = self.internal_batch_size - active.numel()
        if need > 0 and queue.numel() > 0:
            take = queue[:need]
            queue = queue[need:]
            if active.numel() == 0:
                return take, queue
            active = torch.cat([active, take], dim=0)
        return active, queue

    def optimize(
        self,
        problem: Any,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        device = self.device or x.device
        x = x.to(device)
        if y is not None:
            y = y.to(device)

        N = x.shape[0]
        best_param: Optional[torch.Tensor] = None
        best_error = torch.full((N,), float("inf"), device=device)
        best_other: Optional[torch.Tensor] = None  # stores logits / "other" info

        indices = torch.arange(N, device=device)
        active = indices[: self.internal_batch_size]
        queue = indices[self.internal_batch_size:]

        # Cycles per sample, same size as total data
        cycles = torch.zeros(N, dtype=torch.int32, device=device)

        dim_placeholder = None
        outer_pass = 0

        threshold_tensor = self.confidence_threshold.to(device)

        while active.numel() > 0:
            outer_pass += 1
            if verbose:
                print(f"[Pass {outer_pass}] Active={active.numel()} Queue={queue.numel()} "
                      f"Remaining={active.numel()+queue.numel()}")
                print(f"Active indices: {active.tolist()}")

            for opt in self.internal_optimizers:
                if active.numel() == 0:
                    break
                if self.global_call_cap is not None and cycles.sum().item() >= self.global_call_cap:
                    if verbose:
                        print("Global call cap reached.")
                    break

                # Drop over-budget samples
                over_budget_mask = cycles[active] >= self.max_cycles
                if over_budget_mask.any():
                    active = active[~over_budget_mask]
                    active, queue = self._refill_active(active, queue)
                    if active.numel() == 0:
                        break

                x_act = x[active]
                y_act = y[active] if y is not None else None

                seed_params = None
                if best_param is not None:
                    seed_params = best_param[active].clone()
                    invalid = best_error[active].isinf()
                    if invalid.any():
                        seed_params[invalid] = 0.0

                opt_kwargs = {}
                if seed_params is not None and "initial_params" in opt.optimize.__code__.co_varnames:
                    opt_kwargs["initial_params"] = seed_params

                # Pass y if available
                if y_act is not None:
                    result = opt.optimize(problem, x_act, y=y_act, verbose=False, **opt_kwargs)
                else:
                    result = opt.optimize(problem, x_act, verbose=False, **opt_kwargs)

                if not (isinstance(result, (tuple, list)) and len(result) == 3):
                    raise RuntimeError("Internal optimizer must return (param, error, other).")

                p_new, e_new, o_new = result

                if dim_placeholder is None:
                    dim_placeholder = p_new.shape[-1]
                    best_param = torch.zeros((N, dim_placeholder), device=device, dtype=p_new.dtype)
                if (o_new is not None) and (best_other is None):
                    best_other = torch.zeros((N, o_new.shape[-1]), device=device, dtype=o_new.dtype)

                improved = e_new < best_error[active]
                if improved.any():
                    best_param[active[improved]] = p_new[improved]
                    best_error[active[improved]] = e_new[improved]
                    if best_other is not None and o_new is not None:
                        best_other[active[improved]] = o_new[improved]

                # Increment cycles for active batch
                cycles[active] += 1

                conf_active = self._compute_confidence(best_error[active])

                # Per-class threshold using predicted class from best_other
                if threshold_tensor.numel() == 1 or best_other is None:
                    done_conf_mask = conf_active >= threshold_tensor.item()
                else:
                    pred_class = best_other[active].argmax(dim=-1)
                    done_conf_mask = conf_active >= threshold_tensor[pred_class]

                done_budget_mask = cycles[active] >= self.max_cycles
                done_mask = done_conf_mask | done_budget_mask

                if done_mask.any():
                    keep = active[~done_mask]
                    active = keep
                    active, queue = self._refill_active(active, queue)

                if (active.numel() == 0 and queue.numel() == 0) or \
                   (self.global_call_cap is not None and cycles.sum().item() >= self.global_call_cap):
                    break

                # Force refill if all active exhausted budget
                if active.numel() > 0 and (cycles[active] >= self.max_cycles).all():
                    empty = active.new_empty(0)
                    active, queue = self._refill_active(empty, queue)

        # --- Update cumulative history once per optimize() call ---
        if self.retain_history:
            n_samples = N
            percentage_used = (cycles.float() / self.max_cycles).mean().item() * 100
            self._history["n_samples"].append(n_samples)
            self._history["percentage_max_calls"].append(percentage_used)

        return best_param, best_error, best_other


    def get_and_reset_history(self) -> Optional[Tuple[Dict[str, list], float]]:
        """
        Returns the cumulative history and resets it for new runs.
        Returns a tuple:
          - hist_copy: the raw lists of n_samples and percentage_max_calls
          - weighted_avg: weighted average of percentage_max_calls over all samples
        """
        if not self.retain_history:
            return None

        hist_copy = {k: v.copy() for k, v in self._history.items()}

        if len(hist_copy["n_samples"]) > 0:
            weighted_avg = np.average(
                hist_copy["percentage_max_calls"],
                weights=hist_copy["n_samples"]
            )
        else:
            weighted_avg = 0.0

        # reset history
        self._history = {k: [] for k in self._history.keys()}

        return hist_copy, weighted_avg