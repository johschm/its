import numpy as np
import torch
import math


class ParallelGradientDescent:
    def __init__(self, optimizer_class=torch.optim.Adam, optimizer_params=None,
                 learning_rate=0.1, max_iterations=1000, parallel_runs=4,
                 lr_decay_rate=0.95, project_param: bool = True,reflect=True):
        """
        :param learning_rate: Learning rate for gradient descent.
        :param max_iterations: Maximum number of iterations for gradient descent.
        :param parallel_runs: Number of parallel runs to perform.
        :param lr_decay_rate: Rate at which learning rate decays per iteration.
        """
        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer_params = optimizer_params
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.parallel_runs = parallel_runs
        self.lr_decay_rate = lr_decay_rate
        self.project_param = project_param
        self.reflect = reflect


        #stuck detection parameters


        # Better ideas could be to keep track of errors and see if performance improvements stagnates or look for occillations or flat gradients
        #or maybe scaling the noise for noise parall descent based on how much better the global best is than the current point.
        # This would add add more nose to worse samples which likely are stuck.


    def optimize(self, transformation_problem, x, y=None, verbose=False):
        """
        Run gradient descent to optimize the transformation parameters.

        :param transformation_problem: An instance of TransformationProblem.
        :param x: Input image tensor.
        :param y: Optional targets (will be repeated across parallel runs if provided).
        :param verbose: Whether to print progress information.
        :return: Tuple (best_param, best_error, best_other_data)
        """
        batch_size = x.shape[0]
        total_batches = self.parallel_runs * batch_size
        current_param_pre = transformation_problem.initial_param(batch_size, self.parallel_runs)
        current_param = current_param_pre.reshape(total_batches, -1).requires_grad_(True)
        x_repeated = x.repeat_interleave(self.parallel_runs, dim=0)
        y_repeated = y.repeat_interleave(self.parallel_runs, dim=0) if y is not None else None
        
        # Extract max_batch_size for chunked computation
        max_chunk = transformation_problem.max_batch_size if transformation_problem.max_batch_size is not None else total_batches
        
        # initialize stuck-run detection buffers
        self._init_stuck_detection(total_batches, current_param.shape[-1], device=x.device)

        optimizer = self.optimizer_class([current_param], lr=self.learning_rate, **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay_rate)

        best_param, best_error, best_other = None, None, None
        with torch.enable_grad():
            self.optimize_start()
            for iteration in range(self.max_iterations):
                optimizer.zero_grad(set_to_none=True)
                
                p_old = current_param.detach().clone()
                
                # Accumulate gradients per chunk to avoid keeping the full graph in memory
                total = current_param.shape[0]
                all_errors = []
                all_others = []
                
                for start in range(0, total, max_chunk):
                    end = min(start + max_chunk, total)
                    
                    # Compute error for the chunk; do NOT collect all chunks before backward
                    err_chunk, cls_chunk = transformation_problem._calculate_error(
                        x_repeated[start:end], 
                        current_param[start:end],
                        y=y_repeated[start:end] if y_repeated is not None else None
                    )
                    
                    # Backprop immediately to free the graph of this chunk
                    err_chunk.mean().backward()
                    
                    # Detach for tracking/consolidation
                    all_errors.append(err_chunk.detach())
                    all_others.append(cls_chunk.detach())
                
                # Concatenate all chunks
                error = torch.cat(all_errors, dim=0)
                other = torch.cat(all_others, dim=0)
                e_old = error.clone()

                # Now step once with the accumulated gradients
                self._custom_pre_step(optimizer, current_param)
                
                del all_errors, all_others
                
                #check that grad is not None
                if current_param.grad is None:
                    raise ValueError("Gradient is None. Check the transformation problem and ensure gradients are being computed correctly.")
                # For gradient modifications
                self._custom_update(optimizer, current_param, x_repeated, transformation_problem)
                self._custom_post_step(current_param, iteration)

                with torch.no_grad():
                    # Project (optional) then normalize if not projecting
                    if self.project_param:
                        current_param.data = transformation_problem.correct_param(current_param,reflect=self.reflect).data
                    else:
                        current_param.data = transformation_problem.normalize(current_param).data

                    if iteration == 0:
                        best_param = p_old.clone()
                        best_error = error.detach().clone()
                        best_other = other.detach().clone()
                    else:
                        improved = error.detach() < best_error
                        best_param[improved] = p_old[improved]
                        best_error[improved] = error.detach()[improved]
                        best_other[improved] = other.detach()[improved]

                    if verbose and (iteration % 10 == 0 or iteration == self.max_iterations-1):
                        lr = scheduler.get_last_lr()[0]
                        print(f"Iter {iteration+1}/{self.max_iterations}, LR:{lr:.6f}, Err:{e_old.mean():.4f}, Best:{best_error.mean():.4f}")

                    # update stuck-run detection history
                    self._update_stuck_history(iteration, error, current_param)

                    # detect and reinitialize any stuck runs
                    self._detect_and_reinit_stuck(
                        iteration,
                        current_param,
                        error.detach().clone(),
                        transformation_problem,
                        verbose
                    )

        # final evaluation on last params (with y)
        with torch.no_grad():
            final_error, final_other = transformation_problem.calculate_error(x_repeated, current_param, y=y_repeated)
            improved_final = final_error < best_error
            best_param[improved_final] = current_param[improved_final]
            best_error[improved_final] = final_error[improved_final]
            best_other[improved_final] = final_other[improved_final]

        # reshape and consolidate
        return self._reshape_results(x, best_param, best_error, best_other, transformation_problem)


    #TODO stuck detection
    def _init_stuck_detection(self, total_pts, param_dim, device):
        pass

    def _update_stuck_history(self, iteration, error, params):
        pass

    def _detect_and_reinit_stuck(self, iteration, current_param,prev_error, transformation_problem, verbose):
        pass

    def _reshape_results(self, x, params, error, other, problem):
        params = params.view(x.shape[0], self.parallel_runs, -1)
        error = error.view(x.shape[0], self.parallel_runs)
        other = other.view(x.shape[0], self.parallel_runs, -1)
        return problem.consolidate(x, params, error, other)

    def _custom_update(self, optimizer, params, x_repeated, transformation_problem):
        optimizer.step()

    def _custom_pre_step(self, optimizer, params):
        pass

    def _custom_post_step(self, params, iteration):
        pass

    def optimize_start(self):
        pass


class ParallelRestartingDescent(ParallelGradientDescent):
    def __init__(self,
                 reinit_interval=10, reinit_amount=0.0,**kwargs):
        super().__init__(**kwargs)
        self.reinit_interval = reinit_interval
        self.reinit_amount = reinit_amount


    def optimize_start(self):
        """For any setup before optimization starts"""
        pass

    def _detect_and_reinit_stuck(self, iteration, current_param,prev_error, transformation_problem, verbose):
        with torch.no_grad():
            if (iteration + 1) % self.reinit_interval == 0:
                if isinstance(self.reinit_amount, float):
                    amount = math.floor(self.parallel_runs * self.reinit_amount)
                else:
                    amount = self.reinit_amount
                if amount > 0:
                    total = current_param.shape[0]
                    batch_size = total // self.parallel_runs
                    current_param_reshaped = current_param.view(batch_size, self.parallel_runs, -1)
                    error_reshaped = prev_error.view(batch_size, self.parallel_runs)
                    worst_indices = error_reshaped.argsort(dim=1)[:, -amount:]
                    new_params = transformation_problem.initial_param(batch_size, amount).to(current_param.device)
                    new_error = torch.full((batch_size, amount), float("1e30"), device=prev_error.device)
                    batch_indices = torch.arange(batch_size, device=current_param.device).unsqueeze(1)
                    current_param_reshaped[batch_indices, worst_indices] = new_params
                    error_reshaped[batch_indices, worst_indices] = new_error
                    if verbose:
                        print(f"Iteration {iteration + 1}: Reinitialized {amount} worst runs per sample.")
                    current_param.data = current_param_reshaped.view(total, -1)

        if self.project_param:
            current_param.data = transformation_problem.correct_param(current_param).data


class WindowStuckDetectionDescent(ParallelGradientDescent):
    """
    Subclass that detects and reinitializes runs that have stagnated, flattened, or oscillated.
    """
    def __init__(self,
                 stuck_window_size=5,
                 improvement_threshold=1e-3,
                 flat_threshold=1e-4,
                 oscillation_threshold=0.6,
                 **kwargs):
        super().__init__(**kwargs)
        self.stuck_window_size = stuck_window_size
        self.improvement_threshold = improvement_threshold
        self.flat_threshold = flat_threshold
        self.oscillation_threshold = oscillation_threshold
        self.error_window = None
        self.window_ptr = 0

    def _init_stuck_detection(self, total_pts, param_dim, device):
        # Circular buffer for recent errors: [window, total_pts]
        self.error_window = torch.full(
            (self.stuck_window_size, total_pts),
            float('inf'),
            device=device
        )
        self.window_ptr = 0

    def _update_stuck_history(self, iteration, error, params):
        idx = self.window_ptr % self.stuck_window_size
        self.error_window[idx] = error.detach()
        self.window_ptr += 1
    #todo check what happens with reinit runs
    def _detect_and_reinit_stuck(self,
                                 iteration,
                                 current_param,
                                 prev_error,
                                 transformation_problem,
                                 verbose):
        # begin detection after filling buffer
        if self.window_ptr < self.stuck_window_size:
            return

        with torch.no_grad():
            total = current_param.shape[0]
            batch_size = total // self.parallel_runs
            # Align buffer so idx 0 is oldest
            errors = torch.roll(
                self.error_window,
                -self.window_ptr % self.stuck_window_size,
                dims=0
            )  # [window, total]


            # Metrics
            #maybe add some smoothing here? add smooth options to caclulate the average error over half the window?
            start_error= errors[0]
            end_error = errors[-1]
            rel_improve = (start_error - end_error) / (start_error + 1e-8)

            #check for flat regions if all errors are similar
            err_range = errors.max(dim=0).values - errors.min(dim=0).values
            flat_mask = err_range < self.flat_threshold

            #check wether the error is oscillating(difference between consecutive errors changes sign)
            diffs = errors[1:] - errors[:-1]
            signs = diffs.sign()
            flips = (signs[1:] * signs[:-1] < 0).sum(dim=0).float()#change in sign means sign times prev sign is negative

            osc_ratio = flips / (self.stuck_window_size - 2)
            osc_mask = osc_ratio > self.oscillation_threshold

            stuck = (rel_improve < self.improvement_threshold) | flat_mask | osc_mask
            if not stuck.any():
                return

            stuck_indices = stuck.nonzero(as_tuple=True)[0]
            batch_idx_stuck = stuck_indices // self.parallel_runs
            run_idx_stuck = stuck_indices % self.parallel_runs

            current_param_reshaped = current_param.view(batch_size, self.parallel_runs, -1)


            # Generate new params for each stuck run
            total_stuck = stuck_indices.size(0)
            new_param = transformation_problem.initial_param(total_stuck, 1)
            new_param = new_param.squeeze(1).to(current_param.device)  # [total_stuck, param_dim]

            # TODO make parallel
            for i, (b, r) in enumerate(zip(batch_idx_stuck.tolist(), run_idx_stuck.tolist())):
                current_param_reshaped[b, r] = new_param[i]

            # Reset history slots for these runs
            self.error_window[:, stuck_indices] = float('inf')

            # Update param tensor
            current_param.data = current_param_reshaped.view(total, -1)
            if self.project_param:
                current_param.data = transformation_problem.correct_param(current_param).data

            if verbose:
                print(f"Iter {iteration+1}: Reinitialized {total_stuck} stuck runs \
                      (Δ<{self.improvement_threshold}, flat<{self.flat_threshold}, osc>{self.oscillation_threshold})")


#TODO not good enough remove
class PointNoiseParallelDescent(ParallelGradientDescent):
    def __init__(self, noise_scale=0.1, noise_decay=0.95, **kwargs):
        super().__init__(**kwargs)
        self.noise_scale_max = noise_scale
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay


    def optimize_start(self):
        """For any setup before optimization starts"""
        self.noise_scale = self.noise_scale_max


    def _custom_pre_step(self, optimizer, params):
        with torch.no_grad():
            params.grad += torch.randn_like(params.grad) * self.noise_scale
            self.noise_scale *= self.noise_decay

class PointNoiseRestartingDescent(ParallelRestartingDescent, PointNoiseParallelDescent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



#TODO not good enough remove
class GradientNoiseParallelDescent(ParallelGradientDescent):
    def __init__(self, initial_temp=1.0, temp_decay=0.95, **kwargs):
        super().__init__(**kwargs)
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.temp_decay = temp_decay

    def optimize_start(self):
        """For any setup before optimization starts"""
        self.current_temp = self.initial_temp

    def _custom_post_step(self, params, iteration):
        with torch.no_grad():
            # Add temperature-scaled noise to parameters after update
            params += torch.randn_like(params) * self.current_temp
            self.current_temp *= self.temp_decay

class GradientNoiseRestartingDescent(ParallelRestartingDescent, GradientNoiseParallelDescent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
