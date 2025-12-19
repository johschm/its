# --------------------
# Triton kernels for sparse distance computation
# --------------------
import torch
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available. Falling back to PyTorch implementation.")



if TRITON_AVAILABLE:
    @triton.jit
    def masked_cdist_triton_kernel(
        x_ptr, y_ptr, idx_x_ptr, idx_y_ptr, out_ptr,
        N_pairs, D,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        if pid >= N_pairs:
            return

        i = tl.load(idx_x_ptr + pid)
        j = tl.load(idx_y_ptr + pid)

        # accumulator per-thread (split D in BLOCK_SIZE chunks)
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        offs = tl.arange(0, BLOCK_SIZE)
        for d in range(0, D, BLOCK_SIZE):
            mask_d = d + offs < D
            xi = tl.load(x_ptr + i * D + d + offs, mask=mask_d, other=0.0)
            yj = tl.load(y_ptr + j * D + d + offs, mask=mask_d, other=0.0)
            diff = xi - yj
            acc += diff * diff

        dist = tl.sqrt(tl.sum(acc))
        tl.store(out_ptr + pid, dist)

    @triton.jit
    def masked_cosine_triton_kernel(
        x_ptr, y_ptr, idx_x_ptr, idx_y_ptr, out_ptr,
        N_pairs, D,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        if pid >= N_pairs:
            return

        i = tl.load(idx_x_ptr + pid)
        j = tl.load(idx_y_ptr + pid)

        offs = tl.arange(0, BLOCK_SIZE)

        # accumulators for dot product and norms
        dot_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        x_norm_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        y_norm_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for d in range(0, D, BLOCK_SIZE):
            mask_d = d + offs < D
            xi = tl.load(x_ptr + i * D + d + offs, mask=mask_d, other=0.0)
            yj = tl.load(y_ptr + j * D + d + offs, mask=mask_d, other=0.0)
            dot_acc += xi * yj
            x_norm_acc += xi * xi
            y_norm_acc += yj * yj

        dot = tl.sum(dot_acc)
        x_norm = tl.sqrt(tl.sum(x_norm_acc))
        y_norm = tl.sqrt(tl.sum(y_norm_acc))

        cos_sim = dot / (x_norm * y_norm + 1e-8)
        tl.store(out_ptr + pid, cos_sim)



class MaskedCDistTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_query, x_train, y_idx_query, y_idx_train, k):
        mask = y_idx_query.unsqueeze(1) == y_idx_train.unsqueeze(0)
        idx_x, idx_y = mask.nonzero(as_tuple=True)
        num_pairs = idx_x.numel()

        if num_pairs == 0:
            dists = torch.full(mask.shape, float("inf"), device=x_query.device, dtype=x_query.dtype)
        else:
            idx_x_i32 = idx_x.to(dtype=torch.int32).contiguous()
            idx_y_i32 = idx_y.to(dtype=torch.int32).contiguous()
            out = torch.empty(num_pairs, device=x_query.device, dtype=x_query.dtype)
            masked_cdist_triton_kernel[(num_pairs,)](
                x_query, x_train, idx_x_i32, idx_y_i32, out,
                num_pairs, x_query.shape[1], BLOCK_SIZE=128
            )
            dists = torch.full(mask.shape, float("inf"), device=x_query.device, dtype=x_query.dtype)
            dists[idx_x, idx_y] = out

        k_safe = min(k, dists.shape[1])
        top_k_dists, top_k_indices = torch.topk(dists, k=k_safe, dim=1, largest=False)

        ctx.save_for_backward(x_query, x_train, top_k_indices, top_k_dists)
        ctx.k = k_safe
        return top_k_dists.mean(1)

    @staticmethod
    def backward(ctx, grad_output):
        x_query, x_train, top_k_indices, top_k_dists = ctx.saved_tensors
        k = ctx.k
        grad_x_query = torch.zeros_like(x_query)

        grad_output_expanded = grad_output.unsqueeze(1) / k

        top_k_neighbors = torch.gather(x_train.unsqueeze(0).expand(x_query.shape[0], -1, -1), 1, top_k_indices.unsqueeze(-1).expand(-1, -1, x_train.shape[1]))

        diff = x_query.unsqueeze(1) - top_k_neighbors

        # Avoid division by zero for zero distances
        inv_dist = 1.0 / (top_k_dists + 1e-8)

        grad = (diff * inv_dist.unsqueeze(-1)) * grad_output_expanded.unsqueeze(-1)

        grad_x_query = grad.sum(dim=1)

        return grad_x_query, None, None, None, None

class MaskedCosineTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_query, x_train, y_idx_query, y_idx_train, k):
        mask = y_idx_query.unsqueeze(1) == y_idx_train.unsqueeze(0)
        idx_x, idx_y = mask.nonzero(as_tuple=True)
        num_pairs = idx_x.numel()

        if num_pairs == 0:
            sim = torch.zeros(mask.shape, device=x_query.device, dtype=x_query.dtype)
        else:
            idx_x_i32 = idx_x.to(dtype=torch.int32).contiguous()
            idx_y_i32 = idx_y.to(dtype=torch.int32).contiguous()
            out = torch.empty(num_pairs, device=x_query.device, dtype=x_query.dtype)
            masked_cosine_triton_kernel[(num_pairs,)](
                x_query, x_train, idx_x_i32, idx_y_i32, out,
                num_pairs, x_query.shape[1], BLOCK_SIZE=128
            )
            sim = torch.zeros(mask.shape, device=x_query.device, dtype=x_query.dtype)
            sim[idx_x, idx_y] = out

        dists = 1.0 - sim
        dists[~mask] = float("inf")

        k_safe = min(k, dists.shape[1])
        top_k_dists, top_k_indices = torch.topk(dists, k=k_safe, dim=1, largest=False)

        ctx.save_for_backward(x_query, x_train, top_k_indices)
        ctx.k = k_safe
        return top_k_dists.mean(1)

    @staticmethod
    def backward(ctx, grad_output):
        x_query, x_train, top_k_indices = ctx.saved_tensors
        k = ctx.k
        grad_x_query = torch.zeros_like(x_query)

        grad_output_expanded = grad_output.unsqueeze(1) / k

        top_k_neighbors = torch.gather(x_train.unsqueeze(0).expand(x_query.shape[0], -1, -1), 1, top_k_indices.unsqueeze(-1).expand(-1, -1, x_train.shape[1]))

        x_norm = torch.linalg.norm(x_query, dim=1, keepdim=True)
        y_norm = torch.linalg.norm(top_k_neighbors, dim=2, keepdim=True)

        dot_product = (x_query.unsqueeze(1) * top_k_neighbors).sum(-1, keepdim=True)

        # Grad of cosine sim w.r.t x_query
        grad_sim = (top_k_neighbors / (x_norm.unsqueeze(1) * y_norm + 1e-8)) - \
                   (dot_product * x_query.unsqueeze(1) / (x_norm.pow(3).unsqueeze(1) * y_norm + 1e-8))

        # Grad of cosine dist is -grad_sim
        grad = -grad_sim * grad_output_expanded.unsqueeze(-1)

        grad_x_query = grad.sum(dim=1)

        return grad_x_query, None, None, None, None


import torch
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available. Falling back to PyTorch implementation.")


if TRITON_AVAILABLE:
    @triton.jit
    def masked_mixed_triton_kernel(
        x_ptr, y_ptr, idx_x_ptr, idx_y_ptr,
        out_euclid_ptr, out_cos_ptr,
        N_pairs, D, eps,
        BLOCK_SIZE: tl.constexpr, SQUARED: tl.constexpr
    ):
        """
        For each program id (pair) compute:
          dot = x · y
          x_norm_sq = ||x||^2
          y_norm_sq = ||y||^2
        Then:
          euclid_sq = x_norm_sq + y_norm_sq - 2*dot
          euclid = euclid_sq (if SQUARED) else sqrt(euclid_sq + eps)
          cos_sim = dot / (sqrt(x_norm_sq) * sqrt(y_norm_sq) + eps)
          cos_dist = 1 - clamp(cos_sim, -1, 1)
        Store euclid (or euclid_sq) to out_euclid_ptr[pid]
        Store cos_dist to out_cos_ptr[pid]
        """
        pid = tl.program_id(0)
        if pid >= N_pairs:
            return

        i = tl.load(idx_x_ptr + pid)
        j = tl.load(idx_y_ptr + pid)

        offs = tl.arange(0, BLOCK_SIZE)
        dot_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        x_norm_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        y_norm_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # accumulate in BLOCK_SIZE chunks
        for d in range(0, D, BLOCK_SIZE):
            mask_d = d + offs < D
            xi = tl.load(x_ptr + i * D + d + offs, mask=mask_d, other=0.0)
            yj = tl.load(y_ptr + j * D + d + offs, mask=mask_d, other=0.0)
            dot_acc += xi * yj
            x_norm_acc += xi * xi
            y_norm_acc += yj * yj

        dot = tl.sum(dot_acc)
        x_norm_sq = tl.sum(x_norm_acc)
        y_norm_sq = tl.sum(y_norm_acc)

        # Euclidean
        euclid_sq = x_norm_sq + y_norm_sq - 2.0 * dot
        # guard numerical small negatives
        euclid_sq = tl.max(euclid_sq, 0.0)
        if SQUARED:
            euclid_val = euclid_sq
        else:
            euclid_val = tl.sqrt(euclid_sq + eps)

        # Cosine
        x_norm = tl.sqrt(x_norm_sq + eps)
        y_norm = tl.sqrt(y_norm_sq + eps)
        cos_sim = dot / (x_norm * y_norm + eps)
        # numeric safety clamp
        cos_sim = tl.max(tl.min(cos_sim, 1.0), -1.0)
        cos_dist = 1.0 - cos_sim

        tl.store(out_euclid_ptr + pid, euclid_val)
        tl.store(out_cos_ptr + pid, cos_dist)


class MaskedMixedDistanceTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_query, x_train, y_idx_query, y_idx_train,
                alpha: float = 0.5, eps: float = 1e-8,
                squared: bool = False, normalize_euclid: bool = True, k: int = 1):
        """
        x_query: (N, D)
        x_train: (M, D)
        y_idx_query: (N,) labels/ids for query
        y_idx_train: (M,) labels/ids for train
        Returns: per-query mean of top-k mixed distances (shape (N,))
        """
        assert x_query.dim() == 2 and x_train.dim() == 2
        N, D = x_query.shape
        M = x_train.shape[0]

        # build mask of allowed pairs = same label
        mask = y_idx_query.unsqueeze(1) == y_idx_train.unsqueeze(0)
        idx_x, idx_y = mask.nonzero(as_tuple=True)
        num_pairs = idx_x.numel()

        device = x_query.device
        dtype = x_query.dtype

        if num_pairs == 0:
            # nothing to compute: return inf distances so topk behaves
            dists = torch.full((N, M), float("inf"), device=device, dtype=dtype)
            euclid_mat = torch.zeros((N, M), device=device, dtype=dtype)
            cos_mat = torch.zeros((N, M), device=device, dtype=dtype)
        else:
            idx_x_i32 = idx_x.to(dtype=torch.int32).contiguous()
            idx_y_i32 = idx_y.to(dtype=torch.int32).contiguous()

            out_euclid = torch.empty(num_pairs, device=device, dtype=torch.float32)
            out_cos = torch.empty(num_pairs, device=device, dtype=torch.float32)

            # choose a good block size (tunable)
            BLOCK = 128
            masked_mixed_triton_kernel[(num_pairs,)](
                x_query, x_train, idx_x_i32, idx_y_i32,
                out_euclid, out_cos,
                num_pairs, D, float(eps),
                BLOCK_SIZE=BLOCK, SQUARED=bool(squared)
            )

            # Put results back into full (N, M) matrices
            dists = torch.full((N, M), float("inf"), device=device, dtype=dtype)
            euclid_mat = torch.zeros((N, M), device=device, dtype=dtype)
            cos_mat = torch.zeros((N, M), device=device, dtype=dtype)

            euclid_mat[idx_x, idx_y] = out_euclid.to(dtype)
            cos_mat[idx_x, idx_y] = out_cos.to(dtype)
            # dists will be built after optional normalization

        # normalize euclid if requested (done on host)
        if normalize_euclid:
            # compute mean only over the allowed pairs (euclid_mat>0 or mask)
            if mask.any():
                # Only consider finite euclid entries (mask true)
                mean_euclid = euclid_mat[mask].mean()
                mean_euclid = mean_euclid.clamp(min=eps)
            else:
                mean_euclid = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            mean_euclid = torch.tensor(1.0, device=device, dtype=dtype)

        euclid_mat = euclid_mat / (mean_euclid + eps)

        # Combine
        mixed = alpha * euclid_mat + (1.0 - alpha) * cos_mat

        # distances not allowed remain inf already (we only filled mask entries)
        dists = mixed
        dists[~mask] = float("inf")

        # safe k
        k_safe = min(k, dists.shape[1])
        top_k_dists, top_k_indices = torch.topk(dists, k=k_safe, dim=1, largest=False)

        # gather euclid and cos values of topk for backward
        top_k_euclid = torch.gather(euclid_mat, 1, top_k_indices)
        top_k_cos = torch.gather(cos_mat, 1, top_k_indices)

        # Save tensors for backward
        ctx.save_for_backward(x_query, x_train, top_k_indices, top_k_euclid, top_k_cos)
        ctx.k = k_safe
        ctx.alpha = float(alpha)
        ctx.squared = bool(squared)
        ctx.eps = float(eps)
        ctx.mean_euclid = mean_euclid  # scalar tensor

        # As in your previous functions, return mean over top-k per query
        return top_k_dists.mean(dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        x_query, x_train, top_k_indices, top_k_euclid, top_k_cos = ctx.saved_tensors
        k = ctx.k
        alpha = ctx.alpha
        squared = ctx.squared
        eps = ctx.eps
        mean_euclid = ctx.mean_euclid

        N, D = x_query.shape
        grad_x_query = torch.zeros_like(x_query)

        if k == 0:
            return grad_x_query, None, None, None, None, None, None, None

        # per-query scalar multiplier (grad of mean over k elements)
        grad_output_expanded = (grad_output.unsqueeze(1) / k).unsqueeze(-1)  # (N,1,1) after unsqueeze

        # gather neighbors: (N, k, D)
        expanded_x_train = x_train.unsqueeze(0).expand(N, -1, -1)
        top_k_neighbors = torch.gather(
            expanded_x_train,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, x_train.shape[1])
        )

        # EUCLIDEAN gradient:
        # If squared: d(euclid_sq)/dx = 2 * (x - y)
        # If not squared: d(euclid)/dx = (x - y) / (euclid + eps)
        if squared:
            grad_euclid = 2.0 * (x_query.unsqueeze(1) - top_k_neighbors)  # (N,k,D)
        else:
            # avoid divide-by-zero
            denom = (top_k_euclid.unsqueeze(-1) + eps)
            grad_euclid = (x_query.unsqueeze(1) - top_k_neighbors) / denom

        # account for normalization factor (euclid was divided by mean_euclid)
        grad_euclid = grad_euclid / (mean_euclid + eps)

        # COSINE gradient:
        # follow gradient from earlier code: grad_sim = y/(||x||*||y||) - ( (x·y) * x / (||x||^3 * ||y||) )
        x_norm = torch.linalg.norm(x_query, dim=1, keepdim=True)  # (N,1)
        # y_norm for each neighbor (N,k,1)
        y_norm = torch.linalg.norm(top_k_neighbors, dim=2, keepdim=True)  # (N,k,1)

        dot_product = (x_query.unsqueeze(1) * top_k_neighbors).sum(dim=2, keepdim=True)  # (N,k,1)

        # ensure stable denominators
        x_norm_safe = x_norm.unsqueeze(1) + eps  # (N,1,1) -> broadcasting to (N,k,1)
        y_norm_safe = y_norm + eps
        x_norm_cubed = (x_norm_safe ** 3)

        grad_sim = (top_k_neighbors / (x_norm_safe * y_norm_safe)) - \
                   (dot_product * x_query.unsqueeze(1) / (x_norm_cubed * y_norm_safe))

        # grad of cos_dist = -grad_sim
        grad_cos_dist = -grad_sim  # (N,k,1)

        # Combine grads: grad_mixed = alpha * grad_euclid + (1-alpha) * grad_cos_dist
        alpha_t = alpha
        mixed_grad = alpha_t * grad_euclid + (1.0 - alpha_t) * grad_cos_dist  # (N,k,D)

        # Multiply by upstream grad scalar and sum over k neighbors
        grad = mixed_grad * grad_output_expanded  # (N,k,D)
        grad_x_query = grad.sum(dim=1)  # (N,D)

        # Return grads in same order as forward inputs:
        # (x_query, x_train, y_idx_query, y_idx_train, alpha, eps, squared, normalize_euclid, k)
        # Only grad for x_query is needed/meaningful. Others are None.
        return grad_x_query, None, None, None, None, None, None, None