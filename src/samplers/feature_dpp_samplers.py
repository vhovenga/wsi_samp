import torch
from typing import List, Dict, Any, Optional, Tuple, Literal
from .base_sampler import NonLearnableSampler
import time 

# @torch.no_grad()
# def sample_low_rank_dpp_batched(
#     Phi: torch.Tensor,                      # [B, N, D]
#     return_log_likelihood: bool = False,
#     eps: float = 1e-10
# ) -> Tuple[List[List[int]], Optional[torch.Tensor]]:
#     """
#     Batched sampling from L-ensemble DPPs with low-rank L = Phi @ Phi.T.
#     Prints elapsed time for each major stage and detailed timings within sampling loop.
#     """
#     assert Phi.dim() == 3, "Phi must be [B, N, D]"
#     B, N, D = Phi.shape
#     device = Phi.device
#     dtype = Phi.dtype

#     times = {}
#     t0 = time.perf_counter()

#     # --- Step 1: dual kernel and eigendecomposition ---
#     C = torch.matmul(Phi.transpose(1, 2), Phi)
#     torch.cuda.synchronize(device) if device.type == "cuda" else None
#     times["matmul"] = time.perf_counter() - t0

#     t1 = time.perf_counter()
#     evals, evecs = torch.linalg.eigh(C)
#     torch.cuda.synchronize(device) if device.type == "cuda" else None
#     times["eigh"] = time.perf_counter() - t1
#     evals = torch.clamp(evals, min=0.0)

#     # --- Step 2: Bernoulli selection ---
#     t2 = time.perf_counter()
#     probs = evals / (evals + 1.0)
#     keep = torch.bernoulli(torch.clamp(probs, 0.0, 1.0)).bool()
#     torch.cuda.synchronize(device) if device.type == "cuda" else None
#     times["bernoulli"] = time.perf_counter() - t2

#     # --- Step 3: normalization constant ---
#     log_norm = None
#     t3 = time.perf_counter()
#     if return_log_likelihood:
#         I = torch.eye(D, dtype=dtype, device=device).expand(B, D, D)
#         chol = torch.linalg.cholesky(I + C)
#         log_norm = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), dim=1)
#         torch.cuda.synchronize(device) if device.type == "cuda" else None
#     times["log_norm"] = time.perf_counter() - t3

#     # --- Step 4: loop over batches for sampling ---
#     t4 = time.perf_counter()
#     all_selected = []
#     log_ll_list = []

#     # Sub-timings within loop_sampling
#     subtimes = {
#         "build_V": 0.0,
#         "row_sq": 0.0,
#         "multinomial": 0.0,
#         "householder": 0.0,
#         "qr_update": 0.0,
#         "loglike": 0.0,
#     }

#     for b in range(B):
#         kept = keep[b]
#         k = int(kept.sum().item())

#         if k == 0:
#             idx = torch.randint(0, N, (1,), device=device)
#             all_selected.append(idx)
#             if return_log_likelihood:
#                 log_ll_list.append(-log_norm[b])
#             continue

#         # Build V
#         t_bv = time.perf_counter()
#         U = evecs[b, :, kept]
#         lam = torch.clamp(evals[b, kept], min=eps)
#         V = Phi[b] @ (U @ torch.diag(lam.rsqrt()))
#         V, _ = torch.linalg.qr(V, mode="reduced")
#         subtimes["build_V"] += time.perf_counter() - t_bv

#         selected = []
#         print(V.shape[1])
#         while V.shape[1] > 0:
#             # Row norms
#             t_rs = time.perf_counter()
#             row_sq = torch.einsum('ij,ij->i', V, V)
#             total = row_sq.sum()
#             if total <= eps:
#                 break
#             subtimes["row_sq"] += time.perf_counter() - t_rs

#             # Sampling index
#             t_m = time.perf_counter()
#             i = torch.multinomial(row_sq / total, 1).item()
#             subtimes["multinomial"] += time.perf_counter() - t_m
#             selected.append(i)

#             vi = V[i, :]
#             nrm = torch.linalg.norm(vi)
#             if nrm <= eps:
#                 V[i, :] = 0.0
#                 t_qr = time.perf_counter()
#                 if torch.linalg.matrix_rank(V) < V.shape[1]:
#                     V, _ = torch.linalg.qr(V, mode="reduced")
#                 subtimes["qr_update"] += time.perf_counter() - t_qr
#                 continue

#             # Householder step
#             t_h = time.perf_counter()
#             f = (vi / nrm).reshape(-1, 1)
#             sign = 1.0 if f[0, 0] >= 0 else -1.0
#             u = f.clone()
#             u[0, 0] += sign
#             u_nrm = torch.linalg.norm(u)
#             if u_nrm <= eps:
#                 H = torch.eye(f.numel(), dtype=dtype, device=device) - 2.0 * (f @ f.t())
#             else:
#                 v = u / u_nrm
#                 H = torch.eye(f.numel(), dtype=dtype, device=device) - 2.0 * (v @ v.t())
#             subtimes["householder"] += time.perf_counter() - t_h

#             # Update V
#             t_qr = time.perf_counter()
#             V = V @ H
#             V = V[:, 1:]
#             if V.numel() > 0:
#                 V, _ = torch.linalg.qr(V, mode="reduced")
#             subtimes["qr_update"] += time.perf_counter() - t_qr

#         all_selected.append(torch.tensor(selected))

#         if return_log_likelihood:
#             t_ll = time.perf_counter()
#             if len(selected) == 0:
#                 ll = -log_norm[b]
#             else:
#                 Phi_Y = Phi[b, selected, :]
#                 LY = Phi_Y @ Phi_Y.t()
#                 LY = 0.5 * (LY + LY.t())
#                 cholY = torch.linalg.cholesky(LY + eps * torch.eye(LY.shape[0], device=device, dtype=dtype))
#                 log_det_LY = 2.0 * torch.sum(torch.log(torch.diagonal(cholY)))
#                 ll = log_det_LY - log_norm[b]
#             log_ll_list.append(ll)
#             subtimes["loglike"] += time.perf_counter() - t_ll

#     torch.cuda.synchronize(device) if device.type == "cuda" else None
#     times["loop_sampling"] = time.perf_counter() - t4

#     total = sum(times.values())
#     print("\n---- Profiling ----")
#     for k, v in times.items():
#         print(f"{k:<15}: {v:.4f} s")
#     print(f"Total             : {total:.4f} s")

#     print("\n---- Loop Breakdown ----")
#     for k, v in subtimes.items():
#         print(f"{k:<15}: {v:.4f} s")
#     print(f"Loop total (check): {sum(subtimes.values()):.4f} s\n")

#     log_ll_out = torch.stack(log_ll_list) if return_log_likelihood else None
#     return all_selected, log_ll_out


@torch.no_grad()
def sample_low_rank_dpp_batched(
    Phi: torch.Tensor,                      # [B, N, D]
    return_log_likelihood: bool = False,
    eps: float = 1e-10,
    qr_period: int = 8,
    rank_tol: float = 1e-8
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Incremental low-rank DPP sampling (GPU-optimized).
    - Uses QR-based rank trimming, periodic stabilization.
    - Fully GPU-compatible, no host syncs.
    - Returns: list of index tensors and optional log-likelihoods.
    """
    assert Phi.dim() == 3, "Phi must be [B, N, D]"
    B, N, D = Phi.shape
    device, dtype = Phi.device, Phi.dtype

    # Dual kernel and eigendecomposition
    C = Phi.transpose(1, 2) @ Phi                         # [B, D, D]
    evals, evecs = torch.linalg.eigh(C)                   # [B, D], [B, D, D]
    evals = torch.clamp(evals, min=0.0)

    # Bernoulli selection
    probs = evals / (evals + 1.0)
    keep = torch.bernoulli(probs.clamp_(0.0, 1.0)).bool() # [B, D]

    # Normalization constant (optional)
    log_norm = None
    if return_log_likelihood:
        I = torch.eye(D, dtype=dtype, device=device).expand(B, D, D)
        chol = torch.linalg.cholesky(I + C)
        log_norm = 2.0 * torch.sum(
            torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), dim=1
        )

    all_selected: List[torch.Tensor] = []
    log_ll_list = []

    # Main sampling loop
    for b in range(B):
        kept = keep[b]
        k = int(kept.sum().item())
        if k == 0:
            idx = torch.randint(0, N, (1,), device=device)
            all_selected.append(idx)
            if return_log_likelihood:
                log_ll_list.append(-log_norm[b])
            continue

        U = evecs[b, :, kept]                              # [D, k]
        lam = torch.clamp(evals[b, kept], min=eps)         # [k]
        V = Phi[b] @ (U * lam.rsqrt().unsqueeze(0))        # [N, k]
        V, R = torch.linalg.qr(V, mode="reduced")          # V orthonormal
        r = (R.abs().diagonal().gt(rank_tol)).sum()
        if int(r) < V.shape[1]:
            V = V[:, :int(r)]

        row_sq = torch.einsum('ij,ij->i', V, V)            # [N]
        selected_tensors = []
        it = 0
        k_cur = V.shape[1]

        while k_cur > 0:
            total = row_sq.sum()
            if total <= eps:
                break

            i_tensor = torch.multinomial(row_sq / total, 1, replacement=False)
            selected_tensors.append(i_tensor)

            vi = V[i_tensor, :].squeeze(0)                 # [k]
            nrm = torch.linalg.norm(vi)
            if nrm <= eps:
                V.index_fill_(0, i_tensor, 0.0)
                if (it % qr_period) == 0 and k_cur > 0:
                    V, R = torch.linalg.qr(V, mode="reduced")
                    r = (R.abs().diagonal().gt(rank_tol)).sum()
                    if int(r) < V.shape[1]:
                        V = V[:, :int(r)]
                        k_cur = V.shape[1]
                    row_sq = torch.einsum('ij,ij->i', V, V)
                it += 1
                continue

            v = vi / nrm
            proj = V @ v
            V -= proj.unsqueeze(1) * v.unsqueeze(0)
            row_sq -= proj * proj
            row_sq.clamp_(min=0.0)

            if V.shape[1] > 1:
                V = V[:, 1:].contiguous()
            else:
                V = V[:, :0]
            k_cur = V.shape[1]

            if (it % qr_period) == 0 and k_cur > 0:
                V, R = torch.linalg.qr(V, mode="reduced")
                r = (R.abs().diagonal().gt(rank_tol)).sum()
                if int(r) < V.shape[1]:
                    V = V[:, :int(r)]
                    k_cur = V.shape[1]
                row_sq = torch.einsum('ij,ij->i', V, V)
            it += 1

        sel = (
            torch.cat(selected_tensors)
            if selected_tensors
            else torch.empty(0, dtype=torch.long, device=device)
        )
        all_selected.append(sel)

        if return_log_likelihood:
            if sel.numel() == 0:
                ll = -log_norm[b]
            else:
                Phi_Y = Phi[b].index_select(0, sel)
                LY = Phi_Y @ Phi_Y.t()
                LY = 0.5 * (LY + LY.t())
                cholY = torch.linalg.cholesky(
                    LY + eps * torch.eye(LY.shape[0], device=device, dtype=dtype)
                )
                log_det_LY = 2.0 * torch.sum(torch.log(torch.diagonal(cholY)))
                ll = log_det_LY - log_norm[b]
            log_ll_list.append(ll)

    log_ll_out = torch.stack(log_ll_list) if return_log_likelihood else None
    return all_selected, log_ll_out

@torch.no_grad()
def sample_low_rank_k_dpp_batched(
    Phi: torch.Tensor,                   # [B, N, D]
    k: int,
    return_log_likelihood: bool = False,
    eps: float = 1e-10,
    qr_period: int = 8,
    rank_tol: float = 1e-8
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Optimized batched k-DPP sampler (GPU-native, low-rank).
    - Fully GPU-compatible; no CPU syncs or numpy ops.
    - Vectorized DP for eigenvector selection when k < D.
    - Fast path when k >= D: keep all eigenvectors.
    - Periodic QR stabilization for orthogonality.
    - Fixed-size (k) subsets per batch.
    """
    assert Phi.dim() == 3, "Phi must be [B, N, D]"
    B, N, D = Phi.shape
    device, dtype = Phi.device, Phi.dtype

    # Dual kernel and eigendecomposition
    C = Phi.transpose(1, 2) @ Phi                      # [B, D, D]
    evals, evecs = torch.linalg.eigh(C)                # [B, D], [B, D, D]
    evals = torch.clamp(evals, min=0.0)

    all_selected: List[torch.Tensor] = []
    log_ll_list: List[torch.Tensor] = []

    for b in range(B):
        lam = evals[b]                                  # [D]
        D_b = lam.shape[0]

        # --- Eigenvector selection ---
        if k >= D_b:
            # Keep all eigenvectors; no DP needed.
            keep = torch.ones(D_b, dtype=torch.bool, device=device)
            log_norm = lam.clamp_min(eps).log().sum() if return_log_likelihood else None
        else:
            lam_eps = lam + eps
            # DP table E[i, j] for elementary symmetric polynomials (GPU)
            E = torch.zeros((D_b + 1, k + 1), dtype=dtype, device=device)
            E[:, 0] = 1.0
            for i in range(1, D_b + 1):
                lam_i = lam_eps[i - 1]
                j_max = min(i, k)
                if j_max >= 1:
                    j = torch.arange(1, j_max + 1, device=device)
                    E[i, j] = E[i - 1, j] + lam_i * E[i - 1, j - 1]

            # Backward sampling of eigenvectors
            selected_eigs = []
            rem = k
            for i in reversed(range(D_b)):
                if rem == 0:
                    break
                denom = E[i + 1, rem] + eps
                p = lam_eps[i] * E[i, rem - 1] / denom
                if torch.rand(1, device=device) < p:
                    selected_eigs.append(i)
                    rem -= 1
            selected_eigs.reverse()

            keep = torch.zeros(D_b, dtype=torch.bool, device=device)
            if selected_eigs:
                keep[torch.tensor(selected_eigs, device=device)] = True

            log_norm = torch.log(E[D_b, k] + eps) if return_log_likelihood else None

        # --- Build projection basis V ---
        U = evecs[b, :, keep]                           # [D, k_sel], k_sel==k if rank full
        lam_k = lam[keep].clamp_min(eps)                # [k_sel]
        V = Phi[b] @ (U * lam_k.rsqrt().unsqueeze(0))   # [N, k_sel]
        if V.numel() == 0:
            all_selected.append(torch.empty(0, dtype=torch.long, device=device))
            if return_log_likelihood:
                log_ll_list.append(-log_norm)
            continue

        V, R = torch.linalg.qr(V, mode="reduced")
        r = (R.abs().diagonal().gt(rank_tol)).sum()
        if int(r) < V.shape[1]:
            V = V[:, :int(r)]

        # --- Projection DPP selection over items ---
        selected_tensors = []
        it = 0
        while V.shape[1] > 0:
            row_sq = torch.einsum("ij,ij->i", V, V)     # [N]
            total = row_sq.sum()
            if total <= eps:
                break

            i_tensor = torch.multinomial(row_sq / total, 1)
            selected_tensors.append(i_tensor)

            vi = V[i_tensor, :].squeeze(0)
            nrm = torch.linalg.norm(vi)
            if nrm <= eps:
                V.index_fill_(0, i_tensor, 0.0)
            else:
                v = vi / nrm
                proj = V @ v
                V -= proj.unsqueeze(1) * v.unsqueeze(0)
                # remove the selected direction
                if V.shape[1] > 1:
                    V = V[:, 1:].contiguous()
                else:
                    V = V[:, :0]

            if (it % qr_period) == 0 and V.shape[1] > 0:
                V, R = torch.linalg.qr(V, mode="reduced")
                r = (R.abs().diagonal().gt(rank_tol)).sum()
                if int(r) < V.shape[1]:
                    V = V[:, :int(r)]
            it += 1

        sel = torch.cat(selected_tensors) if selected_tensors else torch.empty(0, dtype=torch.long, device=device)
        all_selected.append(sel)

        # --- Optional log-likelihood ---
        if return_log_likelihood:
            if sel.numel() == 0:
                ll = -log_norm
            else:
                Phi_Y = Phi[b].index_select(0, sel)
                LY = Phi_Y @ Phi_Y.t()
                LY = 0.5 * (LY + LY.t())
                cholY = torch.linalg.cholesky(LY + eps * torch.eye(LY.shape[0], device=device, dtype=dtype))
                log_det_LY = 2.0 * torch.sum(torch.log(torch.diagonal(cholY)))
                ll = log_det_LY - log_norm
            log_ll_list.append(ll)

    log_ll_out = torch.stack(log_ll_list) if return_log_likelihood else None
    return all_selected, log_ll_out

# @torch.no_grad()
# def sample_low_rank_k_dpp_batched(
#     Phi: torch.Tensor,                   # [B, N, D]
#     k: int,
#     return_log_likelihood: bool = False,
#     eps: float = 1e-10
# ) -> Tuple[List[List[int]], Optional[torch.Tensor]]:
#     """
#     Batched sampling from k-DPPs with low-rank L = Phi @ Phi.T.
#     Each batch element is treated as an independent k-DPP.

#     Args:
#         Phi: [B, N, D] tensor of feature matrices
#         k: number of elements to sample
#         return_log_likelihood: if True, returns per-batch log-likelihoods
#         eps: numerical stability term

#     Returns:
#         all_selected: list of lists of sampled indices per batch
#         log_ll_out: [B] tensor if return_log_likelihood=True, else None
#     """
#     assert Phi.dim() == 3, "Phi must be [B, N, D]"
#     B, N, D = Phi.shape
#     device = Phi.device
#     dtype = Phi.dtype

#     # --- Step 1: dual kernel C = Phi^T Phi  -> [B, D, D]
#     C = torch.matmul(Phi.transpose(1, 2), Phi)

#     # --- Step 2: eigendecomposition (batched)
#     evals, evecs = torch.linalg.eigh(C)   # [B, D], [B, D, D]
#     evals = torch.clamp(evals, min=0.0)

#     all_selected = []
#     log_ll_list = []

#     for b in range(B):
#         lam = evals[b].cpu().numpy()
#         D_b = len(lam)

#         # --- DP table for elementary symmetric polynomials ---
#         E = torch.zeros((D_b + 1, k + 1), dtype=dtype, device=device)
#         E[:, 0] = 1.0
#         for i in range(1, D_b + 1):
#             lam_i = lam[i - 1]
#             for j in range(1, min(i, k) + 1):
#                 E[i, j] = E[i - 1, j] + lam_i * E[i - 1, j - 1]

#         # --- Sample eigenvectors to keep ---
#         selected_eigs = []
#         rem = k
#         for i in reversed(range(D_b)):
#             if rem == 0:
#                 break
#             prob = lam[i] * E[i, rem - 1] / (E[i + 1, rem] + eps)
#             if torch.rand(1).item() < prob:
#                 selected_eigs.append(i)
#                 rem -= 1
#         selected_eigs.reverse()

#         keep = torch.zeros(D_b, dtype=torch.bool, device=device)
#         keep[selected_eigs] = True

#         # --- log normalization term (log E[D,k]) ---
#         log_norm = None
#         if return_log_likelihood:
#             log_norm = torch.log(E[D_b, k] + eps)

#         # --- Projection DPP sampling ---
#         U = evecs[b, :, keep]                            # [D, k]
#         lam_k = torch.clamp(evals[b, keep], min=eps)     # [k]
#         V = Phi[b] @ (U @ torch.diag(lam_k.rsqrt()))     # [N, k]
#         V, _ = torch.linalg.qr(V, mode="reduced")

#         selected = []
#         while V.shape[1] > 0:
#             row_sq = torch.sum(V * V, dim=1)
#             total = row_sq.sum()
#             if total <= eps:
#                 break
#             i = torch.multinomial(row_sq / total, 1).item()
#             selected.append(i)

#             vi = V[i, :]
#             nrm = torch.linalg.norm(vi)
#             if nrm <= eps:
#                 V[i, :] = 0.0
#                 if torch.linalg.matrix_rank(V) < V.shape[1]:
#                     V, _ = torch.linalg.qr(V, mode="reduced")
#                 continue

#             f = (vi / nrm).reshape(-1, 1)
#             sign = 1.0 if f[0, 0] >= 0 else -1.0
#             u = f.clone()
#             u[0, 0] += sign
#             u_nrm = torch.linalg.norm(u)
#             if u_nrm <= eps:
#                 H = torch.eye(f.numel(), dtype=dtype, device=device) - 2.0 * (f @ f.t())
#             else:
#                 v = u / u_nrm
#                 H = torch.eye(f.numel(), dtype=dtype, device=device) - 2.0 * (v @ v.t())
#             V = V @ H
#             V = V[:, 1:]
#             if V.numel() > 0:
#                 V, _ = torch.linalg.qr(V, mode="reduced")

#         all_selected.append(torch.tensor(selected, device=device))

#         # --- Optional log-likelihood ---
#         if return_log_likelihood:
#             if len(selected) == 0:
#                 ll = -log_norm
#             else:
#                 Phi_Y = Phi[b, selected, :]
#                 LY = Phi_Y @ Phi_Y.t()
#                 LY = 0.5 * (LY + LY.t())
#                 cholY = torch.linalg.cholesky(LY + eps * torch.eye(LY.shape[0], device=device, dtype=dtype))
#                 log_det_LY = 2.0 * torch.sum(torch.log(torch.diagonal(cholY)))
#                 ll = log_det_LY - log_norm
#             log_ll_list.append(ll)

#     log_ll_out = torch.stack(log_ll_list) if return_log_likelihood else None
#     return all_selected, log_ll_out


class FixedFeatureDppSampler(NonLearnableSampler):
    def __init__(
        self,
        method: Literal["standard", "k_dpp"] = "standard",
        random_proj_dim: Optional[int] = None,  # if None, no projection
        k: Optional[int] = None,                # only used if method == "k_dpp"
    ) -> None:
        super().__init__()
        assert method in ["standard", "k_dpp"], "method must be 'standard' or 'k_dpp'"
        self.method = method
        self.random_proj_dim = random_proj_dim
        self.k = k
        self.register_buffer("proj_matrix", None)

    def _init_proj_matrix(self, feat_dim: int, device: torch.device):
        """Initialize and store the random projection matrix once."""
        if self.random_proj_dim is not None and self.proj_matrix is None:
            W = torch.randn(feat_dim, self.random_proj_dim, device=device)
            W = W / torch.linalg.norm(W, dim=0, keepdim=True).clamp(min=1e-6)
            self.register_buffer("proj_matrix", W)

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        views = batch["views"]
        feats_list = []

        for view in views:
            assert view.has_features, "FeatureDppSampler requires pre-computed features"
            feats = view.fetch_features()  # [N, D]
            feats_list.append(feats)

        feats = torch.stack(feats_list)  # [B, N, D]
        B, N, D = feats.shape

        if self.random_proj_dim is not None:
            # Lazy init projection once we know D
            self._init_proj_matrix(D, feats.device)
            # Apply fixed random projection across all slides
            feats = feats @ self.proj_matrix  # [B, N, random_proj_dim]
        feats = feats.to(batch['coords_pad'].device)
        return feats, {}

    def sample(
        self,
        scores: Optional[torch.Tensor],
        coord_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        if self.method == "standard":
            samps, _ = sample_low_rank_dpp_batched(
                scores, return_log_likelihood=False
            )
        else:  # k-DPP
            samps, _ = sample_low_rank_k_dpp_batched(
                scores, k=self.k, return_log_likelihood=False
            )
        return samps, {"method": f"feature_{self.method}"}

FEATURE_DPP_SAMPLERS = {
    "FixedFeatureDppSampler": FixedFeatureDppSampler,
}