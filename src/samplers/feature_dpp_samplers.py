import torch
from typing import List, Dict, Any, Optional, Tuple, Literal
from .base_sampler import NonLearnableSampler

@torch.no_grad()
def sample_low_rank_dpp_batched(
    Phi: torch.Tensor,                      # [B, N, D]
    return_log_likelihood: bool = False,
    eps: float = 1e-10
) -> Tuple[List[List[int]], Optional[torch.Tensor]]:
    """
    Batched sampling from L-ensemble DPPs with low-rank L = Phi @ Phi.T.
    Each batch element is treated as an independent DPP.

    Args:
        Phi: [B, N, D] tensor of feature matrices
        return_log_likelihood: if True, also return per-batch log-likelihoods

    Returns:
        selected_indices_list: list of lists, each with selected indices for that batch item
        optional_log_likelihoods: [B] tensor (if requested)
    """
    assert Phi.dim() == 3, "Phi must be [B, N, D]"
    B, N, D = Phi.shape
    device = Phi.device
    dtype = Phi.dtype

    # --- Step 1: dual kernel and eigendecomposition ---
    # C_b = Phi_b^T Phi_b  -> [B, D, D]
    C = torch.matmul(Phi.transpose(1, 2), Phi)
    #TODO: GPU batch this. 
    # eigendecomposition (batched)
    evals, evecs = torch.linalg.eigh(C)     # [B, D], [B, D, D]
    evals = torch.clamp(evals, min=0.0)

    # --- Step 2: Bernoulli selection ---
    probs = evals / (evals + 1.0)
    keep = torch.bernoulli(torch.clamp(probs, 0.0, 1.0)).bool()  # [B, D]

    # --- Step 3: normalization constant ---
    log_norm = None
    if return_log_likelihood:
        I = torch.eye(D, dtype=dtype, device=device).expand(B, D, D)
        chol = torch.linalg.cholesky(I + C)
        log_norm = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), dim=1)  # [B]

    # --- Step 4: loop over batches for sampling ---
    all_selected = []
    log_ll_list = []

    for b in range(B):
        kept = keep[b]
        k = int(kept.sum().item())

        if k == 0:
            # fallback: choose one uniformly at random
            idx = torch.randint(0, N, (1,), device=device)
            all_selected.append(idx)
            if return_log_likelihood:
                log_ll_list.append(-log_norm[b])
            continue

        # Build projection basis V
        U = evecs[b, :, kept]                              # [D, k]
        lam = torch.clamp(evals[b, kept], min=eps)         # [k]
        V = Phi[b] @ (U @ torch.diag(lam.rsqrt()))         # [N, k]
        V, _ = torch.linalg.qr(V, mode="reduced")          # [N, k]

        selected = []
        while V.shape[1] > 0:
            row_sq = torch.sum(V * V, dim=1)
            total = row_sq.sum()
            if total <= eps:
                break
            i = torch.multinomial(row_sq / total, 1).item()
            selected.append(i)

            vi = V[i, :]
            nrm = torch.linalg.norm(vi)
            if nrm <= eps:
                V[i, :] = 0.0
                if torch.linalg.matrix_rank(V) < V.shape[1]:
                    V, _ = torch.linalg.qr(V, mode="reduced")
                continue

            f = (vi / nrm).reshape(-1, 1)
            sign = 1.0 if f[0, 0] >= 0 else -1.0
            u = f.clone()
            u[0, 0] += sign
            u_nrm = torch.linalg.norm(u)
            if u_nrm <= eps:
                H = torch.eye(f.numel(), dtype=dtype, device=device) - 2.0 * (f @ f.t())
            else:
                v = u / u_nrm
                H = torch.eye(f.numel(), dtype=dtype, device=device) - 2.0 * (v @ v.t())
            V = V @ H
            V = V[:, 1:]
            if V.numel() > 0:
                V, _ = torch.linalg.qr(V, mode="reduced")
        
        all_selected.append(torch.tensor(selected))

        # optional log-likelihood
        if return_log_likelihood:
            if len(selected) == 0:
                ll = -log_norm[b]
            else:
                Phi_Y = Phi[b, selected, :]                          # [|Y|, D]
                LY = Phi_Y @ Phi_Y.t()
                LY = 0.5 * (LY + LY.t())
                cholY = torch.linalg.cholesky(LY + eps * torch.eye(LY.shape[0], device=device, dtype=dtype))
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
    eps: float = 1e-10
) -> Tuple[List[List[int]], Optional[torch.Tensor]]:
    """
    Batched sampling from k-DPPs with low-rank L = Phi @ Phi.T.
    Each batch element is treated as an independent k-DPP.

    Args:
        Phi: [B, N, D] tensor of feature matrices
        k: number of elements to sample
        return_log_likelihood: if True, returns per-batch log-likelihoods
        eps: numerical stability term

    Returns:
        all_selected: list of lists of sampled indices per batch
        log_ll_out: [B] tensor if return_log_likelihood=True, else None
    """
    assert Phi.dim() == 3, "Phi must be [B, N, D]"
    B, N, D = Phi.shape
    device = Phi.device
    dtype = Phi.dtype

    # --- Step 1: dual kernel C = Phi^T Phi  -> [B, D, D]
    C = torch.matmul(Phi.transpose(1, 2), Phi)

    # --- Step 2: eigendecomposition (batched)
    evals, evecs = torch.linalg.eigh(C)   # [B, D], [B, D, D]
    evals = torch.clamp(evals, min=0.0)

    all_selected = []
    log_ll_list = []

    for b in range(B):
        lam = evals[b].cpu().numpy()
        D_b = len(lam)

        # --- DP table for elementary symmetric polynomials ---
        E = torch.zeros((D_b + 1, k + 1), dtype=dtype, device=device)
        E[:, 0] = 1.0
        for i in range(1, D_b + 1):
            lam_i = lam[i - 1]
            for j in range(1, min(i, k) + 1):
                E[i, j] = E[i - 1, j] + lam_i * E[i - 1, j - 1]

        # --- Sample eigenvectors to keep ---
        selected_eigs = []
        rem = k
        for i in reversed(range(D_b)):
            if rem == 0:
                break
            prob = lam[i] * E[i, rem - 1] / (E[i + 1, rem] + eps)
            if torch.rand(1).item() < prob:
                selected_eigs.append(i)
                rem -= 1
        selected_eigs.reverse()

        keep = torch.zeros(D_b, dtype=torch.bool, device=device)
        keep[selected_eigs] = True

        # --- log normalization term (log E[D,k]) ---
        log_norm = None
        if return_log_likelihood:
            log_norm = torch.log(E[D_b, k] + eps)

        # --- Projection DPP sampling ---
        U = evecs[b, :, keep]                            # [D, k]
        lam_k = torch.clamp(evals[b, keep], min=eps)     # [k]
        V = Phi[b] @ (U @ torch.diag(lam_k.rsqrt()))     # [N, k]
        V, _ = torch.linalg.qr(V, mode="reduced")

        selected = []
        while V.shape[1] > 0:
            row_sq = torch.sum(V * V, dim=1)
            total = row_sq.sum()
            if total <= eps:
                break
            i = torch.multinomial(row_sq / total, 1).item()
            selected.append(i)

            vi = V[i, :]
            nrm = torch.linalg.norm(vi)
            if nrm <= eps:
                V[i, :] = 0.0
                if torch.linalg.matrix_rank(V) < V.shape[1]:
                    V, _ = torch.linalg.qr(V, mode="reduced")
                continue

            f = (vi / nrm).reshape(-1, 1)
            sign = 1.0 if f[0, 0] >= 0 else -1.0
            u = f.clone()
            u[0, 0] += sign
            u_nrm = torch.linalg.norm(u)
            if u_nrm <= eps:
                H = torch.eye(f.numel(), dtype=dtype, device=device) - 2.0 * (f @ f.t())
            else:
                v = u / u_nrm
                H = torch.eye(f.numel(), dtype=dtype, device=device) - 2.0 * (v @ v.t())
            V = V @ H
            V = V[:, 1:]
            if V.numel() > 0:
                V, _ = torch.linalg.qr(V, mode="reduced")

        all_selected.append(torch.tensor(selected, device=device))

        # --- Optional log-likelihood ---
        if return_log_likelihood:
            if len(selected) == 0:
                ll = -log_norm
            else:
                Phi_Y = Phi[b, selected, :]
                LY = Phi_Y @ Phi_Y.t()
                LY = 0.5 * (LY + LY.t())
                cholY = torch.linalg.cholesky(LY + eps * torch.eye(LY.shape[0], device=device, dtype=dtype))
                log_det_LY = 2.0 * torch.sum(torch.log(torch.diagonal(cholY)))
                ll = log_det_LY - log_norm
            log_ll_list.append(ll)

    log_ll_out = torch.stack(log_ll_list) if return_log_likelihood else None
    return all_selected, log_ll_out


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