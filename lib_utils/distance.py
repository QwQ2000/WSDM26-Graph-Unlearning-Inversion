import torch
import math
import numpy as np
from scipy.spatial import distance

# ----------------- Batched PyTorch Distance Functions -----------------

def cosine_distance(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    1 - (P·Q) / (||P|| * ||Q||)  computed along the last dim, in parallel.
    Returns a tensor of shape P.shape[:-1].
    """
    P, Q = P.float(), Q.float()
    num = torch.sum(P * Q, dim=-1)
    denom = torch.norm(P, dim=-1) * torch.norm(Q, dim=-1)
    return 1 - num / (denom + eps)

def euclidean_distance(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """||P - Q||₂ along last dim."""
    return torch.norm(P - Q, p=2, dim=-1)

def correlation_distance(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    1 - corr(P, Q) = 1 - covariance(P,Q)/(σ_P σ_Q).
    Subtracts the mean along last dim, then same as cosine.
    """
    Pc = P - P.mean(dim=-1, keepdim=True)
    Qc = Q - Q.mean(dim=-1, keepdim=True)
    return cosine_distance(Pc, Qc, eps)

def chebyshev_distance(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """max_i |P_i - Q_i| along last dim."""
    return torch.max(torch.abs(P - Q), dim=-1).values

def braycurtis_distance(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """∑|P - Q| / ∑|P + Q| along last dim."""
    num = torch.sum(torch.abs(P - Q), dim=-1)
    den = torch.sum(torch.abs(P + Q), dim=-1)
    return num / (den + eps)

def canberra_distance(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """∑ |P - Q| / (|P| + |Q|) along last dim, with 0/0 → 0."""
    denom = torch.abs(P) + torch.abs(Q)
    frac = torch.abs(P - Q) / torch.where(denom == 0, torch.ones_like(denom), denom)
    return torch.sum(frac, dim=-1)

def cityblock_distance(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """∑|P - Q| along last dim."""
    return torch.sum(torch.abs(P - Q), dim=-1)

def sqeuclidean_distance(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """∑(P - Q)² along last dim."""
    return torch.sum((P - Q) ** 2, dim=-1)


# ----------------- Batched PyTorch Info-Theoretic Functions -----------------

def kl_divergence(P: torch.Tensor, Q: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    ∑ P * log(P/Q) along last dim.
    Both P and Q assumed to be non-negative probability vectors along last dim.
    """
    P = P + epsilon
    Q = Q + epsilon
    return torch.sum(P * torch.log(P / Q), dim=-1)

def js_divergence(P: torch.Tensor,
                  Q: torch.Tensor,
                  epsilon: float = 1e-8,
                  log_base: float = 2.0) -> torch.Tensor:
    """
    Jensen–Shannon *divergence* per row, using log base `log_base`.
    Returns a tensor of shape P.shape[:-1].
    """
    P = P + epsilon
    Q = Q + epsilon
    M = 0.5 * (P + Q)

    # Compute with natural log, then convert to log_base
    factor = 1.0 / math.log(log_base)
    kl1 = torch.sum(P * torch.log(P / M), dim=-1)
    kl2 = torch.sum(Q * torch.log(Q / M), dim=-1)
    js = 0.5 * (kl1 + kl2) * factor
    return js

def js_distance(P: torch.Tensor,
                Q: torch.Tensor,
                epsilon: float = 1e-8,
                log_base: float = 2.0) -> torch.Tensor:
    """
    Jensen–Shannon *distance* per row, i.e. sqrt of the divergence in log base `log_base`.
    """
    div = js_divergence(P, Q, epsilon, log_base)
    return torch.sqrt(div)

def entropy(P: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """−∑ P * log P along last dim."""
    P = P + epsilon
    return -torch.sum(P * torch.log(P), dim=-1)


# ----------------- Validation Script -----------------

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    n, d = 5, 10
    # Random feature matrices
    P = torch.randn(n, d)
    Q = torch.randn(n, d)

    # For information-theoretic measures, make them non-negative and sum to 1 per row
    Pm = torch.abs(P)
    Qm = torch.abs(Q)
    Pm = Pm / Pm.sum(dim=1, keepdim=True)
    Qm = Qm / Qm.sum(dim=1, keepdim=True)

    # Compute all distances/divergences in batch
    torch_results = {
        "cosine":      cosine_distance(P, Q).cpu().numpy(),
        "euclidean":   euclidean_distance(P, Q).cpu().numpy(),
        "correlation": correlation_distance(P, Q).cpu().numpy(),
        "chebyshev":   chebyshev_distance(P, Q).cpu().numpy(),
        "braycurtis":  braycurtis_distance(P.abs(), Q.abs()).cpu().numpy(),  # both non-neg
        "canberra":    canberra_distance(P.abs(), Q.abs()).cpu().numpy(),
        "cityblock":   cityblock_distance(P, Q).cpu().numpy(),
        "sqeuclidean": sqeuclidean_distance(P, Q).cpu().numpy(),
        "KL":          kl_divergence(Pm, Qm).cpu().numpy(),
        "JS":          js_distance(Pm, Qm).cpu().numpy(),
        "Entropy(P)":  entropy(Pm).cpu().numpy(),
    }

    # Compute reference results one row at a time
    numpy_results = {k: [] for k in torch_results}
    for i in range(n):
        p_np = P[i].cpu().numpy()
        q_np = Q[i].cpu().numpy()
        p_m = Pm[i].cpu().numpy()
        q_m = Qm[i].cpu().numpy()

        numpy_results["cosine"].append(distance.cosine(p_np, q_np))
        numpy_results["euclidean"].append(distance.euclidean(p_np, q_np))
        numpy_results["correlation"].append(distance.correlation(p_np, q_np))
        numpy_results["chebyshev"].append(distance.chebyshev(p_np, q_np))
        numpy_results["braycurtis"].append(distance.braycurtis(np.abs(p_np), np.abs(q_np)))
        numpy_results["canberra"].append(distance.canberra(np.abs(p_np), np.abs(q_np)))
        numpy_results["cityblock"].append(distance.cityblock(p_np, q_np))
        numpy_results["sqeuclidean"].append(distance.sqeuclidean(p_np, q_np))

        numpy_results["KL"].append( np.sum(p_m * np.log(p_m / q_m)) )
        numpy_results["JS"].append(  distance.jensenshannon(p_m, q_m, 2.0) )
        numpy_results["Entropy(P)"].append( -np.sum(p_m * np.log(p_m)) )

    # Compare
    print(f"{'metric':<12} | {'max abs diff':>12}")
    print("-" * 28)
    for name, torch_arr in torch_results.items():
        diff = np.abs(torch_arr - np.array(numpy_results[name]))
        print(f"{name:<12} | {diff.max():12.3e}")
