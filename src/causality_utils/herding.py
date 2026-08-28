"""
Kernel herding from an estimated mean embedding.

Kernel herding (Chen, Welling, Smola, 2010) turns a mean embedding into a set of
representative points by greedy selection. Applied to the counterfactual policy
mean embedding it is Algorithm 2 of Zenati, Bozkurt and Gretton (2025): given the
estimated embedding chi_hat(pi) = sum_i v_i k_Y(y_i, .) and a finite candidate
set, it repeatedly selects

    y_t = argmax_y [ chi_hat(pi)(y) - (1/t) sum_{s < t} k_Y(y_s, y) ].

The first term attracts selections to regions the embedding puts mass on; the
second repels them from what has already been chosen, so the empirical measure of
the selections tracks the embedding.

This is a *selection* sampler: every output is an element of the candidate set. It
therefore cannot leave the data manifold — but neither can it synthesise an
outcome the candidate set does not contain, and repeated selections are permitted,
so the result can be a quantisation of the target rather than a sample from it.
Report `herding_diagnostics` alongside the selections to make that visible.
"""

from typing import Dict, Optional, Tuple, Union

import torch


def evaluate_embedding(weights: torch.Tensor,
                       kernel_y,
                       Y_support: torch.Tensor,
                       y_query: torch.Tensor) -> torch.Tensor:
    """
    Evaluates chi_hat(pi)(y) = sum_i v_i k_Y(y_i, y) at the query points.

    Parameters
    ----------
    weights : torch.Tensor of shape (n,)
        Expansion weights of the embedding over `Y_support`; may be signed.
    kernel_y : Kernel
    Y_support : torch.Tensor of shape (n, d_y)
        Expansion points of the embedding, e.g. the logged outcomes.
    y_query : torch.Tensor of shape (m, d_y)

    Returns
    -------
    torch.Tensor of shape (m,)
    """
    return kernel_y(Y_support, y_query).T @ weights


def kernel_herding(weights: torch.Tensor,
                   kernel_y,
                   Y_support: torch.Tensor,
                   candidates: torch.Tensor,
                   num_samples: int,
                   return_indices: bool = False
                   ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Kernel herding over a finite candidate set (Algorithm 2).

    Parameters
    ----------
    weights : torch.Tensor of shape (n,)
    kernel_y : Kernel
        Fixed-bandwidth kernel; lock any length-scale heuristic beforehand.
    Y_support : torch.Tensor of shape (n, d_y)
    candidates : torch.Tensor of shape (c, d_y)
        Set the selections are drawn from. Using the logged outcomes makes every
        selection a real observed outcome.
    num_samples : int
    return_indices : bool
        If True, also return the index into `candidates` of each selection, which
        is what lets a selection be mapped back to the observation it came from.

    Returns
    -------
    selections : torch.Tensor of shape (num_samples, d_y)
    indices : torch.Tensor of shape (num_samples,), only if `return_indices`.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if weights.shape[0] != Y_support.shape[0]:
        raise ValueError("weights and Y_support must have the same length")

    chi = evaluate_embedding(weights, kernel_y, Y_support, candidates)
    repulsion = torch.zeros_like(chi)

    indices = torch.empty(num_samples, dtype=torch.long, device=candidates.device)
    for t in range(num_samples):
        objective = chi if t == 0 else chi - repulsion / t
        index = int(objective.argmax())
        indices[t] = index
        repulsion = repulsion + kernel_y(candidates[index].view(1, -1), candidates)[0]

    selections = candidates[indices]
    return (selections, indices) if return_indices else selections


def herding_diagnostics(indices: torch.Tensor,
                        n_candidates: Optional[int] = None) -> Dict[str, float]:
    """
    Summarises how much of the candidate set a herding run actually used.

    Herding selects with replacement, so a run can concentrate on a few points and
    return a quantisation rather than a sample. The unique count and the maximum
    multiplicity make that explicit.

    Parameters
    ----------
    indices : torch.Tensor of shape (num_samples,)
    n_candidates : int, optional

    Returns
    -------
    dict with keys "n_selected", "n_unique", "unique_fraction",
    "max_multiplicity", and "candidate_coverage" when `n_candidates` is given.
    """
    counts = torch.bincount(indices.flatten())
    nonzero = counts[counts > 0]
    diagnostics = {
        "n_selected": int(indices.numel()),
        "n_unique": int(nonzero.numel()),
        "unique_fraction": float(nonzero.numel() / indices.numel()),
        "max_multiplicity": int(counts.max()),
    }
    if n_candidates is not None:
        diagnostics["candidate_coverage"] = float(nonzero.numel() / n_candidates)
    return diagnostics
