"""
CelebA attribute-rebalancing data-generating process (Dance et al., DecFM, arXiv:2605.07665, App. B.3).

    A = Male        (A = 1 male, A = 0 female)     -- binary treatment
    X = Blond_Hair  (X = 1 blond, X = 0 brown)     -- binary confounder (hair colour)
    Y = 64 x 64 RGB aligned CelebA image, normalised to [-1, 1]

Reimplements the reference construction of Dance et al. (2026, Appendix B.3; the
"custom" split mode of their released code) with identical sampling order and
cell-count rounding, so the controlled observational sample and the balanced
interventional references P(Y(a)) are reproduced exactly.

Paper design: combine the official train + valid splits into one cleaned pool (keep exactly one of
Blond/Brown hair, drop Black/Gray/Bald/Hat), then draw observed and reference images WITHOUT
replacement and DISJOINTLY per (A, X) cell, so that

    observed:   P(X=1) = 0.3,  P(A=1 | X=0) = 0.6,  P(A=1 | X=1) = 0.1     (n_obs = 20000)
    reference:  P_ref(X=1 | do(A=a)) = 0.3                                 (n_ref = 2000 per arm)

The male+blond cell is the limiting one (~1266 images), which is why the paper caps n_obs at ~20000.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

_SPLIT_MAP = {"train": 0, "valid": 1, "val": 1, "test": 2, "all": None}


@dataclass(frozen=True)
class CelebARecord:
    filename: str
    A: int
    X: int


class CelebAAttrs:
    """
    Lightweight CelebA attribute loader (avoids torchvision.datasets.CelebA).

    Reads the standard layout ``root/{img_align_celeba/, list_attr_celeba.txt,
    list_eval_partition.txt}``, keeps the requested split, and with ``clean_hair=True`` restricts to
    images with exactly one of ``hair_pair`` active while dropping ``drop_attrs`` -- so with the
    default ``x_name="Blond_Hair"`` the confounder is X=1 blond / X=0 brown.
    """

    def __init__(self,
                 root: str | Path,
                 split: str = "train",
                 image_size: int = 64,
                 a_name: str = "Male",
                 x_name: str = "Blond_Hair",
                 clean_hair: bool = True,
                 hair_pair: Sequence[str] = ("Blond_Hair", "Brown_Hair"),
                 drop_attrs: Sequence[str] = ("Black_Hair", "Gray_Hair", "Bald", "Wearing_Hat"),
                 crop: str = "decfm"):
        self.root = Path(root)
        self.img_dir = self.root / "img_align_celeba"
        self.image_size = int(image_size)
        self.hair_pair = tuple(hair_pair)
        self.crop = crop            # "decfm": the paper's centre-crop 178

        split_id = _SPLIT_MAP[split.lower()]

        # filename -> membership of the requested split
        keep_files: set[str] = set()
        with open(self.root / "list_eval_partition.txt") as f:
            for line in f:
                name, sid = line.strip().split()
                if split_id is None or int(sid) == split_id:
                    keep_files.add(name)

        with open(self.root / "list_attr_celeba.txt") as f:
            _ = int(f.readline().strip())                       # count line
            attr_names = f.readline().strip().split()
            attr_idx = {name: j for j, name in enumerate(attr_names)}
            a_idx = attr_idx[a_name]
            x_idx = attr_idx[x_name]
            hair_idx = [attr_idx[h] for h in self.hair_pair]
            drop_idx = [attr_idx[h] for h in drop_attrs]

            records: List[CelebARecord] = []
            for line in f:
                parts = line.strip().split()
                filename = parts[0]
                if filename not in keep_files:
                    continue
                vals = tuple(1 if int(v) == 1 else 0 for v in parts[1:])
                if clean_hair:
                    if sum(vals[j] for j in hair_idx) != 1:     # exactly one of blond/brown
                        continue
                    if any(vals[j] == 1 for j in drop_idx):     # drop ambiguous hair categories
                        continue
                records.append(CelebARecord(filename, int(vals[a_idx]), int(vals[x_idx])))

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def A_tensor(self) -> torch.Tensor:
        return torch.tensor([r.A for r in self.records], dtype=torch.long)

    def X_tensor(self) -> torch.Tensor:
        return torch.tensor([r.X for r in self.records], dtype=torch.long)

    def cell_indices(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """Record indices for each (A, X) cell, in record order."""
        A, X = self.A_tensor(), self.X_tensor()
        return {(a, x): torch.where((A == a) & (X == x))[0] for a in (0, 1) for x in (0, 1)}

    def _load_image(self, filename: str) -> torch.Tensor:
        img = Image.open(self.img_dir / filename).convert("RGB")
        w, h = img.size                                         # aligned CelebA is 178 x 218
        crop = min(w, h, 178)                                   # the paper's centre-crop 178
        left, top = (w - crop) // 2, (h - crop) // 2
        img = img.crop((left, top, left + crop, top + crop))
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = 2.0 * arr - 1.0                                   # [0,1] -> [-1,1]
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def load_images(self, indices, device="cpu", dtype=torch.float32, chunk_size=512) -> torch.Tensor:
        """Load images for the given record indices into an (n, 3, S, S) tensor (chunked)."""
        idx_list = indices.detach().cpu().tolist() if isinstance(indices, torch.Tensor) else list(indices)
        chunks = []
        for start in range(0, len(idx_list), chunk_size):
            sub = idx_list[start:start + chunk_size]
            chunks.append(torch.stack([self._load_image(self.records[i].filename).to(dtype) for i in sub]))
        if not chunks:
            return torch.empty(0, 3, self.image_size, self.image_size, dtype=dtype, device=device)
        return torch.cat(chunks, 0).to(device)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def binary_stats(A: torch.Tensor, X: torch.Tensor) -> Dict[str, object]:
    """Marginals, conditionals and (A, X) cell counts for a binary (A, X) sample."""
    A = A.detach().cpu().view(-1).long()
    X = X.detach().cpu().view(-1).long()
    out: Dict[str, object] = {"n": int(A.numel()),
                              "P(A=1)": float(A.float().mean()) if A.numel() else float("nan"),
                              "P(X=1)": float(X.float().mean()) if X.numel() else float("nan"),
                              "cells": {}}
    for a in (0, 1):
        m = A == a
        out[f"P(X=1|A={a})"] = float(X[m].float().mean()) if m.any() else float("nan")
    for x in (0, 1):
        m = X == x
        out[f"P(A=1|X={x})"] = float(A[m].float().mean()) if m.any() else float("nan")
    for a in (0, 1):
        for x in (0, 1):
            out["cells"][(a, x)] = int(((A == a) & (X == x)).sum())
    return out


def print_binary_stats(A: torch.Tensor, X: torch.Tensor, prefix: str = "") -> None:
    s = binary_stats(A, X)
    p = f"{prefix}: " if prefix else ""
    print(f"{p}n={s['n']} | P(A=1)={s['P(A=1)']:.3f} | P(X=1)={s['P(X=1)']:.3f} | "
          f"P(X=1|A=0)={s['P(X=1|A=0)']:.3f} | P(X=1|A=1)={s['P(X=1|A=1)']:.3f} | "
          f"P(A=1|X=0)={s['P(A=1|X=0)']:.3f} | P(A=1|X=1)={s['P(A=1|X=1)']:.3f}")
    print(f"{p}cells={s['cells']}")


# ---------------------------------------------------------------------------
# Deterministic cell counts (largest-remainder rounding, exactly as the paper code)
# ---------------------------------------------------------------------------

def _counts_from_probs(probs: Dict[Tuple[int, int], float], n: int) -> Dict[Tuple[int, int], int]:
    """Integer cell counts summing exactly to n via largest-remainder rounding (deterministic)."""
    raw = {k: n * float(v) for k, v in probs.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n - sum(counts.values())
    order = sorted(raw, key=lambda k: (raw[k] - counts[k], str(k)), reverse=True)
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def _obs_cell_counts(n: int, target_px1: float, p_a1_x0: float, p_a1_x1: float):
    q, p0, p1 = target_px1, p_a1_x0, p_a1_x1
    probs = {(0, 0): (1 - q) * (1 - p0), (1, 0): (1 - q) * p0,
             (0, 1): q * (1 - p1), (1, 1): q * p1}
    return _counts_from_probs(probs, n)


def _ref_cell_counts(n_per_arm: int, target_px1: float, arm: int):
    return _counts_from_probs({(arm, 0): 1 - target_px1, (arm, 1): target_px1}, n_per_arm)


def _load_combined_pool(root, source_splits, image_size, a_name, x_name, clean_hair, hair_pair, verbose, crop="decfm"):
    """Concatenate the cleaned records of several official splits into one pool (order = split order)."""
    pool = CelebAAttrs(root=root, split=source_splits[0], image_size=image_size, a_name=a_name,
                       x_name=x_name, clean_hair=clean_hair, hair_pair=hair_pair, crop=crop)
    for sp in source_splits[1:]:
        pool.records = pool.records + CelebAAttrs(root=root, split=sp, image_size=image_size,
                                                  a_name=a_name, x_name=x_name, clean_hair=clean_hair,
                                                  hair_pair=hair_pair, crop=crop).records
    if verbose:
        print(f"Loaded {len(pool.records)} CelebA custom-pool examples from splits={tuple(source_splits)}")
        print_binary_stats(pool.A_tensor(), pool.X_tensor(), prefix="custom pool")
    return pool


# ---------------------------------------------------------------------------
# Main generator (paper defaults; custom disjoint split from a combined pool)
# ---------------------------------------------------------------------------

def generate_celeba_confounding(*,
                                root: str | Path,
                                image_size: int = 64,
                                n_obs: int = 20_000,
                                n_ref: int = 2_000,
                                source_splits: Sequence[str] = ("train", "valid"),
                                a_name: str = "Male",
                                x_name: str = "Blond_Hair",
                                clean_hair: bool = True,
                                hair_pair: Sequence[str] = ("Blond_Hair", "Brown_Hair"),
                                target_px1: float = 0.3,
                                p_a1_x0: float = 0.6,
                                p_a1_x1: float = 0.1,
                                crop: str = "decfm",
                                seed: int = 0,
                                device: torch.device | str = "cpu",
                                dtype: torch.dtype = torch.float32,
                                verbose: bool = True) -> Dict[str, object]:
    """
    Build the DecFM CelebA confounding benchmark (App. B.3), defaults = the paper's exact settings.

    Returns a dict with fields ``X, A, Y`` (observed sample), ``Y0_ref, Y1_ref`` (balanced P(Y(a))
    references), ``X0_ref, X1_ref``, ``pi_true`` (oracle step propensity) and ``meta`` (config +
    sampled record indices).
    """
    g = torch.Generator().manual_seed(seed)
    source_splits = tuple(s.lower() for s in source_splits)

    obs_counts = _obs_cell_counts(n_obs, target_px1, p_a1_x0, p_a1_x1)
    ref_counts = {0: _ref_cell_counts(n_ref, target_px1, 0), 1: _ref_cell_counts(n_ref, target_px1, 1)}

    pool = _load_combined_pool(root, source_splits, image_size, a_name, x_name, clean_hair, hair_pair, verbose, crop)
    pool_cells = pool.cell_indices()

    # Draw observed + both references disjointly within each (A, X) cell (fixed cell order).
    obs_parts: List[torch.Tensor] = []
    ref_parts: Dict[int, List[torch.Tensor]] = {0: [], 1: []}
    for cell in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        obs_n = int(obs_counts.get(cell, 0))
        r0, r1 = int(ref_counts[0].get(cell, 0)), int(ref_counts[1].get(cell, 0))
        total = obs_n + r0 + r1
        avail = int(pool_cells[cell].numel())
        if total > avail:
            raise ValueError(f"cell {cell}: need obs={obs_n}+ref={r0 + r1}={total}, available={avail}. "
                             "Reduce n_obs/n_ref or add source_splits.")
        if total == 0:
            continue
        shuffled = pool_cells[cell][torch.randperm(avail, generator=g)]
        if obs_n:
            obs_parts.append(shuffled[:obs_n])
        cur = obs_n
        if r0:
            ref_parts[0].append(shuffled[cur:cur + r0]); cur += r0
        if r1:
            ref_parts[1].append(shuffled[cur:cur + r1])

    obs_idx = torch.cat(obs_parts)
    obs_idx = obs_idx[torch.randperm(obs_idx.numel(), generator=g)]
    ref_idx = {}
    for arm in (0, 1):
        idx = torch.cat(ref_parts[arm]) if ref_parts[arm] else torch.empty(0, dtype=torch.long)
        ref_idx[arm] = idx[torch.randperm(idx.numel(), generator=g)]

    # Materialise tensors.
    A = torch.tensor([pool.records[int(i)].A for i in obs_idx], dtype=torch.long).view(-1, 1).to(device)
    X = torch.tensor([pool.records[int(i)].X for i in obs_idx], dtype=torch.float32).view(-1, 1).to(device)
    Y = pool.load_images(obs_idx, device=device, dtype=dtype)
    Y_ref, X_ref = {}, {}
    for arm in (0, 1):
        Y_ref[arm] = pool.load_images(ref_idx[arm], device=device, dtype=dtype)
        X_ref[arm] = torch.tensor([pool.records[int(i)].X for i in ref_idx[arm]],
                                  dtype=torch.float32).view(-1, 1).to(device)

    pi_true = torch.where(X.view(-1) > 0.5, torch.full((X.shape[0],), float(p_a1_x1), device=device),
                          torch.full((X.shape[0],), float(p_a1_x0), device=device)).view(-1, 1)

    if verbose:
        print_binary_stats(A.cpu(), X.cpu(), prefix="observed")
        print_binary_stats(torch.zeros_like(X_ref[0]).long(), X_ref[0].cpu(), prefix="target ref a=0")
        print_binary_stats(torch.ones_like(X_ref[1]).long(), X_ref[1].cpu(), prefix="target ref a=1")

    meta = {"root": str(root), "image_size": int(image_size), "n_obs": int(n_obs), "n_ref": int(n_ref),
            "source_splits": source_splits, "a_name": a_name, "x_name": x_name, "hair_pair": tuple(hair_pair),
            "target_px1": float(target_px1), "p_a1_x0": float(p_a1_x0), "p_a1_x1": float(p_a1_x1),
            "crop": crop, "seed": int(seed), "obs_counts": {str(k): int(v) for k, v in obs_counts.items()},
            "obs_indices": obs_idx.cpu(), "ref_indices_a0": ref_idx[0].cpu(), "ref_indices_a1": ref_idx[1].cpu()}

    return {"X": X, "A": A, "Y": Y, "pi_true": pi_true, "Y0_ref": Y_ref[0], "Y1_ref": Y_ref[1],
            "X0_ref": X_ref[0], "X1_ref": X_ref[1], "meta": meta}


def load_celeba_paper(cache_path: str | Path, *, root: str | Path, seed: int = 0,
                      regenerate: bool = False, device: torch.device | str = "cpu",
                      verbose: bool = True, **gen_kwargs) -> Dict[str, object]:
    """
    Load the DecFM CelebA dataset from ``cache_path`` (a .pt made by :func:`generate_celeba_confounding`),
    generating and caching it on first use. Essential because materialising ~24k images takes minutes.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not regenerate:
        data = torch.load(cache_path, map_location=device, weights_only=False)
        # Guard the common footgun: the cache filename does not encode n_obs, so a request for a
        # different n_obs silently returns the old cache (gen_kwargs are only used when generating).
        requested_n_obs = gen_kwargs.get("n_obs")
        cached_n_obs = data["A"].shape[0]
        if requested_n_obs is not None and int(requested_n_obs) != int(cached_n_obs):
            print(f"[load_celeba_paper] WARNING: cache '{cache_path.name}' holds {cached_n_obs} samples, "
                  f"but n_obs={requested_n_obs} was requested. Returning the cache unchanged. To rebuild, "
                  f"pass regenerate=True or use a cache_path that encodes n_obs.")
        if verbose:
            print_binary_stats(data["A"], data["X"], prefix="observed (cached)")
        return data
    data = generate_celeba_confounding(root=root, seed=seed, device=device, verbose=verbose, **gen_kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    if verbose:
        print(f"cached -> {cache_path}")
    return data


def to_image_range(Y: torch.Tensor) -> torch.Tensor:
    """[-1, 1] model range -> [0, 1] for plotting (clamped)."""
    return ((Y + 1.0) / 2.0).clamp(0.0, 1.0)


class OracleBinaryPropensity(torch.nn.Module):
    """Known binary-X propensity table pi(a=1|x) for the controlled design."""

    def __init__(self, p_a1_x0: float = 0.6, p_a1_x1: float = 0.1, eps: float = 1e-3):
        super().__init__()
        self.p_a1_x0, self.p_a1_x1, self.eps = float(p_a1_x0), float(p_a1_x1), float(eps)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = X.squeeze(-1).long()
        p1 = torch.where(x == 1, torch.full_like(x, self.p_a1_x1, dtype=torch.float32),
                         torch.full_like(x, self.p_a1_x0, dtype=torch.float32))
        return p1.to(X.device).clamp(self.eps, 1.0 - self.eps)
