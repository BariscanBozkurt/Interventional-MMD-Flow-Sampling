"""Rank and plot the 128 largest CelebA IMQ--Euler particle changes per arm.

The diagnostic reproduces the seed-0 arm-specific IMQ flows from the paired
sampler ablation.  It compares each decoded logged initializer ``D(E(Y_i))``
with the decoded endpoint of that same particle and displays a common-scale
absolute pixel-change heatmap.  Ranking by the full-image RGB displacement is
post hoc and is used only for qualitative inspection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parents[1]
SRC = DEPLOY / "src"
HAIR_DIAGNOSTIC = (
    DEPLOY
    / "AdditonalExperiments"
    / "CelebA_Hair_Colour_Diagnostic"
    / "celeba_top128_change_diagnostic.py"
)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoencoder_utils import load_sd_vae
from causality_utils import cpme_dr_weights_projected_nystrom, select_regularization
from dens_ratio_utils import LogisticPropensityModel
from kernel_utils import imq_c_from_median

from imq_euler import euler_flow, target_self_inner_product


def _load_hair_diagnostic():
    spec = importlib.util.spec_from_file_location(
        "celeba_hair_colour_diagnostic", HAIR_DIAGNOSTIC
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import plotting helpers from {HAIR_DIAGNOSTIC}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hair = _load_hair_diagnostic()


SOURCE_DGP = DEPLOY / "data" / "celeba_dgp" / "celeba_paper_64_n20000_seed0.pt"
HEADLINE_PARTICLES = (
    DEPLOY
    / "Simulations"
    / "CelebA"
    / "dr_mmd_flow"
    / "results"
    / "celeba_est_headline_sz256_nobs20000_M2000_arms01_ds0_drnosplit_projnys_estplugin-ipw-dr_particles.npz"
)
DEFAULT_CACHE = HERE / "cache" / "celeba_imq_euler_top128_n20000_M2000.pt"
DEFAULT_FIGURES = HERE / "figures" / "imq_euler_eta0p5"
DEFAULT_RESULTS = HERE / "results" / "imq_euler_eta0p5"

N_OBS = 20_000
N_PARTICLES = 2_000
N_LANDMARKS = 1_024
IMAGE_SIZE = 256
N_ITERATIONS = 4_000
IMQ_ETA = 0.5
IMQ_C_MULTIPLIER = 1.0
DATA_SEED = 0
PIPELINE_SEED = 0
KEEP = 128


def deterministic_imq_bandwidth(points: torch.Tensor) -> float:
    """Use the exact fixed-subset median rule from the production ablation."""
    n_subset = min(5_000, len(points))
    rows = torch.randperm(
        len(points), generator=torch.Generator().manual_seed(0)
    )[:n_subset].to(points.device)
    return IMQ_C_MULTIPLIER * imq_c_from_median(
        points[rows], subset_limit=n_subset
    )


def build_cache(cache_path: Path, celeba_root: Path) -> None:
    """Reproduce both seed-0 IMQ flows and retain their top-128 decoded pairs."""
    if not torch.cuda.is_available():
        raise RuntimeError("--rebuild-cache requires a visible CUDA device")
    if not SOURCE_DGP.is_file():
        raise FileNotFoundError(f"missing seed-0 observation metadata: {SOURCE_DGP}")
    if not HEADLINE_PARTICLES.is_file():
        raise FileNotFoundError(
            f"missing headline particle-index artifact: {HEADLINE_PARTICLES}"
        )

    np.random.seed(PIPELINE_SEED)
    torch.manual_seed(PIPELINE_SEED)
    torch.cuda.manual_seed_all(PIPELINE_SEED)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda")
    print(f"device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

    hair.CELEBA_ROOT = celeba_root
    hair.SOURCE_DGP = SOURCE_DGP
    vae = load_sd_vae(device)
    A, X, observed_indices, raw_latents = hair.load_headline_latents(vae, device)
    latent_mean = raw_latents.mean(dim=0, keepdim=True)
    latent_std = raw_latents.std(dim=0, keepdim=True).clamp_min(1e-3)
    standardized = (raw_latents - latent_mean) / latent_std
    del raw_latents

    cell = (A * 2 + X).long()
    gram = lambda rows, cols: (
        cell[rows][:, None] == cell[cols][None, :]
    ).double()
    propensity = LogisticPropensityModel().fit(
        X.double().unsqueeze(1), A.double()
    )
    e1 = propensity.predict(X.double().unsqueeze(1)).reshape(-1).double()
    cv_rows = torch.randperm(
        N_OBS, generator=torch.Generator().manual_seed(0)
    )[:2_000]
    ridge, _ = select_regularization(
        gram(cv_rows.numpy(), cv_rows.numpy()),
        standardized[cv_rows, 0].double().cpu(),
        seed=0,
    )
    c_imq = deterministic_imq_bandwidth(standardized)
    headline_particles = np.load(HEADLINE_PARTICLES)
    print(f"ridge={ridge:g}, c_imq={c_imq:.6f}, eta={IMQ_ETA}", flush=True)

    cache = {
        "arms": {},
        "sampler": "imq_euler_deterministic",
        "kernel": "normalized_imq_beta_minus_half",
        "n_obs": N_OBS,
        "n_particles": N_PARTICLES,
        "keep": KEEP,
        "data_seed": DATA_SEED,
        "pipeline_seed": PIPELINE_SEED,
        "image_size": IMAGE_SIZE,
        "flow_iterations": N_ITERATIONS,
        "imq_eta": IMQ_ETA,
        "imq_c": c_imq,
        "imq_c_multiplier": IMQ_C_MULTIPLIER,
        "noise_beta0": 0.0,
        "n_landmarks": N_LANDMARKS,
        "ridge_lambda": float(ridge),
        "ranking": "decoded start-to-final full-image RGB L2",
        "start_definition": "D(E(Y_initial)); excludes encoder-decoder reconstruction from the heatmap",
    }

    for arm in (0, 1):
        print(f"constructing projected-Nystrom DR target for arm {arm}", flush=True)
        w_hat = (A == arm).double() / (e1 if arm == 1 else 1.0 - e1)
        on_arm = (A == arm).double()
        gram_pi = lambda rows, cols, on_arm=on_arm: (
            (X[rows][:, None] == X[cols][None, :]).double()
            * on_arm[rows][:, None]
        )
        v_dr = cpme_dr_weights_projected_nystrom(
            gram,
            gram_pi,
            w_hat,
            ridge,
            n_landmarks=N_LANDMARKS,
            generator=torch.Generator().manual_seed(PIPELINE_SEED),
        ).float().to(device)

        arm_rows = torch.nonzero(A == arm).ravel()
        local_pick = torch.randperm(
            len(arm_rows), generator=torch.Generator().manual_seed(PIPELINE_SEED)
        )[:N_PARTICLES]
        particle_rows = arm_rows[local_pick]
        expected = headline_particles[f"ds0_r0_a{arm}"]
        if not np.array_equal(particle_rows.numpy(), expected):
            raise AssertionError(f"arm {arm} initial particles differ from headline")

        initial = standardized[particle_rows.to(device)].clone()
        target_self = target_self_inner_product(standardized, v_dr, c_imq)
        print(f"running arm {arm} IMQ--Euler flow", flush=True)
        final, info = euler_flow(
            initial,
            standardized,
            v_dr,
            c=c_imq,
            eta=IMQ_ETA,
            n_iterations=N_ITERATIONS,
            target_self=target_self,
            noise_beta0=0.0,
            seed=PIPELINE_SEED + 1,
            log_every=500,
        )
        if info["diverged"] or info["steps_completed"] != N_ITERATIONS:
            raise RuntimeError(
                f"arm {arm} flow failed after {info['steps_completed']} steps"
            )
        print(
            f"arm {arm}: normalized IMQ MMD^2 "
            f"{info['mmd2_initial']:.6g} -> {info['mmd2_final']:.6g}; "
            f"mean transport {info['mean_transport']:.3f}",
            flush=True,
        )

        print(f"decoding arm {arm} starts and endpoints", flush=True)
        decoded_start = hair.decode_all(vae, initial, latent_mean, latent_std)
        decoded_final = hair.decode_all(vae, final, latent_mean, latent_std)
        change = (decoded_final - decoded_start).flatten(1).norm(dim=1)
        ranked = torch.argsort(change, descending=True)[:KEEP].cpu()
        ranked_rows = particle_rows[ranked].cpu()

        cache["arms"][arm] = {
            "start": hair.uint8_images(decoded_start, ranked),
            "end": hair.uint8_images(decoded_final, ranked),
            "change": change[ranked].cpu(),
            "particle_index": ranked,
            "logged_row": ranked_rows,
            "pool_record_index": observed_indices[ranked_rows],
            "initial_X": X[ranked_rows].cpu(),
            "change_all_median": float(change.median()),
            "change_all_max": float(change.max()),
            "mean_transport": info["mean_transport"],
            "mmd2_initial": info["mmd2_initial"],
            "mmd2_final": info["mmd2_final"],
        }
        del decoded_start, decoded_final, change, final, initial, v_dr
        torch.cuda.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"saved IMQ top-128 cache: {cache_path}", flush=True)


def plot_cache(cache_path: Path, figures: Path, results: Path) -> None:
    """Use the audited reviewer-response plotting convention with an IMQ title."""
    hair.CACHE = cache_path
    hair.OUT_DIR = figures
    hair.RESULTS_DIR = results
    original_make_page = hair.make_page

    def make_imq_page(cache, arm, first, vmax):
        figure = original_make_page(cache, arm, first, vmax)
        count = min(hair.PER_PAGE, KEEP - first)
        last = first + count
        figure._suptitle.set_text(
            f"CelebA IMQ--Euler flow: decoded pixel-change ranks "
            f"{first + 1}\N{EN DASH}{last}, arm $a={arm}$"
        )
        return figure

    hair.make_page = make_imq_page
    hair.plot_cache()
    metadata = torch.load(cache_path, map_location="cpu", weights_only=False)
    provenance = {
        key: metadata[key]
        for key in (
            "sampler",
            "kernel",
            "n_obs",
            "n_particles",
            "data_seed",
            "image_size",
            "flow_iterations",
            "imq_eta",
            "imq_c",
            "noise_beta0",
            "n_landmarks",
            "ridge_lambda",
            "ranking",
            "start_definition",
        )
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--celeba-root",
        type=Path,
        help="CelebA root; required with --rebuild-cache",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = args.cache.expanduser().resolve()
    figures = args.out_dir.expanduser().resolve()
    results = args.results_dir.expanduser().resolve()
    if args.rebuild_cache:
        if args.celeba_root is None:
            raise ValueError("--celeba-root is required with --rebuild-cache")
        build_cache(cache_path, args.celeba_root.expanduser().resolve())
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"missing {cache_path}; run once with --rebuild-cache"
        )
    plot_cache(cache_path, figures, results)


if __name__ == "__main__":
    main()
