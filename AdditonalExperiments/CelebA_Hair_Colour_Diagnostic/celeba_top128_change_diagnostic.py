"""Regenerate and plot the 128 largest CelebA particle changes per arm.

The active headline cache retains only the top 64 particles.  With
``--rebuild-cache`` this script reproduces the seed-0, n=20,000, M=2,000
headline flow and stores the top 128 decoded start/end pairs.  Plotting is
paginated so that every pair and its absolute-difference map remain legible.
The submitted notebooks and manuscript are never modified.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parents[1]
SRC = DEPLOY / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoencoder_utils import decode_latents, encode_images, load_sd_vae
from causality_utils import cpme_dr_weights_projected_nystrom, select_regularization
from data_utils.celeba import _load_combined_pool
from dens_ratio_utils import LogisticPropensityModel
from mmd_flow_utils import adamw_riesz_flow


SOURCE_DGP = DEPLOY / "data" / "celeba_dgp" / "celeba_paper_64_n20000_seed0.pt"
SOURCE_TOP64 = (
    DEPLOY
    / "NotebookExamples"
    / "Interventional_MMD_Flow"
    / "figures"
    / "celeba_figure_cache_n20000.pt"
)
CACHE_DIR = HERE / "cache"
CACHE = CACHE_DIR / "celeba_figure_cache_top128_n20000_M2000.pt"
OUT_DIR = HERE / "figures"
RESULTS_DIR = HERE / "results"
CELEBA_ROOT = None

N_OBS = 20_000
N_PARTICLES = 2_000
N_LANDMARKS = 1_024
IMAGE_SIZE = 256
FLOW_ITERATIONS = 4_000
LEARNING_RATE = 0.001
WARMUP_STEPS = 100
SEED = 0
KEEP = 128
PER_PAGE = 16
N_COLUMNS = 8


def load_headline_latents(vae, device: torch.device):
    """Load the exact observed rows and encode their 256-pixel images in chunks."""
    print(f"loading observed-row metadata: {SOURCE_DGP}", flush=True)
    saved = torch.load(SOURCE_DGP, map_location="cpu", weights_only=False)
    A = saved["A"].view(-1).long().clone()
    X = saved["X"].view(-1).long().clone()
    observed_indices = saved["meta"]["obs_indices"].clone()
    del saved

    pool = _load_combined_pool(
        CELEBA_ROOT, ("train", "valid"), IMAGE_SIZE, "Male", "Blond_Hair",
        True, ("Blond_Hair", "Brown_Hair"), False, crop="decfm"
    )
    latent_parts = []
    image_chunk = 64
    for start in range(0, N_OBS, image_chunk):
        stop = min(N_OBS, start + image_chunk)
        images = pool.load_images(observed_indices[start:stop], chunk_size=image_chunk)
        latent_parts.append(encode_images(vae, images, chunk_size=16).cpu())
        if stop % 1_000 == 0 or stop == N_OBS:
            print(f"encoded {stop:5d}/{N_OBS} observed images", flush=True)
        del images

    latents = torch.cat(latent_parts).to(device)
    print(f"observed latents: {tuple(latents.shape)}", flush=True)
    return A, X, observed_indices, latents


@torch.no_grad()
def decode_all(vae, latents: torch.Tensor, latent_mean: torch.Tensor,
               latent_std: torch.Tensor) -> torch.Tensor:
    """Decode a full particle cloud to CPU images in manageable chunks."""
    images = []
    for start in range(0, len(latents), 64):
        raw = latents[start:start + 64] * latent_std + latent_mean
        images.append(decode_latents(vae, raw, IMAGE_SIZE, chunk_size=16))
    return torch.cat(images)


def uint8_images(images: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select model-range images and store them compactly on the 0--255 scale."""
    selected = images[indices].cpu().clamp(-1.0, 1.0)
    return ((selected + 1.0) * 127.5).round().to(torch.uint8)


def build_cache() -> None:
    """Reproduce both headline flows and retain the top-128 decoded pairs."""
    if not torch.cuda.is_available():
        raise RuntimeError("--rebuild-cache requires the local CUDA execution context")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda")
    print(f"device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

    vae = load_sd_vae(device)
    A, X, observed_indices, raw_latents = load_headline_latents(vae, device)
    latent_mean = raw_latents.mean(dim=0, keepdim=True)
    latent_std = raw_latents.std(dim=0, keepdim=True).clamp_min(1e-3)
    Z_std = (raw_latents - latent_mean) / latent_std
    del raw_latents

    cell = (A * 2 + X).long()
    gram = lambda rows, cols: (cell[rows][:, None] == cell[cols][None, :]).double()
    propensity = LogisticPropensityModel().fit(X.double().unsqueeze(1), A.double())
    e1 = propensity.predict(X.double().unsqueeze(1)).reshape(-1).double()
    cv_rows = torch.randperm(N_OBS, generator=torch.Generator().manual_seed(0))[:2_000]
    reg_lambda, _ = select_regularization(
        gram(cv_rows.numpy(), cv_rows.numpy()),
        Z_std[cv_rows, 0].double().cpu(), seed=0,
    )
    print(f"ridge lambda: {reg_lambda:g}", flush=True)

    cache = {
        "arms": {},
        "n_obs": N_OBS,
        "n_particles": N_PARTICLES,
        "keep": KEEP,
        "data_seed": 0,
        "pipeline_seed": SEED,
        "image_size": IMAGE_SIZE,
        "flow_iterations": FLOW_ITERATIONS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "n_landmarks": N_LANDMARKS,
        "ridge_lambda": float(reg_lambda),
        "ranking": "decoded start-to-final full-image RGB L2",
        "start_definition": "D(E(Y_initial))",
    }

    for arm in (0, 1):
        print(f"constructing DR target for arm {arm}", flush=True)
        w_hat = (A == arm).double() / (e1 if arm == 1 else 1.0 - e1)
        on_arm = (A == arm).double()
        gram_pi = lambda rows, cols: (
            (X[rows][:, None] == X[cols][None, :]).double()
            * on_arm[rows][:, None]
        )
        v_dr = cpme_dr_weights_projected_nystrom(
            gram, gram_pi, w_hat, reg_lambda, n_landmarks=N_LANDMARKS,
            generator=torch.Generator().manual_seed(SEED),
        )

        arm_rows = torch.nonzero(A == arm).ravel()
        local_pick = torch.randperm(
            len(arm_rows), generator=torch.Generator().manual_seed(SEED)
        )[:N_PARTICLES]
        particle_rows = arm_rows[local_pick]
        initial = Z_std[particle_rows].clone()
        torch.manual_seed(SEED + 1)
        print(f"running arm {arm} flow ({FLOW_ITERATIONS} iterations)", flush=True)
        final, history = adamw_riesz_flow(
            initial.clone(), Z_std, target_weights=v_dr.float().to(device),
            sliced=False, n_iterations=FLOW_ITERATIONS,
            learning_rate=LEARNING_RATE, warmup_steps=WARMUP_STEPS,
            lr_schedule="cosine", log_every=500, snapshot_every=None,
            verbose=True,
        )
        print(
            f"arm {arm}: loss {history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}; "
            f"mean transport {(final - initial).norm(dim=1).mean().item():.2f}",
            flush=True,
        )

        print(f"decoding arm {arm} initial and final particles", flush=True)
        decoded_start = decode_all(vae, initial, latent_mean, latent_std)
        decoded_final = decode_all(vae, final, latent_mean, latent_std)
        change = (decoded_final - decoded_start).flatten(1).norm(dim=1)
        ranked_particles = torch.argsort(change, descending=True)[:KEEP].cpu()
        ranked_rows = particle_rows[ranked_particles].cpu()

        cache["arms"][arm] = {
            "start": uint8_images(decoded_start, ranked_particles),
            "end": uint8_images(decoded_final, ranked_particles),
            "change": change[ranked_particles].cpu(),
            "particle_index": ranked_particles,
            "logged_row": ranked_rows,
            "pool_record_index": observed_indices[ranked_rows],
            "initial_X": X[ranked_rows].cpu(),
            "change_all_median": float(change.median()),
            "change_all_max": float(change.max()),
        }
        del decoded_start, decoded_final, change, final, initial, v_dr
        torch.cuda.empty_cache()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE)
    print(f"saved top-128 cache: {CACHE}", flush=True)
    if SOURCE_TOP64.exists():
        validate_against_top64(cache)
    else:
        print(f"prior top-64 cache not found; skipped validation: {SOURCE_TOP64}",
              flush=True)


def validate_against_top64(cache: dict) -> None:
    """Report rankwise agreement with the previously saved top-64 cache."""
    old = torch.load(SOURCE_TOP64, map_location="cpu", weights_only=False)
    for arm in (0, 1):
        old_block = old["arms"][arm]
        new_block = cache["arms"][arm]
        start_rmse = (
            new_block["start"][:64].float() - old_block["start"].float()
        ).square().mean(dim=(1, 2, 3)).sqrt()
        end_rmse = (
            new_block["end"][:64].float() - old_block["end"].float()
        ).square().mean(dim=(1, 2, 3)).sqrt()
        change_error = (new_block["change"][:64] - old_block["change"]).abs()
        print(
            f"arm {arm} versus old top-64 (same rank): "
            f"start RMSE mean/max {start_rmse.mean():.4f}/{start_rmse.max():.4f}; "
            f"end {end_rmse.mean():.4f}/{end_rmse.max():.4f}; "
            f"change max abs {change_error.max():.4f}",
            flush=True,
        )


def image_metrics(start: torch.Tensor, end: torch.Tensor) -> dict:
    """Per-particle summaries on the uint8 RGB scale."""
    delta = end.float() - start.float()
    rmse = delta.square().mean(dim=(1, 2, 3)).sqrt()
    luminance_delta = (
        0.2126 * delta[:, 0] + 0.7152 * delta[:, 1] + 0.0722 * delta[:, 2]
    ).mean(dim=(1, 2))
    return {"rmse": rmse, "luminance_delta": luminance_delta}


def make_page(cache: dict, arm: int, first: int, vmax: float):
    """Make one readable page containing 16 ranked start/final/difference triples."""
    block = cache["arms"][arm]
    count = min(PER_PAGE, KEEP - first)
    groups = math.ceil(count / N_COLUMNS)
    height_ratios = []
    for group in range(groups):
        height_ratios.extend([1.0, 1.0, 1.08])
        if group < groups - 1:
            height_ratios.append(0.24)
    fig, axes = plt.subplots(
        len(height_ratios), N_COLUMNS, figsize=(13.0, 4.35 * groups),
        gridspec_kw={
            "height_ratios": height_ratios, "hspace": 0.08, "wspace": 0.025,
        },
        squeeze=False,
    )
    metrics = image_metrics(block["start"], block["end"])
    difference_handle = None

    for slot in range(groups * N_COLUMNS):
        group, column = divmod(slot, N_COLUMNS)
        row0 = 4 * group
        rows = (row0, row0 + 1, row0 + 2)
        rank_index = first + slot
        if slot >= count:
            for row in rows:
                axes[row, column].axis("off")
            continue

        start = block["start"][rank_index].permute(1, 2, 0).numpy()
        end = block["end"][rank_index].permute(1, 2, 0).numpy()
        difference = (
            block["end"][rank_index].float() - block["start"][rank_index].float()
        ).abs().mean(dim=0).numpy()
        axes[rows[0], column].imshow(start)
        axes[rows[1], column].imshow(end)
        difference_handle = axes[rows[2], column].imshow(
            difference, cmap="magma", vmin=0.0, vmax=vmax
        )
        axes[rows[0], column].set_title(
            f"rank {rank_index + 1} · X={int(block['initial_X'][rank_index])}\n"
            f"RMSE {float(metrics['rmse'][rank_index]):.1f}",
            fontsize=7,
        )
        axes[rows[2], column].text(
            0.03, 0.04,
            f"$\\Delta L$ {float(metrics['luminance_delta'][rank_index]):+.1f}",
            transform=axes[rows[2], column].transAxes, fontsize=7, color="white",
            ha="left", va="bottom",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"},
        )
        if column == 0:
            for row, label in zip(rows, ("start", "final", r"$|\Delta|$")):
                axes[row, column].set_ylabel(
                    f"$a={arm}$\n{label}", rotation=0, ha="right", va="center",
                    labelpad=10, fontsize=9,
                )

    for group in range(groups - 1):
        spacer_row = 4 * group + 3
        for column in range(N_COLUMNS):
            axes[spacer_row, column].axis("off")

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    last = first + count
    fig.suptitle(
        f"CelebA headline flow: decoded pixel-change ranks {first + 1}\u2013{last}, arm $a={arm}$",
        fontsize=13, y=0.985,
    )
    fig.text(
        0.5, 0.008,
        "$X$ is the logged particle's Blond_Hair label. Ranks use full-image RGB L2; "
        "RMSE and $\\Delta L$ use the 0--255 scale. All difference maps share one scale.",
        ha="center", va="bottom", fontsize=8,
    )
    fig.subplots_adjust(left=0.07, right=0.92, top=0.91, bottom=0.07)
    color_axis = fig.add_axes([0.935, 0.14, 0.012, 0.72])
    colorbar = fig.colorbar(difference_handle, cax=color_axis)
    colorbar.set_label("mean absolute RGB change (0--255)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    return fig


def write_metrics(cache: dict) -> None:
    """Save the rank, provenance, and basic pixel summaries for all displayed pairs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "celeba_ranked_pixel_change_top128_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "arm", "rank", "particle_index", "logged_row", "pool_record_index",
            "initial_blond_label", "cached_l2_minus1_plus1", "rmse_uint8",
            "mean_luminance_delta_uint8",
        ])
        for arm in (0, 1):
            block = cache["arms"][arm]
            metrics = image_metrics(block["start"], block["end"])
            for rank in range(KEEP):
                writer.writerow([
                    arm, rank + 1, int(block["particle_index"][rank]),
                    int(block["logged_row"][rank]),
                    int(block["pool_record_index"][rank]),
                    int(block["initial_X"][rank]), float(block["change"][rank]),
                    float(metrics["rmse"][rank]),
                    float(metrics["luminance_delta"][rank]),
                ])
    print(f"metrics: {path}", flush=True)


def plot_cache() -> None:
    """Write per-arm multipage PDFs and individual page PNGs."""
    if not CACHE.exists():
        raise FileNotFoundError(f"missing {CACHE}; run with --rebuild-cache")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    differences = []
    for arm in (0, 1):
        block = cache["arms"][arm]
        differences.append(
            (block["end"].float() - block["start"].float()).abs().mean(dim=1)
        )
    vmax = float(torch.quantile(torch.cat(differences).flatten(), 0.99))
    print(f"shared difference-map 99th percentile: {vmax:.2f}", flush=True)

    combined_pdf_path = OUT_DIR / "celeba_ranked_pixel_change_top128_both_arms.pdf"
    with PdfPages(combined_pdf_path) as combined_pdf:
        for arm in (0, 1):
            arm_pdf_path = OUT_DIR / f"celeba_ranked_pixel_change_top128_a{arm}.pdf"
            with PdfPages(arm_pdf_path) as arm_pdf:
                for first in range(0, KEEP, PER_PAGE):
                    fig = make_page(cache, arm, first, vmax)
                    arm_pdf.savefig(fig, bbox_inches="tight")
                    combined_pdf.savefig(fig, bbox_inches="tight")
                    last = min(KEEP, first + PER_PAGE)
                    png = OUT_DIR / (
                        f"celeba_ranked_pixel_change_a{arm}_"
                        f"r{first + 1:03d}-{last:03d}.png"
                    )
                    fig.savefig(png, dpi=220, bbox_inches="tight")
                    plt.close(fig)
            print(f"arm {arm} PDF: {arm_pdf_path}", flush=True)
    print(f"combined PDF: {combined_pdf_path}", flush=True)
    write_metrics(cache)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="rerun the two exact headline flows before plotting",
    )
    parser.add_argument(
        "--celeba-root", type=Path,
        help="CelebA root containing img_align_celeba and the attribute files; "
             "required only with --rebuild-cache",
    )
    parser.add_argument("--source-dgp", type=Path, default=SOURCE_DGP,
                        help="saved seed-0 DGP metadata containing obs_indices")
    parser.add_argument("--source-top64", type=Path, default=SOURCE_TOP64,
                        help="optional prior top-64 cache used for validation")
    parser.add_argument("--cache", type=Path, default=CACHE,
                        help="top-128 decoded-pair cache")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="directory for PDFs and PNG pages")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="directory for the metrics CSV")
    return parser.parse_args()


def main() -> None:
    global CELEBA_ROOT, SOURCE_DGP, SOURCE_TOP64, CACHE, CACHE_DIR, OUT_DIR, RESULTS_DIR
    args = parse_args()
    SOURCE_DGP = args.source_dgp.expanduser()
    SOURCE_TOP64 = args.source_top64.expanduser()
    CACHE = args.cache.expanduser()
    CACHE_DIR = CACHE.parent
    OUT_DIR = args.out_dir.expanduser()
    RESULTS_DIR = args.results_dir.expanduser()
    if args.rebuild_cache:
        if args.celeba_root is None:
            raise ValueError("--celeba-root is required with --rebuild-cache")
        CELEBA_ROOT = args.celeba_root.expanduser()
        build_cache()
    plot_cache()


if __name__ == "__main__":
    main()
