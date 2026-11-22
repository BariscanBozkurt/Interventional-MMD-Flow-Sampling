"""Matched ColorMNIST kernel-herding baseline.

This script reuses the current ColorMNIST DR-IMMD simulation for the data,
autoencoder, projected-Nystrom DR embedding, oracle samples, metrics, and timing
accounting. It replaces only the sampler: instead of moving latent particles,
it greedily selects points from the finite set of all logged latent outcomes.

The candidate set is deliberately the whole DR support, not the observations in
the target treatment arm. This is the selection-sampler protocol used by the
existing kernel-herding experiments: the DR embedding, rather than an oracle arm
filter, must make the selected empirical measure resemble the intervention.
Every primary returned image is therefore an unmodified logged image. For a
like-for-like decoder diagnostic, the script also scores the same selection
after the autoencoder round trip ``D(E(Y_selected))``. The CSV records the
selected target-arm fraction, number of unique observations, and maximum
multiplicity so that this restriction can be inspected after the SW2/FID
comparison.

The familiar ``*_flow`` columns are retained for compatibility with the existing
analysis code; in this script they contain the raw kernel-herding output metrics.
Explicit ``*_immd`` columns contain a matched IMMD flow run inside the same
process. Thus the autoencoder, target, initial cloud, references, and evaluation
randomness are identical rather than merely matched by seed.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEPLOY_ROOT = HERE.parents[1]
DR_SIM_DIR = DEPLOY_ROOT / "Simulations" / "ColorMNIST" / "dr_mmd_flow"
sys.path.insert(0, str(DR_SIM_DIR))

import colormnist_dr_mmd_flow_simulation as base

from autoencoder_utils import encode_in_batches
from causality_utils import herding_diagnostics, kernel_herding
from data_utils import color_ratio, sample_interventional, to_image_range, to_model_range
from kernel_utils import IMQKernel, RBF, imq_c_from_median
from metric_utils import sliced_wasserstein


EXTRA_COLUMNS = [
    "sampler", "candidate_set", "output_source", "n_candidates",
    "n_herding_steps", "n_unique", "unique_fraction", "max_multiplicity",
    "candidate_coverage", "selected_target_arm_fraction", "seconds_herding",
    "latent_sw2_herding", "colour_mean_herding", "colour_std_herding",
    "colour_w2_herding", "sw1_herding", "sw2_herding", "mmd2_herding",
    "fid_legacy_herding", "fid_clean_herding",
    "colour_mean_herding_decoded", "colour_std_herding_decoded",
    "colour_w2_herding_decoded", "sw1_herding_decoded",
    "sw2_herding_decoded", "mmd2_herding_decoded",
    "fid_legacy_herding_decoded", "fid_clean_herding_decoded",
    "seconds_herding_decode",
    "objective_initial_immd", "objective_final_immd", "latent_sw2_immd",
    "colour_mean_immd", "colour_std_immd", "colour_w2_immd",
    "sw1_immd", "sw2_immd", "mmd2_immd",
    "fid_legacy_immd", "fid_clean_immd",
    "seconds_immd_flow", "seconds_immd_decode", "seconds_immd_eval",
    "seconds_immd_fid",
]

HERDING_SELECTIONS = {}


def target_embedding_norm(kernel, support: torch.Tensor,
                          weights: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """Compute the squared RKHS norm of a weighted empirical embedding."""
    total = torch.zeros((), dtype=support.dtype, device=support.device)
    for start in range(0, len(support), chunk_size):
        stop = min(start + chunk_size, len(support))
        total = total + weights[start:stop] @ (
            kernel(support[start:stop], support) @ weights)
    return total


def embedding_objective(kernel, sample: torch.Tensor, support: torch.Tensor,
                        weights: torch.Tensor, target_norm: torch.Tensor,
                        chunk_size: int = 2048) -> torch.Tensor:
    """Squared RKHS distance from an equal-weight sample to a weighted embedding."""
    cross = torch.zeros(len(sample), dtype=sample.dtype, device=sample.device)
    for start in range(0, len(support), chunk_size):
        stop = min(start + chunk_size, len(support))
        cross = cross + kernel(sample, support[start:stop]) @ weights[start:stop]
    return kernel(sample, sample).mean() - 2.0 * cross.mean() + target_norm


def run_arm(target_a: int, ae, pools: list, data: torch.Tensor,
            causal_idx: torch.Tensor, A_obs: torch.Tensor, A_s: torch.Tensor,
            Z_s: torch.Tensor, e1: torch.Tensor, K_x, K_a, gram, gram_x, solver: str,
            fold_ids: np.ndarray, fold_lambdas: dict, args,
            seed: int, device: torch.device, fid_scorers: dict = None,
            nosplit_lambda: float = None) -> tuple:
    """Construct one arm's DR target, herd logged outcomes, and score the selection."""
    # Importance ratios and the fixed DR target are exactly the same objects used by
    # the matched DR-IMMD run.
    t0 = time.time()
    e_target = e1 if target_a == 1 else 1.0 - e1
    w_hat = ((A_s == target_a).double() / e_target).to(device)
    on_arm = w_hat[(A_s == target_a).to(device)]
    ess = float(on_arm.sum() ** 2 / (on_arm ** 2).sum())
    base.sync_device(device)
    seconds_ratio = time.time() - t0

    t0 = time.time()
    vectors = base.target_weight_vectors(
        target_a, A_s, w_hat, K_x, K_a, gram, gram_x, solver,
        fold_ids, fold_lambdas, args, seed, device, nosplit_lambda, ["dr"])
    v_target = vectors["dr"]
    base.sync_device(device)
    seconds_cme = time.time() - t0

    # The matched initial distribution remains the logged target arm, as in the
    # DR-IMMD headline run. Herding itself sees all logged outcomes as candidates.
    t0 = time.time()
    arm_causal = causal_idx[A_obs[causal_idx] == target_a]
    Z_arm = encode_in_batches(ae.encode, data[arm_causal])
    base.sync_device(device)
    seconds_ae = time.time() - t0

    t0 = time.time()
    n_particles = min(args.n_particles, Z_arm.shape[0])
    pick = torch.randperm(
        Z_arm.shape[0], generator=base.particle_generator(args, seed, 1)
    )[:n_particles].to(device)
    initial_particles = Z_arm[pick].clone()

    c_imq = imq_c_from_median(Z_s)
    kernel = IMQKernel(c=c_imq, beta=-0.5)

    eval_idx = torch.arange(n_particles, device=device)
    if args.eval_particles is not None and n_particles > args.eval_particles:
        eval_idx = torch.randperm(
            n_particles,
            generator=torch.Generator(device=device).manual_seed(seed + 11),
            device=device,
        )[:args.eval_particles]
    with torch.no_grad():
        decoded_initial = to_image_range(ae.decode(initial_particles[eval_idx]))

    _, Y_oracle = sample_interventional(
        pools, args.n_ref, a=target_a, tau=args.tau, alpha=args.alpha,
        k_mask=args.k_mask,
        generator=torch.Generator(device=device).manual_seed(seed + 9))
    _, Y_oracle_2 = sample_interventional(
        pools, args.n_ref, a=target_a, tau=args.tau, alpha=args.alpha,
        k_mask=args.k_mask,
        generator=torch.Generator(device=device).manual_seed(seed + 10))
    with torch.no_grad():
        decoded_oracle = to_image_range(ae.decode(encode_in_batches(
            ae.encode, to_model_range(Y_oracle_2))))

    kernel_pixel = RBF(use_length_scale_heuristic=True)
    _ = kernel_pixel(Y_oracle.reshape(len(Y_oracle), -1))
    with torch.no_grad():
        Z_oracle = encode_in_batches(ae.encode, to_model_range(Y_oracle))
        Z_oracle_2 = encode_in_batches(ae.encode, to_model_range(Y_oracle_2))

    k_pixel = min(len(eval_idx), len(Y_oracle))
    k_latent = min(n_particles, len(Z_oracle))
    latent_generator = torch.Generator().manual_seed(seed + 13)
    latent_floor_matched = sliced_wasserstein(
        base.subsample_rows(Z_oracle_2, k_latent, latent_generator),
        base.subsample_rows(Z_oracle, k_latent, latent_generator),
        p=2, seed=0).item()

    ratio_oracle = color_ratio(Y_oracle)
    shared = {
        "target_a": target_a,
        "n_arm_causal": int(len(arm_causal)),
        "n_particles_requested": args.n_particles,
        "n_particles": int(n_particles),
        "n_flow_eval": int(len(eval_idx)),
        "n_ref": int(len(Y_oracle)),
        "ess_on_arm": ess,
        "n_on_arm": int(len(on_arm)),
        "max_weight": float(w_hat.max()),
        "c_imq": float(c_imq),
        "step_size": float("nan"),
        "latent_sw2_initial": sliced_wasserstein(
            initial_particles, Z_oracle, p=2, seed=0).item(),
        "latent_sw2_floor": sliced_wasserstein(
            Z_oracle_2, Z_oracle, p=2, seed=0).item(),
        "latent_sw2_floor_matched": latent_floor_matched,
        "n_floor_matched": int(k_pixel),
        "n_floor_matched_latent": int(k_latent),
        "colour_mean_oracle": float(ratio_oracle.mean()),
        "colour_std_oracle": float(ratio_oracle.std()),
    }
    for name, images in (("initial", decoded_initial), ("floor", Y_oracle_2),
                         ("roundtrip", decoded_oracle)):
        metrics = base.evaluate_images(images, Y_oracle, kernel_pixel)
        shared.update({f"{key}_{name}": value for key, value in metrics.items()})
    matched = base.matched_floor(
        Y_oracle_2, Y_oracle, kernel_pixel, k_pixel, seed + 12)
    shared.update({f"{key}_floor_matched": value for key, value in matched.items()})
    base.sync_device(device)
    seconds_eval_shared = time.time() - t0

    t0 = time.time()
    oracle_features = {}
    for impl, scorer in (fid_scorers or {}).items():
        oracle_features[impl] = scorer.features(Y_oracle, image_range="zero_one")
        for name, images in (("initial", decoded_initial), ("floor", Y_oracle_2),
                             ("roundtrip", decoded_oracle)):
            shared[f"fid_{impl}_{name}"] = scorer.distance_from_features(
                scorer.features(images, image_range="zero_one"),
                oracle_features[impl])
    base.sync_device(device)
    seconds_fid_shared = time.time() - t0

    # Matched IMMD reference. Running it here, before herding, ensures that both
    # samplers see the same trained AE, DR coefficients, initial particles, oracle
    # draws, and FID reference features. A separately launched CUDA AE is not an
    # exact pairing even when every nominal seed agrees.
    immd = base.flow_and_score(
        "dr", v_target, initial_particles.clone(), Z_s, kernel,
        0.15 * c_imq ** 2, eval_idx, ae, Y_oracle, Z_oracle,
        kernel_pixel, fid_scorers, oracle_features, args)

    # Selection sampler: support and candidates are both the full logged latent set.
    # The result is mapped back to the corresponding raw logged images. Those images
    # remain the primary selection baseline; an encode-decode view is scored separately
    # below so its representation cost is not mistaken for herding error.
    t0 = time.time()
    with torch.no_grad():
        herded_latents, selected = kernel_herding(
            v_target, kernel, Z_s, Z_s, n_particles, return_indices=True)
    base.sync_device(device)
    seconds_herding = time.time() - t0

    selected_observations = causal_idx[selected]
    herded_images = to_image_range(data[selected_observations])
    diagnostics = herding_diagnostics(selected, n_candidates=len(Z_s))
    selected_arm_fraction = float((A_s[selected] == target_a).float().mean())

    t_decode = time.time()
    with torch.no_grad():
        decoded_herded = to_image_range(ae.decode(herded_latents[eval_idx]))
    base.sync_device(device)
    seconds_herding_decode = time.time() - t_decode

    selection_key = f"ds{seed // 1000}_r{seed % 1000}_a{target_a}"
    HERDING_SELECTIONS[f"{selection_key}_candidate_positions"] = selected.cpu().numpy()
    HERDING_SELECTIONS[f"{selection_key}_observation_indices"] = (
        selected_observations.cpu().numpy())

    # This row's own evaluation. The expensive target norm is computed once and
    # shared by the initial and final RKHS objectives.
    t0 = time.time()
    with torch.no_grad():
        target_norm = target_embedding_norm(kernel, Z_s, v_target)
        objective_initial = embedding_objective(
            kernel, initial_particles, Z_s, v_target, target_norm)
        objective_final = embedding_objective(
            kernel, herded_latents, Z_s, v_target, target_norm)

    latent_sw2_herding = sliced_wasserstein(
        herded_latents, Z_oracle, p=2, seed=0).item()
    herding_metrics = base.evaluate_images(
        herded_images[eval_idx], Y_oracle, kernel_pixel)
    decoded_herding_metrics = base.evaluate_images(
        decoded_herded, Y_oracle, kernel_pixel)
    row = {
        "estimator": "dr",
        "sampler": "kernel_herding",
        "candidate_set": "all_logged_outcomes",
        "output_source": "raw_logged_outcome_and_ae_roundtrip",
        "n_candidates": int(len(Z_s)),
        "n_herding_steps": int(n_particles),
        "n_unique": diagnostics["n_unique"],
        "unique_fraction": diagnostics["unique_fraction"],
        "max_multiplicity": diagnostics["max_multiplicity"],
        "candidate_coverage": diagnostics["candidate_coverage"],
        "selected_target_arm_fraction": selected_arm_fraction,
        "objective_initial": float(objective_initial),
        "objective_final": float(objective_final),
        "latent_sw2_flow": latent_sw2_herding,
        "latent_sw2_herding": latent_sw2_herding,
    }
    row.update(base.weight_diagnostics(v_target))
    row.update({f"{key}_flow": value for key, value in herding_metrics.items()})
    row.update({f"{key}_herding": value for key, value in herding_metrics.items()})
    row.update({f"{key}_herding_decoded": value
                for key, value in decoded_herding_metrics.items()})
    row.update({
        "objective_initial_immd": immd["objective_initial"],
        "objective_final_immd": immd["objective_final"],
        "latent_sw2_immd": immd["latent_sw2_flow"],
        "seconds_immd_flow": immd["seconds_flow"],
        "seconds_immd_decode": immd["seconds_decode"],
        "seconds_immd_eval": immd["seconds_eval"],
        "seconds_immd_fid": immd["seconds_fid"],
    })
    for metric in ("colour_mean", "colour_std", "colour_w2",
                   "sw1", "sw2", "mmd2"):
        row[f"{metric}_immd"] = immd[f"{metric}_flow"]
    t_fid = time.time()
    for impl, scorer in (fid_scorers or {}).items():
        fid_value = scorer.distance_from_features(
            scorer.features(herded_images[eval_idx], image_range="zero_one"),
            oracle_features[impl])
        row[f"fid_{impl}_flow"] = fid_value
        row[f"fid_{impl}_herding"] = fid_value
        row[f"fid_{impl}_herding_decoded"] = scorer.distance_from_features(
            scorer.features(decoded_herded, image_range="zero_one"),
            oracle_features[impl])
        row[f"fid_{impl}_immd"] = immd[f"fid_{impl}_flow"]
    seconds_fid = time.time() - t_fid
    base.sync_device(device)
    seconds_eval = time.time() - t0

    # seconds_flow is the compatibility name consumed by the shared timing code.
    # seconds_herding states what was actually timed; the two are identical.
    row.update({
        "seconds_flow": round(seconds_herding + immd["seconds_flow"], 1),
        "seconds_herding": round(seconds_herding, 4),
        "seconds_decode": round(seconds_herding_decode + immd["seconds_decode"], 1),
        "seconds_herding_decode": round(seconds_herding_decode, 4),
        "seconds_eval": round(seconds_eval + immd["seconds_eval"], 1),
        "seconds_fid": round(seconds_fid + immd["seconds_fid"], 1),
    })
    row = {**shared, **row}
    return [row], {
        "ae": seconds_ae,
        "ratio": seconds_ratio,
        "cme": seconds_cme,
        "eval_shared": seconds_eval_shared,
        "fid_shared": seconds_fid_shared,
    }


ORIGINAL_PARSE_ARGS = base.parse_args


def parse_args():
    """Use the shared CLI, with guards for the matched herding experiment."""
    args = ORIGINAL_PARSE_ARGS()
    if base.resolve_estimators(args) != ["dr"]:
        raise SystemExit("kernel-herding headline runs require --estimators dr")
    if args.init != "observed":
        raise SystemExit("the matched kernel-herding baseline requires --init observed")
    if args.n_dr is not None:
        raise SystemExit("leave --n_dr unset so candidate indices map to all causal rows")
    if args.snapshot_every is not None:
        raise SystemExit("--snapshot_every is an MMD-flow option and does not apply here")
    return args


def csv_path(args) -> Path:
    """Keep all herding artifacts separate from the DR-IMMD result directory."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    arms = "".join(str(arm) for arm in args.target_arms)
    name = (
        f"colormnist_kernel_herding_{args.tag}_nobs{args.n_obs}_M{args.n_particles}"
        f"_w{args.w_conf:g}_a{args.alpha:g}_k{args.k_mask:g}_tau{args.tau:g}"
        f"_arms{arms}_ds{seeds}_{args.dr_fit}_{args.solver}.csv"
    )
    return Path(args.out_dir) / name


def write_herding_indices(out_path: Path) -> None:
    """Save the candidate positions and original logged-row indices verbatim."""
    if HERDING_SELECTIONS:
        np.savez(out_path.with_name(out_path.stem + "_selections.npz"),
                 **HERDING_SELECTIONS)


def main() -> None:
    # The imported parser's default is beside its own source file. Supply the local
    # result directory only when the caller did not explicitly choose another one.
    if "--out_dir" not in sys.argv:
        sys.argv.extend(["--out_dir", str(HERE / "results" / "ColorMNIST")])

    base.COLUMNS = base.COLUMNS + EXTRA_COLUMNS
    base.parse_args = parse_args
    base.csv_path = csv_path
    base.run_arm = run_arm
    base.write_particle_indices = write_herding_indices
    base.main()


if __name__ == "__main__":
    main()
