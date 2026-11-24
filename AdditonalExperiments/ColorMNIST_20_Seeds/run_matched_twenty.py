"""Run one DR-only ColorMNIST seed, replaying the revised controls for seeds 0–2."""

import argparse
import csv
import hashlib
import math
import os
from contextlib import nullcontext
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Simulations/ColorMNIST/dr_mmd_flow"))
import colormnist_dr_mmd_flow_simulation as sim
from autoencoder_utils.colormnist_ae import ConvAE


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reference_arms(folder, seed):
    path = folder / f"quality_n20000_seed{seed}.csv"
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["solver"] == "projected_nystrom" and r["estimator"] == "dr"]
    arms = {int(r["target_a"]): r for r in rows}
    if len(rows) != 2 or set(arms) != {0, 1}:
        raise ValueError(f"Incomplete DR reference for seed {seed}")
    return arms


def run(seed, out, reference, mnist_root, fid_workers):
    out.mkdir(parents=True, exist_ok=True)
    result = out / f"seed_{seed}.csv"
    if result.exists():
        raise FileExistsError(result)
    archived = seed < 3
    refs = reference_arms(reference, seed) if archived else {}
    checkpoint = (reference / f"ae_n20000_seed{seed * 1000}.pt" if archived
                  else out / f"ae_seed{seed}.pt")

    flags = ["matched_twenty", "--n_obs", "20000", "--n_particles", "2000", "--n_ref", "2000",
             "--eval_particles", "2000", "--target_arms", "0", "1", "--data_seeds", str(seed),
             "--n_repeats", "1", "--estimators", "dr", "--init", "observed", "--ae_split", "none",
             "--dr_fit", "nosplit", "--solver", "projected_nystrom", "--n_landmarks", "1024",
             "--lambda_rule", "cv", "--w_conf", "10", "--tau", "0.08", "--alpha", "0.2",
             "--k_mask", "20", "--latent_dim", "32", "--ae_iterations", "3000",
             "--flow_iterations", "3000", "--fid", "both", "--fid_workers", str(fid_workers),
             "--mnist_root", str(mnist_root), "--tag", "revised_twenty"]
    with patch.object(sys, "argv", flags):
        config = sim.parse_args()
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda", torch.cuda.current_device())
    if "L40S" not in torch.cuda.get_device_name(device):
        raise ValueError("This repetition uses L40S for the same CUDA SW2 projections as revised Table 1")
    pools = sim.load_digit_pools(root=mnist_root, device=device)
    x, a, y = sim.sample_observed(
        pools, config.n_obs, w_conf=config.w_conf, tau=config.tau, alpha=config.alpha,
        k_mask=config.k_mask, generator=torch.Generator(device=device).manual_seed(seed))
    scorers = {"legacy": sim.LegacyFIDScorer(device=device, batch_size=64, distance_device=device),
               "clean": sim.CleanFIDScorer(device=device, batch_size=64, num_workers=fid_workers,
                                           allow_download=False)}
    native_train, native_weights = sim.train_conv_ae, sim.target_weight_vectors
    native_arm, native_flow = sim.run_arm, sim.flow_and_score

    def encoder(images, *, latent_dim, device, **settings):
        # An interrupted new seed can reuse its own completed checkpoint.
        if archived or checkpoint.exists():
            model = ConvAE(latent_dim=latent_dim).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            return model, []
        model, history = native_train(images, latent_dim=latent_dim, device=device, **settings)
        with NamedTemporaryFile(dir=out) as tmp:
            torch.save(model.state_dict(), tmp)
            tmp.flush()
            os.link(tmp.name, checkpoint)
        return model, history

    def weights(*args):
        # Table 1 constructs DR as the sum of components of one projected solve.
        vectors = native_weights(*args[:-1], ("plugin", "ipw", "dr"))
        return {"dr": vectors["dr"]}

    def run_arm(arm, *args):
        ref = refs.get(arm)
        evidence = {"ridge_gram": args[-1]}
        def score(name, v, initial, support, kernel, step, idx, ae, oracle, *rest):
            tensors = {"initial": initial, "support": support, "oracle": oracle, "indices": idx}
            for key, tensor in tensors.items():
                field = key + "_sha256"
                evidence[field] = digest(tensor.detach().contiguous().cpu().numpy().tobytes())
                if ref and evidence[field] != ref[field]:
                    raise ValueError(f"Seed {seed}, arm {arm}: {key} differs from revised Table 1")
            selected = sim.PARTICLE_INDICES[f"ds{seed}_r0_a{arm}"]
            evidence["particle_indices_sha256"] = digest(selected.tobytes())
            if ref:
                with np.load(reference / f"quality_n20000_seed{seed}_particles.npz") as original:
                    if not np.array_equal(selected, original[f"ds{seed}_r0_a{arm}"]):
                        raise ValueError(f"Seed {seed}, arm {arm}: initial observation indices differ")
                if step != float(ref["step_size"]) or evidence["ridge_gram"] != float(ref["ridge_gram"]):
                    raise ValueError(f"Seed {seed}, arm {arm}: step or selected ridge differs")
            return native_flow(name, v, initial, support, kernel, step, idx, ae, oracle, *rest)

        scale = (patch.object(sim, "imq_c_from_median", return_value=float(ref["c_imq"]))
                 if ref else nullcontext())
        with scale, patch.object(sim, "flow_and_score", score):
            rows, times = native_arm(arm, *args)
        row, = rows
        row.update(evidence)
        if ref:
            for key, tol in (("sw2_flow", 1e-7), ("latent_sw2_flow", 1e-7), ("fid_legacy_flow", 1e-3)):
                if not math.isclose(row[key], float(ref[key]), rel_tol=0, abs_tol=tol):
                    raise ValueError(f"Seed {seed}, arm {arm}: {key} {row[key]} vs archived {ref[key]}")
        print(f"Seed {seed}, arm {arm}: SW2={row['sw2_flow']:.9f}, FID={row['fid_legacy_flow']:.6f}", flush=True)
        return rows, times

    with patch.object(sim, "train_conv_ae", encoder), patch.object(sim, "run_arm", run_arm), \
            patch.object(sim, "target_weight_vectors", weights):
        rows = sim.run_repeat(pools, sim.to_model_range(y), x, a, {}, config, seed, 0, device, scorers)
    checkpoint_hash = digest(checkpoint.read_bytes())
    for row in rows:
        row.update(encoder_sha256=checkpoint_hash, encoder_source="archived" if archived else "trained")
    with NamedTemporaryFile(dir=out, mode="w", newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        tmp.flush()
        os.link(tmp.name, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(20), required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--mnist-root", type=Path, default=REPO / "data")
    parser.add_argument("--fid-workers", type=int, default=4)
    args = parser.parse_args()
    run(args.seed, args.results_dir, args.reference_dir, args.mnist_root, args.fid_workers)
