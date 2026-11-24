"""Count-matched INITIAL-particle metrics for CelebA, without re-running the flow.

WHY THIS EXISTS. The CelebA simulation's pixel rows are ("observed", "roundtrip", "flow"):
its baseline is the whole observed arm, 2000 images, whatever the particle count M is. The
flow row has M images. FID and sliced Wasserstein are both strongly biased upward at small
sample size, so comparing a 250-image flow row against a 2000-image observed row charges
the sampler for the count. ColorMNIST and dSprites do not have this problem -- their
`initial` row IS the M decoded initial particles -- so only CelebA needs this.

WHAT IT DOES. The simulation saves the chosen particle rows in `<csv stem>_particles.npz`
(`pick = arm_indices[randperm(...)]`, i.e. rows of the full logged sample restricted to the
arm). This script replays only the cheap half of the pipeline: load the cached dataset,
take those rows, pass them through the SAME frozen SD-VAE, and score the decoded result
against the SAME held-out references. The flow -- the expensive part, ~350 s per arm -- is
never run.

The decode matters: the flow row is decoded, so the baseline must be too, or the comparison
measures the autoencoder rather than the transport.
"""
import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from autoencoder_utils import decode_latents, encode_images, load_sd_vae
from data_utils import load_celeba_paper
from metric_utils import CleanFIDScorer, LegacyFIDScorer, sliced_wasserstein


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_seed", type=int, required=True)
    p.add_argument("--n_obs", type=int, default=20000)
    p.add_argument("--n_ref", type=int, default=2000)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--particles", type=int, nargs="+", default=[250, 500, 1000, 2000])
    p.add_argument("--cache_dir", type=str, default=None,
                   help="where the ~20 GB dataset cache lives. Use node-local scratch; the "
                        "home quota cannot hold it.")
    p.add_argument("--celeba_root", type=str,
                   default="/nfs/ghome/live/danceh/data/celeba")
    p.add_argument("--results", type=str, default="results",
                   help="directory holding the *_particles.npz index files")
    p.add_argument("--out_dir", type=str, default="results_initial")
    p.add_argument("--fid_workers", type=int, default=8)
    return p.parse_args()


def particle_indices(results_dir, n_obs, m, seed, arm):
    """The rows the simulation actually flowed, for this (M, seed, arm).

    Matched on n_obs and M in the file name and on the `ds{seed}_r0_a{arm}` key inside, so a
    file from a different cell cannot be picked up silently.
    """
    pattern = os.path.join(results_dir, f"*nobs{n_obs}_M{m}_*_ds{seed}_*_particles.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no particle file matching {pattern}")
    key = f"ds{seed}_r0_a{arm}"
    for path in files:
        with np.load(path) as blob:
            if key in blob.files:
                return np.asarray(blob[key]), os.path.basename(path)
    raise KeyError(f"{key} not in any of {[os.path.basename(f) for f in files]}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir or (Path(__file__).resolve().parents[3] / "data/celeba_dgp"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (f"celeba_paper_{args.image_size}_n{args.n_obs}"
                         f"_nref{args.n_ref}_seed{args.data_seed}.pt")

    t0 = time.time()
    data = load_celeba_paper(cache, root=args.celeba_root, seed=args.data_seed,
                             image_size=args.image_size, n_obs=args.n_obs,
                             n_ref=args.n_ref, verbose=False)
    A = data["A"].view(-1).long()
    Y = data["Y"]
    print(f"data ready in {time.time() - t0:.0f} s   cache {cache}", flush=True)

    vae = load_sd_vae(device)
    flat = lambda images: images.reshape(len(images), -1)
    rgb = lambda images: images.reshape(-1, 3, args.image_size, args.image_size)
    scorers = {"legacy": LegacyFIDScorer(device, batch_size=32),
               "clean": CleanFIDScorer(device, batch_size=32,
                                       num_workers=args.fid_workers, allow_download=False)}

    rows = []
    for arm in (0, 1):
        reference = data[f"Y{arm}_ref"]
        ref_features = {name: s.features(rgb(reference), image_range="minus_one_one")
                        for name, s in scorers.items()}
        ref_flat = flat(reference).double().to(device)
        arm_rows = torch.nonzero(A == arm).ravel()

        for m in args.particles:
            try:
                idx, source = particle_indices(args.results, args.n_obs, m, args.data_seed, arm)
            except (FileNotFoundError, KeyError) as error:
                print(f"  arm {arm} M={m}: SKIPPED ({error})", flush=True)
                continue
            # the saved rows must be rows of THIS arm, or the npz belongs to another cell
            assert set(idx.tolist()) <= set(arm_rows.tolist()), \
                f"particle rows for arm {arm} are not all in that arm (file {source})"

            t = time.time()
            picked = Y[torch.as_tensor(idx, dtype=torch.long)]
            latents = encode_images(vae, picked)
            decoded = decode_latents(vae, latents, args.image_size)
            entry = {"data_seed": args.data_seed, "n_obs": args.n_obs, "arm": arm,
                     "n_particles": m, "n_initial": len(idx), "n_ref": len(reference),
                     "source": source, "seconds": round(time.time() - t, 1)}
            entry["sw2_initial"] = float(sliced_wasserstein(
                flat(decoded).double().to(device), ref_flat, p=2, seed=0))
            entry["sw1_initial"] = float(sliced_wasserstein(
                flat(decoded).double().to(device), ref_flat, p=1, seed=0))
            for name, scorer in scorers.items():
                entry[f"fid_{name}_initial"] = scorer.distance_from_features(
                    scorer.features(rgb(decoded), image_range="minus_one_one"),
                    ref_features[name])
            rows.append(entry)
            print(f"  arm {arm} M={m:5d}: SW2 {entry['sw2_initial']:.5f}  "
                  f"FID legacy {entry['fid_legacy_initial']:.3f}  "
                  f"clean {entry['fid_clean_initial']:.3f}  ({entry['seconds']:.0f} s)",
                  flush=True)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"celeba_initial_nobs{args.n_obs}_ds{args.data_seed}.csv"
    import csv
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"done: {len(rows)} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
