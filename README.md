# Sampling Interventional Distributions with Orthogonal MMD Flow

This repository contains the implementation and experiments for sampling
interventional outcome distributions from logged data with doubly robust kernel
mean embeddings and MMD particle flow.

The method first estimates a finite RKHS expansion of the policy-induced
outcome law,

```text
chi_hat(.) = sum_i v_i k(y_i, .),
```

and then holds this target fixed while moving particles to match its embedding.
For image-valued outcomes, the flow runs in a frozen autoencoder latent space
and the final particles are decoded once.

## Repository layout

```text
src/                    reusable kernels, causal estimators, flows and metrics
Simulations/            batch experiment drivers and table analysis
NotebookExamples/       worked examples and experiment notebooks
AdditonalExperiments/   concise rebuttal ablations and their source code
```

The historical directory spelling `AdditonalExperiments` is retained so that
existing scripts and links remain valid.

## Installation

Python 3.12 was used for the reported experiments. Install PyTorch for the
local CUDA version, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The two-dimensional benchmark runs on CPU. ColorMNIST, dSprites, and CelebA
are intended to run on a GPU.

## Data and generated artifacts

Datasets are not redistributed. MNIST can be obtained through TorchVision;
dSprites and CelebA must be downloaded under their original licenses and passed
to the relevant script through its command-line arguments.

Raw result CSVs, particle arrays, model states, decoded image grids, scheduler
logs, and site-specific launchers are intentionally ignored. They may contain
large arrays or machine provenance such as host names and scheduler IDs. Every
experiment script supports `--help` and writes its outputs to a user-selected
local directory.

## Starting points

- Run `python Simulations/DecFM_2D/dr_mmd_flow/decfm_2d_dr_flow_simulation.py --help`
  for the data-free two-dimensional benchmark.
- See `NotebookExamples/Interventional_MMD_Flow/` for the four main experiment
  pipelines.
- See [`AdditonalExperiments/README.md`](AdditonalExperiments/README.md) for the
  misspecification, small-batch RFF, 20-seed, kernel-herding, CelebA diagnostic,
  and IMQ--Euler ablations.

## License

The repository code is released under the [MIT License](LICENSE). Dataset and
third-party model licenses remain with their respective owners.
