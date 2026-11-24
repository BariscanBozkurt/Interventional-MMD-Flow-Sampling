# ColorMNIST across 20 data seeds

We ran the headline DR--IMMD configuration on data seeds 0–19 in two campaigns.
The rebuttal uses the repetition with the [revised Table 1 encoders](#repetition-with-the-revised-table-1-encoders).
The earlier campaign retrained all 20 encoders.

## Fixed protocol

| Setting | Value |
| --- | --- |
| Logged observations | 20,000 per seed |
| Arms | 0 and 1 |
| Target | Full-sample projected Nyström DR; 1,024 landmarks; CV ridge |
| Initialization | 2,000 observed target-arm particles |
| Flow | IMQ--Euler, 3,000 updates |
| Metrics | Pixel SW2, legacy FID, Clean-FID |
| Replication | One pipeline run per data seed |

The two arms are averaged within each seed. Means, sample standard deviations,
and Student-t intervals are then computed over seed averages.

## Earlier repetition with freshly trained encoders

| Runs | Pixel SW2 | Legacy FID | Clean-FID |
| --- | ---: | ---: | ---: |
| Original seeds 0--2 | 0.009304 +/- 0.000295 | 30.644 +/- 0.809 | 35.146 +/- 0.641 |
| Rerun seeds 0--2 | 0.009510 +/- 0.000106 | 30.790 +/- 0.932 | 35.258 +/- 0.451 |
| Rerun seeds 0--19 | 0.009595 +/- 0.000499 | 30.345 +/- 1.847 | 34.504 +/- 2.082 |

The 95% Student-t interval for the 20-seed mean pixel SW2 is
`[0.009362, 0.009829]`. The reruns may differ slightly across accelerator and
software environments; they are practical reproduction checks rather than
bitwise replays.

## Reproduction

`run_twenty_seeds.py` freezes the simulation source and dependencies before a
run. `summarize_results.py` checks seed/arm coverage and performs the declared
aggregation.

```bash
python AdditonalExperiments/ColorMNIST_20_Seeds/run_twenty_seeds.py \
  --prepare --run-root runs/headline20 --mnist-root data

for seed in $(seq 0 19); do
  python AdditonalExperiments/ColorMNIST_20_Seeds/run_twenty_seeds.py \
    --seed "$seed" --run-root runs/headline20
done

python AdditonalExperiments/ColorMNIST_20_Seeds/summarize_results.py \
  --results-dir runs/headline20/results \
  --baseline-dir runs/headline20/baseline \
  --out-dir runs/headline20/analysis --require-complete
```

Generated runs, frozen snapshots, logs, and CSV summaries are ignored by Git.

## Repetition with the revised Table 1 encoders

The revised-paper experiment uses 20 L40S runs with the same sample sizes and
flow settings as above. Seeds 0–2 reload their individual revised Table 1
checkpoints. Each arm reuses its recorded IMQ scale and checks the fitted
ridge, starting particles, evaluation indices, latent support, oracle images,
pixel and latent SW2, and paper-compatible FID against the archived run.
Seeds 3–19 each train an encoder on that seed’s 20,000 logged images. The
runner saves these encoders so that later experiments can reuse them.

| Runs | Pixel SW2 | Paper-compatible FID | Clean-FID |
| --- | ---: | ---: | ---: |
| Revised Table 1, seeds 0–2 | 0.009292 ± 0.000351 | 30.650 ± 0.788 | — |
| Rerun, seeds 0–2 | 0.009292 ± 0.000351 | 30.650 ± 0.788 | 35.087 ± 0.781 |
| Rerun, seeds 0–19 | 0.009494 ± 0.000408 | 30.232 ± 1.848 | 34.471 ± 2.122 |

The 95% Student’s $t$ interval for the 20-seed mean pixel SW2 is
$[0.009303, 0.009685]$. Pixel and latent SW2 reproduced exactly for all six
original arms. The largest absolute per-arm difference in paper-compatible
FID was 0.00014. Revised Table 1 did not record Clean-FID.

From this directory, set `reference_dir` to the folder containing the revised
`ae_n20000_seed{0,1000,2000}.pt` checkpoints and the corresponding
`quality_n20000_seed{0,1,2}.csv` and `_particles.npz` files. Each seed can
run independently. The summary checks all 40 arm rows before averaging.

```bash
for seed in $(seq 0 19); do
  python run_matched_twenty.py --seed "$seed" \
    --results-dir runs/revised20 --reference-dir "$reference_dir"
done
python summarize_matched_twenty.py \
  --results-dir runs/revised20 --reference-dir "$reference_dir"
```
