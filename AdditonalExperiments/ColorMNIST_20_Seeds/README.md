# ColorMNIST across 20 data seeds

This study reruns the unchanged headline DR--IMMD configuration for data seeds
0--19. Seeds 0--2 are fresh reruns; seeds 3--19 add independent logged datasets.
The experiment changes neither the estimator nor the sampler.

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
