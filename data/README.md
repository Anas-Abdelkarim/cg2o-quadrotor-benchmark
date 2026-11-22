# Data of the paper

Measurements of the paper, in the same layout as a new run folder `results/<run>/`. No script writes into
this folder: new generations go to `results/<run>/`, analysis outputs to `results/analysis/`. Raw PX4 logs
(`flight.ulg`, about 14 MB per flight) and videos are not included; `make flights` and `make video`
produce them in a new run.

| Folder | Content |
| --- | --- |
| `challenging_campaign/{warm,cold}_NN/` | one folder per flight: `challenging_trace.jsonl` (every problem with its SHA-256 hash and the in-flight solve), `challenging_evaluation.json` (ground truth: goal, airframe clearance, tilt), `truth.npz` (ground-truth trajectory), `solves.csv`, the settings used, `runtime/` (scenario), `replay/paired.json` and `replay/summary.json` (cg2o and IPOPT on every problem) |
| `challenging_campaign/` | `campaign_summary.json`; copies of `starts.json` and `ipopt_cold_default.json` as used (the inputs of new runs are in `src/px4_nmpc_config/config/experiments/challenging_course/`) |
| `tuning/20261110_*` | the four equal-budget tuning studies: `trials.csv`, `selected_parameters.json`, `report.txt`, `log.jsonl` |
| `sitl/challenging_lag_*` | the three flights whose problems the tuning studies replay |
| `challenging_mpc/replay_lag_*` | reference replays of those flights (per-problem reference costs for the tuning gates) |

Paths inside the files are relative to the repository root.
