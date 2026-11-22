# 3. Reproduce the results step by step

Everything in Sec. VII of the paper can be generated from scratch with this repository: installation of the
solvers and of PX4, the tuning, the 20 flights, the paired comparison, the tables, the figures and the video. The
measurements of the paper are also shipped in [`data/`](../data/README.md), so each step can instead start
from them.

The same steps also run through one entry point, `python3 experiments/run_experiment.py tables|replay|tuning|full`
([QUICKSTART.md](../QUICKSTART.md)), which writes the paper's tables and figures (`paper_table_*.txt`,
`paper_fig*_*.pdf`) into a new folder under `results/experiments/`. This page describes the `make` targets behind it.

## Where results go

| Folder | Content | Written by |
| --- | --- | --- |
| `data/` | the measurements of the paper | nothing: no script writes into it |
| `results/<run>/` | one new generation (default `results/run_01`), same layout as `data/` | the generation steps below |
| `results/analysis/<dataset>/` | tables (JSON) and figures made from `data/` or from a run | `make tables figures` |

Use a new run name for every new generation (`make ... RUN=results/run_02`); earlier runs are kept. Within
a run, a completed flight, replay or video is kept and skipped, so an interrupted step can be restarted; the
folder of an attempt that did not complete is renamed `<name>.incomplete_<time>` (never deleted) and redone.

Run every command from the repository root. Timed steps (tuning, flights, replay) must run one at a time.
The paper used CPUs 0–3 isolated from the scheduler (`isolcpus=0-3` on the kernel command line) and
`taskset -c 0-3`; the MPC node of the flights ran on CPUs 2 and 3. Other CPUs can be chosen with the Makefile
variables `CPUS` (tuning and replays, default `0-3`) and `CONTROLLER_CPUS` (flights, default `2,3`), e.g.
`make tune CPUS=0-1 CONTROLLER_CPUS=0,1`. Without isolated CPUs the ratios between the solvers reproduce,
the absolute times do not.

## A. Generate everything

### A1. Install

Ubuntu 22.04 with ROS 2 Humble and Gazebo; system packages in [INSTALL.md](../INSTALL.md) (profile B).

```bash
python3 -m venv --system-site-packages venv && venv/bin/pip install pip==26.2.2 setuptools==79.0.1
venv/bin/pip install -r requirements.txt
HSL_SOURCE=/path/to/coinhsl-2024.05.15.tar.gz make externals
                        # IPOPT 3.14.21, HSL, g2o, cg2o (pinned versions, -march=native as for the paper),
                        # PX4 v1.17 SITL and the Micro XRCE-DDS agent, all under install/externals
make build              # solver backends, benchmark extension (build/challenging_mpc), ROS 2 packages
source scripts/setup_env.sh
make test               # unit and integration tests
```

HSL is licensed separately (free for academic use, https://licences.stfc.ac.uk/product/coin-hsl); the
paper's IPOPT runs use MA57 (cold) and MA27 (warm).

### A2. Tuning flights

```bash
make tuning-flights RUN=results/run_01
```

Three PX4 SITL flights of cg2o with the untuned course settings (warm, warm with video, cold at every
problem) into `results/run_01/sitl/`, and their reference replays with both solvers at 250 iterations
into `results/run_01/challenging_mpc/`. The tuning replays these recorded problem sequences.

### A3. Equal-budget tuning

```bash
make tune RUN=results/run_01
```

Four Bayesian (TPE) studies: cg2o and IPOPT, warm (80 trials) and cold (150 trials), same gates,
objective and validation flight. Each writes `results/run_01/tuning/<stamp>_challenging_<solver>_<mode>/`
with `selected_parameters.json`, `trials.csv` and `report.txt`. The cold cg2o study is also the sensitivity
study.
A study in which no setting passes the gates writes no `selected_parameters.json`; the later steps use the
newest study of the run that selected a setting, or the settings given with `WARM_TUNED`/`COLD_TUNED`
(and `IPOPT_WARM` for the replay).

### A4. Flight campaign

```bash
make flights RUN=results/run_01
```

10 warm and 10 cold PX4 SITL flights with the newest tuned settings of the run and the randomized take-off
points and headings of
[`starts.json`](../src/px4_nmpc_config/config/experiments/challenging_course/starts.json). Each flight
folder `results/run_01/challenging_campaign/<mode>_NN/` holds the problem trace with one hash per problem
(`challenging_trace.jsonl`), the PX4 log (`flight.ulg`), telemetry, every solve (`solves.csv`) and the
ground-truth evaluation (`challenging_evaluation.json`).

### A5. Paired comparison

```bash
make replay RUN=results/run_01
```

Every problem of every flight is rebuilt, checked against its flight hash, and solved with cg2o and with
IPOPT (`<flight>/replay/paired.json`).

### A6. Video

```bash
make video RUN=results/run_01
```

One warm flight recorded with RViz and Gazebo side by side: `results/run_01/video_warm/slalom_sitl.mp4`
(and `screen.mp4`). Needs a visible X11 display. A cold flight:
`benchmark/flight/record_video.sh results/run_01 cold`.

### A7. Tables and figures

```bash
make tables figures DATASET=results/run_01
```

Writes `results/analysis/results_run_01/`: `campaign_summary.json` (flight table and, under `paper`, the
paired-comparison table of the navigation phase), `sensitivity.json`, and `figures/`. The same numbers laid out as
the paper's tables: `python3 experiments/paper_results.py <folder> --dataset results/run_01`.

All steps A2 to A7 in one command: `make all RUN=results/run_01` (after A1).

## B. Start from the shipped data

Each step can use the measurements of the paper instead of new ones.

| Goal | Command | Needs |
| --- | --- | --- |
| Tables and figures of the paper | `make tables figures` (`DATASET=data`) | Python only |
| Re-solve the 2832 shipped flight problems with both solvers | `make replay-shipped RUN=results/run_01`, then `make tables figures DATASET=results/run_01` | A1 without PX4 |
| Tune on the shipped tuning flights | `make tune RUN=results/run_01 TUNE_DATA=data` | A1 without PX4 |
| Fly with the tuned settings of the paper | `WARM_TUNED=data/tuning/20261110_101031_challenging_cg2o_warm/selected_parameters.json COLD_TUNED=data/tuning/20261110_133924_challenging_cg2o_cold/selected_parameters.json make flights RUN=results/run_01` | A1 |
| Video with the tuned settings of the paper | `make video RUN=results/run_01` (uses `data/tuning` when the run has no tuning) | A1 |

`make replay-shipped` copies each flight record into the run before solving it, so `data/` stays as shipped.
