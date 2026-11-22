# Quickstart

From a clean Ubuntu 22.04 (x86-64) to the tables and figures of Sec. VII of the paper. Run every command from
the repository root. Steps 1 to 5 take about one hour, mostly for building IPOPT and g2o.

| You want | Steps | Needs |
| --- | --- | --- |
| The paper's tables and figures from the shipped measurements | 1, 2, 6a | Python only, seconds |
| The paired comparison re-solved on your machine (Table XI, Fig. 11) | 1 to 5, 6b | solvers and HSL, about 30 min |
| The tuning studies (Tables IX, XII) | 1 to 5, 6c | solvers and HSL, many hours |
| Everything from scratch with PX4 flights (Table X, Fig. 10, video) | 1 to 5, 6d | ROS 2 Humble, PX4 SITL, Gazebo; see [INSTALL.md](INSTALL.md) |

## 1. System packages

```bash
sudo apt update
sudo apt install -y git curl unzip cmake ninja-build build-essential gfortran pkg-config \
  python3-pip python3-venv python3-dev \
  libeigen3-dev libspdlog-dev libfmt-dev libsuitesparse-dev libmetis-dev libomp-dev \
  liblapack-dev libblas-dev libhwloc-dev
```

The paper used Intel oneAPI MKL 2024.0 as BLAS/LAPACK of IPOPT and HSL (found automatically in
`/opt/intel/oneapi/mkl/latest`). Without it, the system BLAS/LAPACK is used; the results are the same, IPOPT's
times may differ slightly.

## 2. Python environment

```bash
python3 -m venv --system-site-packages venv
venv/bin/pip install pip==26.2.2 setuptools==79.0.1
venv/bin/pip install -r requirements.txt
```

`--system-site-packages` keeps the ROS 2 Python modules visible for the PX4 flights. The pip of Ubuntu 22.04
(22.0) cannot install the tuning package in editable mode, hence the upgrade. The requirements pin the versions
used for the paper.

## 3. Third-party solvers

HSL (MA27, MA57) is licensed separately and free for academic use: request "Coin-HSL" at
<https://licences.stfc.ac.uk/product/coin-hsl>. The paper used `coinhsl-2024.05.15`.

```bash
HSL_SOURCE=/path/to/coinhsl-2024.05.15.tar.gz MARCH_NATIVE=ON ./scripts/build_externals.sh --solver-only
```

This builds, under `install/externals/`: IPOPT 3.14.21 (with MUMPS and SPRAL), HSL, g2o and cg2o, each at the
exact version used for the paper (see [INSTALL.md](INSTALL.md#versions)). cg2o is downloaded from the link given in
the paper and checked against a hash of its source. `MARCH_NATIVE=ON` compiles g2o and cg2o with
`-march=native`, as for the paper; the benchmark extension (step 4) is always compiled that way and must match.

## 4. Build

```bash
./scripts/build_benchmark.sh            # add --no-ros without ROS 2
```

Builds the solver backends, the benchmark extension `build/challenging_mpc/_challenging*.so` (the MPC problem,
solved by cg2o and by IPOPT) and, if ROS 2 Humble is installed, the ROS 2 workspace.

## 5. Tests

```bash
source scripts/setup_env.sh
make test
```

Five test suites run, each in its own process; all must pass (199 tests).

## 6. Reproduce the results

Every experiment writes into a new folder `results/experiments/<date>_<time>_<preset>/` with the paper's tables
as text and JSON (`paper_table_ix.txt` ... `paper_table_xii.txt`) and its figures
(`paper_fig10_course.pdf`, `paper_fig11_solve_times.pdf`). Earlier folders are never changed.

**a. Tables and figures from the shipped measurements** (`data/`, the paper's 20 flights, replays and tuning
studies):

```bash
python3 experiments/run_experiment.py tables
```

The numbers are those of the paper exactly (below).

**b. Paired comparison re-solved on this machine:**

```bash
python3 experiments/run_experiment.py replay --quick     # one warm and one cold flight, a few minutes
python3 experiments/run_experiment.py replay             # all 2832 problems of the 20 flights
python3 experiments/run_experiment.py replay --no-hsl    # without an HSL licence: IPOPT with MUMPS
```

Every recorded problem is rebuilt, checked against the SHA-256 hash stored in flight, and solved by cg2o and by
IPOPT. cg2o's iteration counts, feasibility and final costs are identical to the paper in every problem, and so
are IPOPT's for the cold starts. IPOPT's warm-start values can differ slightly with a fresh IPOPT/HSL build (a clean
build gave 34.4 instead of 35.1 mean iterations and 95.9% instead of 95.5% feasible solves), because one different
early solve changes the rest of IPOPT's warm-start chain. The times depend on the machine.
Without HSL (`--no-hsl`, build step 3 without `HSL_SOURCE`), IPOPT runs with MUMPS instead of MA57/MA27: cg2o's
numbers are unchanged, IPOPT's differ from the paper (on this machine about four times slower).
The paper's times were measured on an Intel Core i9-12900H with the timed steps on CPUs 0-3 isolated from the
scheduler (`isolcpus=0-3`). Choose other CPUs with `--cpus`; compare times through the ratios between the solvers.

**c. Tuning studies:**

```bash
python3 experiments/run_experiment.py tuning --quick     # 3 trials per study, to check the setup
python3 experiments/run_experiment.py tuning             # 80 + 80 + 150 + 150 trials, many hours
```

The four equal-budget studies (cg2o and IPOPT, warm and cold) replay the shipped tuning flights. Their objective
is the measured solve time, so a new study can select a different setting than the paper's; Table IX then lists
the new settings, and Table XII is computed from the new cold-start study of cg2o.

**d. Everything from scratch, with PX4 flights:** install profile B of [INSTALL.md](INSTALL.md), then

```bash
python3 experiments/run_experiment.py full
```

Tuning flights, tuning, 20 flights, paired replay and video. The step-by-step version with `make` targets is in
[docs/3_reproduce.md](docs/3_reproduce.md).

**Tables and figures of an existing folder** (any of the above, or with other inputs):

```bash
python3 experiments/paper_results.py                     # newest complete folder, --quick runs skipped
python3 experiments/paper_results.py results/experiments/<folder>
python3 experiments/paper_results.py results/experiments/<folder> --compare   # against the paper's numbers
```

`--compare` checks the folder against [`experiments/paper_reference.yaml`](experiments/paper_reference.yaml):
values that do not depend on timing (iterations, feasibility, flights, sensitivity) must be the same, times are
listed with the paper's; the result is written to `comparison_with_paper.txt`.

## Expected numbers

From `python3 experiments/run_experiment.py tables` (and the paper):

Table X, closed-loop flights with cg2o in PX4 SITL:

| Start | Goal | Time to goal (s) | Max. tilt (deg), median / largest | Feasible solves | Solve time (ms), median / P95 |
| --- | --- | --- | --- | --- | --- |
| Warm | 10/10 | 14.0-15.8 | 21 / 41 | 1425/1425 | 20.2 / 60.1 |
| Cold | 10/10 | 14.2-15.0 | 21 / 34 | 1407/1407 | 55.4 / 106.0 |

Table XI, paired comparison (times in ms, IPOPT without its Hessian evaluation):

| Start | Solver | Iter. | Avg | Med. | P95 | Max | <= 100 ms | Feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cold | IPOPT | 51.1 | 176.7 | 167.7 | 286.5 | 540.0 | 14.5% | 88.8% |
| | cg2o | 17.5 | 46.7 | 41.3 | 77.8 | 271.9 | 98.2% | 100% |
| Warm | IPOPT | 35.1 | 144.8 | 112.6 | 294.6 | 560.6 | 39.4% | 95.5% |
| | cg2o | 7.7 | 21.7 | 17.0 | 49.6 | 214.2 | 99.2% | 100% |

Cold starts infeasible with respect to the obstacles: 2582 of 2832 problems (91%).

Table XII, sensitivity (98 trials of the cold-start search of cg2o): for example the scaling safeguard 77 trials,
22% rejected, median relative time 0.80; correction rule 21 trials, 52% rejected, 0.85. Table IX: the selected
settings of `data/tuning/*/selected_parameters.json`; IPOPT cold uses the default options with MA57.

With `replay`, the iteration columns and "Feasible" of cg2o (and of IPOPT cold) are reproduced exactly; with
`replay --quick`, they are those of `warm_01` and `cold_01`.
