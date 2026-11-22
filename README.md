# cg2o in a real-time quadrotor MPC: thrust and body rates to PX4

Code and data of the quadrotor benchmark (Sec. VII) of the paper *cg2o: Constrained Factor Graph Optimization via
an Infeasible-Start Primal–Dual Interior-Point Method*. Every table and figure of that section can be regenerated
from this repository, either from the shipped measurements or from scratch with PX4 software-in-the-loop flights.

The solver itself, cg2o, is a separate library: <https://anonymous.4open.science/r/cg2o/>. The build fetches the
version used for the paper from there.

## The problem

A quadrotor flies through an obstacle course in PX4 software-in-the-loop simulation with Gazebo. A nonlinear model
predictive controller (MPC) solves one optimal control problem every 100 ms and sends **collective thrust and body
rates** directly to the PX4 rate controller; no position or attitude loop of PX4 is used, and there is no planner.

```text
   take-off ──► climb ──► sphere ─ bar (over/under) ─ post ─ large sphere ─ post ─ sphere ─ bar ─ slot ─ sphere ──► goal
               28 m course, guide line passes through the obstacles; altitude follows a wave (1.5 m to 3.5 m)
```

| | |
| --- | --- |
| State (13) | position, velocity, roll/pitch/yaw, body rates, path progress |
| Input (5) | collective thrust, three body-rate setpoints, progress speed |
| Dynamics | rigid body with the PX4 rate loop as a first-order lag (0.095, 0.070, 0.390 s), RK4, 0.1 s, 40 steps (4 s) |
| Cost | contouring control: lag and contour error to a guide line, progress reward, input smoothness |
| Constraints | airframe as five spheres (centre and four rotors) at the true attitude, against 4 spheres, 4 vertical posts and 2 horizontal bars; tilt cone of 40°; thrust and rate limits; arena |
| Size per problem | 760 variables, 520 equality and 2800 inequality constraints (2000 for collision avoidance) |

Because the guide line runs through the obstacles, the cold start is obstacle-infeasible in 91% of the problems,
and the optimizer has to recover feasibility within the control period. Details: [docs/1_problem.md](docs/1_problem.md).

## Results of the paper and how to reproduce them

All 20 PX4 SITL flights (10 warm-started, 10 cold-started at every problem, randomized take-off points and
headings) reached the goal, and cg2o returned a feasible solution in all 2832 MPC problems. The same problems were
then solved offline by cg2o and by IPOPT (HSL, exact Hessian with its evaluation time excluded, tuned with the same
budget):

| Start | Solver | Iterations | Mean time | Within 100 ms | Feasible |
| --- | --- | --- | --- | --- | --- |
| Cold (1407 problems) | IPOPT | 51.1 | 176.7 ms | 14.5% | 88.8% |
| | **cg2o** | **17.5** | **46.7 ms** | **98.2%** | **100%** |
| Warm (1425 problems) | IPOPT | 35.1 | 144.8 ms | 39.4% | 95.5% |
| | **cg2o** | **7.7** | **21.7 ms** | **99.2%** | **100%** |

| Paper | Content | Command | Output |
| --- | --- | --- | --- |
| Table VIII | parameters of the benchmark | configuration | [`course.yaml`](src/px4_nmpc_config/config/experiments/challenging_course/course.yaml) |
| Table IX | selected solver settings | `run_experiment.py tables` (`tuning` for new studies) | `paper_table_ix.txt` |
| Table X | closed-loop flights | `run_experiment.py tables` (`full` for new flights) | `paper_table_x.txt` |
| Table XI | paired comparison | `run_experiment.py tables` (`replay` to re-solve) | `paper_table_xi.txt` |
| Table XII | sensitivity of cg2o | `run_experiment.py tables` (`tuning` for a new study) | `paper_table_xii.txt` |
| Fig. 10 | the 20 flights, top and side view | `run_experiment.py tables` | `paper_fig10_course.pdf` |
| Fig. 11 | solve-time distributions | `run_experiment.py tables` (`replay` to re-solve) | `paper_fig11_solve_times.pdf` |
| Videos | a warm- and a cold-started flight in RViz and Gazebo | `benchmark/flight/record_video.sh results/<run> warm\|cold` | [`videos/`](videos/README.md) |

`run_experiment.py` is `python3 experiments/run_experiment.py <preset>`; each run writes a new folder under
`results/experiments/`. Start with [QUICKSTART.md](QUICKSTART.md); it lists the expected numbers. Where each number
comes from in the data: [docs/4_results_map.md](docs/4_results_map.md).

## Repository layout

| Path | Content |
| --- | --- |
| [`experiments/`](experiments) | `run_experiment.py` (one entry point for every experiment) and `paper_results.py` (tables and figures of a results folder) |
| [`benchmark/model/`](benchmark/model) | the optimal control problem in C++, one model for both solvers (cg2o factor graph, IPOPT TNLP) |
| [`benchmark/flight/`](benchmark/flight) | PX4 SITL launcher, tuning flights, the 20-flight campaign, video, ground-truth evaluation |
| [`benchmark/comparison/`](benchmark/comparison) | paired replay of all flights with cg2o and IPOPT |
| [`benchmark/analysis/`](benchmark/analysis) | flight and comparison statistics, sensitivity table, figures |
| [`data/`](data/README.md) | measurements of the paper: recorded problems, replays, ground truth, tuning studies (17 MB, read-only) |
| `src/` | ROS 2 packages: runtime, controller node, PX4 interface, configuration ([`challenging_course`](src/px4_nmpc_config/config/experiments/challenging_course)) |
| `tools/solver_tuning/` | Bayesian tuning library (Optuna TPE) used for both solvers |
| `scripts/` | builds: `build_externals.sh` (IPOPT, HSL, g2o, cg2o, PX4), `build_benchmark.sh` (this repository) |
| [`docs/`](docs) | problem, comparison protocol, step-by-step reproduction with `make`, map of the results, troubleshooting of the PX4 flights |
| [`videos/`](videos/README.md) | flight videos: one warm- and one cold-started flight with the paper's settings |

Installation details and options: [INSTALL.md](INSTALL.md). Fairness of the comparison:
[docs/2_protocol.md](docs/2_protocol.md). Third-party software: [THIRD_PARTY.md](THIRD_PARTY.md).
