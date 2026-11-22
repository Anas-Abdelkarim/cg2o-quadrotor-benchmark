# 2. Evaluation protocol

The aim is a comparison in which every difference can be attributed to the solver.

## Closed-loop flights (cg2o)

* 20 PX4 SITL flights: 10 with warm starts, 10 with a cold start at every problem.
* Take-off point and heading randomized per flight ([`starts.json`](../src/px4_nmpc_config/config/experiments/challenging_course/starts.json)).
* Settings frozen before the campaign (tuned settings, below). Warm flights use the cold-tuned values for
  their cold solves (start of navigation).
* Iteration budget 80, control period 100 ms; the controller runs on CPUs 2–3.
* Ground truth (Gazebo) is used only after the flight, for clearance, tilt and time to the goal.

## Paired replay (cg2o and IPOPT)

IPOPT is not flown. Instead, every problem that cg2o solved in flight is replayed:

* **Same problem.** Each problem is rebuilt from the flight record and its SHA-256 hash is checked against
  the hash logged in flight.
* **Same start.** Cold: the same guide-line rollout. Warm: each solver's own shifted previous solution,
  with its own history over the flight.
* **Same model code.** Both solvers evaluate the model through the same C++ implementation
  ([`benchmark/model/adapters.cpp`](../benchmark/model/adapters.cpp)).
* **Same budget and acceptance.** 80 iterations; a solution is usable if it is finite with equality and
  inequality residuals at most 10⁻³.
* **IPOPT in its best configuration.** HSL MA57 (MA27 and MUMPS also searched in tuning), exact Hessian, and the
  time spent evaluating the Hessian is **subtracted** from IPOPT's time (about 80% of its call time).
* **Timing.** The optimization call only (`optimize()` for cg2o, `OptimizeTNLP()` for IPOPT), Release
  builds, single-threaded BLAS/MKL/OpenMP, one solver after the other on isolated CPUs 0–3, every problem
  solved once and every measurement kept.

The replay is open loop: it does not show what slower or unusable solves would have done to a flight.

## Equal-budget tuning

Both solvers are tuned with the same Bayesian optimizer (TPE), the same number of trials (80 warm, 150
cold), the same gates (no unusable solve, no solve at the iteration limit, residual and cost limits), the
same objective (mean solve time relative to the untuned setting) and the same validation flight.
Specifications: [`challenging_course/tuning/`](../src/px4_nmpc_config/config/experiments/challenging_course/tuning).
No IPOPT setting passed the cold-start gates, its untuned setting included, so IPOPT's cold solves use
its default options with MA57 ([`ipopt_cold_default.json`](../src/px4_nmpc_config/config/experiments/challenging_course/ipopt_cold_default.json)).

## Sensitivity study

The 150 trials of the cold-start cg2o study are grouped by solver option (prediction, barrier update,
positivity safeguard, ζ_aux, κ_final, β); for each group the share of rejected trials and the median
relative time of the accepted ones are reported, plus Spearman correlations. The search is adaptive and
samples options jointly, so the groups describe this search, not controlled ablations.
