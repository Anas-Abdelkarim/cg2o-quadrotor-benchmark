#pragma once
#include <string>

namespace px4nmpc::cg2o {

// All numerical settings of the cg2o backend (cg2o.yaml). Nothing here changes the
// optimal-control problem itself; the problem is defined by TrajParams.
struct SolverSettings {
  // ---- optimizer: ISPD (the only one; any other value is rejected)
  std::string optimizer = "ispd";
  std::string unconstrained_algorithm = "GN";  // GN only: cg2o rejects Levenberg and Dogleg
  std::string linear_solver = "umfpack";     // key of linear_solvers.h (umfpack, qdldlt, ...)
  // sigma of the quasi-definite KKT system: -sigma on the equality rows (0 = exact KKT matrix).
  // Needed only by LDL^T/Cholesky-type solvers without pivoting; qdldlt has its own delta.
  double equality_regularization = 0.0;
  bool verbose = false;
  int num_inner_iterations_max = 10;
  double alpha_backtracking = -1.0;
  double lagrange_multiplier_initial = 0.0;
  double epsilon_constraint = 1e-4;
  double epsilon_convergence = 1e-4;
  int convergence_criterion = 1;              // 0 UpdateNorm, 1 NewtonDecrement, 2 GradientNorm, 3 RelativeNewtonDecrement

  // ---- start of a COLD solve
  // feasible_start: lift the stage slacks of the cold rollout so that every softened inequality is
  // strictly satisfied (and keep inputs strictly inside their boxes). Default off = the infeasible
  // start that ISPD is designed for.
  bool feasible_start = false;
  double feasible_start_margin = 0.05;

  // ---- warm start (receding horizon)
  // 0: primal shift only (the solution is shifted by the PX4 backend, duals/slacks/kappa start cold)
  // 1: ISPD final iterate (x, nu, lambda, s, kappa), shifted by cg2o (ISPDStartMethod::PreviousFinal)
  // 2: ISPD middle iterate (ISPDStartMethod::PreviousMiddle)
  int warm_start_strategy = 0;
  // roll the last stages out with the dynamics after a shift (consistent tail instead of replicating
  // the last stage; the IPOPT backend does the same)
  bool warm_start_terminal_rollout = true;
  int warm_start_kappa_strategy = 0;         // 0 stored*factor, 1 duality gap, 2 constant value, 3 relative gap
  double warm_start_kappa_factor = 1.0;
  double warm_start_kappa_value = 1.0;
  double warm_start_kappa_gap = 1e-2;
  double warm_start_push = 0.0;
  double warm_start_slack_push = 0.0;
  double warm_start_multiplier_push = 0.0;
  int warm_start_slack_projection = 0;
  double warm_start_centering_lo = 0.0;      // 0 / inf = off
  double warm_start_centering_hi = -1.0;     // < 0 means +inf
  int warm_start_centering_mode = 0;
  bool warm_start_fallback = true;
  bool warm_start_retry = true;
  double warm_start_divergence_factor = 10.0;
  double warm_start_reset_tolerance = 0.5;
  int warm_start_reset_test = 1;
  int warm_start_reset_mode = 2;
  int warm_start_middle_strategy = 3;
  double warm_start_middle_gap = 3e-2;
  double warm_start_middle_kappa = 30.0;
  int warm_start_middle_iteration = 3;

  // ---- ISPD
  int ispd_step_size_strategy = 0;
  double ispd_ineq_backtracking_step_min = 1.0;
  double ispd_aux_backtracking_step_min = 0.7;
  int ispd_init_kappa_strategy = 1;
  double ispd_kappa_initial = 1e-3;
  int ispd_update_kappa_strategy = 0;
  double ispd_kappa_update_factor = 15.0;
  double ispd_kappa_final = 1500.0;
  double ispd_tau = 1.4;
  bool ispd_limit_kappa_final = true;
  int ispd_init_slack_strategy = 0;
  double ispd_slack_variable_initial = 10.0;
  int ispd_init_lagrange_strategy = 0;
  double ispd_lagrange_multiplier_initial = 10.0;
  int ispd_ineq_prediction_strategy = 0;
  int ispd_ineq_prediction_step_strategy = 0;
  double ispd_ineq_prediction_param = 1.0;
  int ispd_keep_aux_positive_strategy = 0;
  double ispd_aux_scaling_factor = 0.01;
  double ispd_aux_correction_value = 1e-5;
};

// What the last solve did (filled by TrajFormulation::solve()).
struct SolveInfo {
  int iterations = 0;
  bool converged = false;      // ISPD's own convergence test (lastConverged())
  bool retried = false;        // ISPD: warm solve broke down and was repeated with cold duals
  bool restored = false;       // ISPD: the returned point is the restored starting point
  int warm_used = 0;           // 0 cold duals, 1 final iterate, 2 middle iterate
  int reset_rows = 0;          // ISPD change detector: rows re-initialised
  double start_kappa = 0.0;
  double final_kappa = 0.0;
  bool primal_warm = false;    // the primal guess came from the previous solution (shifted)
};

} // namespace px4nmpc::cg2o
