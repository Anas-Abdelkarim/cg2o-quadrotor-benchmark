#pragma once
// Reuse this branch's numerical setting schema and cg2o start-point policy.
#include "solver_settings.h"
#include <cg2o/core/sparse_optimizer_ispd.h>
#include <pybind11/pybind11.h>
#include <limits>
namespace slalom {
namespace py = pybind11;
using px4nmpc::cg2o::SolverSettings;
template <typename T> T getOr(const py::dict &d, const char *key, const T &fallback) {
  if (!d.contains(key) || d[py::str(key)].is_none()) return fallback;
  return py::cast<T>(d[py::str(key)]);
}

py::dict childDict(const py::dict &d, const char *key) {
  if (!d.contains(key) || d[py::str(key)].is_none()) return py::dict();
  return py::cast<py::dict>(d[py::str(key)]);
}

SolverSettings parseSolverSettings(const py::dict &cfg) {
  SolverSettings s;
  s.optimizer = getOr<std::string>(cfg, "optimizer", s.optimizer);
  const auto unconstrained = childDict(cfg, "unconstrained_solver");
  const auto linear = childDict(cfg, "linear_solver");
  const auto general = childDict(cfg, "general");
  const auto warm = childDict(cfg, "warm_start");

  s.unconstrained_algorithm = getOr<std::string>(unconstrained, "type", s.unconstrained_algorithm);
  s.linear_solver = getOr<std::string>(linear, "type", s.linear_solver);
  s.equality_regularization = getOr<double>(linear, "equality_regularization", s.equality_regularization);
  s.verbose = getOr<bool>(general, "verbose", s.verbose);
  s.num_inner_iterations_max = getOr<int>(general, "num_inner_iterations_max", s.num_inner_iterations_max);
  s.alpha_backtracking = getOr<double>(general, "alpha_backtracking", s.alpha_backtracking);
  s.lagrange_multiplier_initial = getOr<double>(general, "lagrange_multiplier_initial", s.lagrange_multiplier_initial);
  s.epsilon_constraint = getOr<double>(general, "epsilon_constraint", s.epsilon_constraint);
  s.epsilon_convergence = getOr<double>(general, "epsilon_convergence", s.epsilon_convergence);
  s.convergence_criterion = getOr<int>(general, "convergence_criterion", s.convergence_criterion);
  s.feasible_start = getOr<bool>(general, "feasible_start", s.feasible_start);
  s.feasible_start_margin = getOr<double>(general, "feasible_start_margin", s.feasible_start_margin);

  s.warm_start_strategy = getOr<int>(warm, "strategy", s.warm_start_strategy);
  s.warm_start_terminal_rollout = getOr<bool>(warm, "terminal_rollout", s.warm_start_terminal_rollout);
  s.warm_start_kappa_strategy = getOr<int>(warm, "kappa_strategy", s.warm_start_kappa_strategy);
  s.warm_start_kappa_factor = getOr<double>(warm, "kappa_factor", s.warm_start_kappa_factor);
  s.warm_start_kappa_value = getOr<double>(warm, "kappa_value", s.warm_start_kappa_value);
  s.warm_start_kappa_gap = getOr<double>(warm, "kappa_gap", s.warm_start_kappa_gap);
  s.warm_start_push = getOr<double>(warm, "push", s.warm_start_push);
  s.warm_start_slack_push = getOr<double>(warm, "slack_push", s.warm_start_slack_push);
  s.warm_start_multiplier_push = getOr<double>(warm, "multiplier_push", s.warm_start_multiplier_push);
  s.warm_start_slack_projection = getOr<int>(warm, "slack_projection", s.warm_start_slack_projection);
  s.warm_start_centering_lo = getOr<double>(warm, "centering_lo", s.warm_start_centering_lo);
  s.warm_start_centering_hi = getOr<double>(warm, "centering_hi", s.warm_start_centering_hi);
  s.warm_start_centering_mode = getOr<int>(warm, "centering_mode", s.warm_start_centering_mode);
  s.warm_start_fallback = getOr<bool>(warm, "fallback", s.warm_start_fallback);
  s.warm_start_retry = getOr<bool>(warm, "retry", s.warm_start_retry);
  s.warm_start_divergence_factor = getOr<double>(warm, "divergence_factor", s.warm_start_divergence_factor);
  s.warm_start_reset_tolerance = getOr<double>(warm, "reset_tolerance", s.warm_start_reset_tolerance);
  s.warm_start_reset_test = getOr<int>(warm, "reset_test", s.warm_start_reset_test);
  s.warm_start_reset_mode = getOr<int>(warm, "reset_mode", s.warm_start_reset_mode);
  s.warm_start_middle_strategy = getOr<int>(warm, "middle_strategy", s.warm_start_middle_strategy);
  s.warm_start_middle_gap = getOr<double>(warm, "middle_gap", s.warm_start_middle_gap);
  s.warm_start_middle_kappa = getOr<double>(warm, "middle_kappa", s.warm_start_middle_kappa);
  s.warm_start_middle_iteration = getOr<int>(warm, "middle_iteration", s.warm_start_middle_iteration);

  const auto ispd = childDict(cfg, "ispd");
  s.ispd_step_size_strategy = getOr<int>(ispd, "step_size_strategy", s.ispd_step_size_strategy);
  s.ispd_ineq_backtracking_step_min = getOr<double>(ispd, "ineq_backtracking_step_min", s.ispd_ineq_backtracking_step_min);
  s.ispd_aux_backtracking_step_min = getOr<double>(ispd, "aux_backtracking_step_min", s.ispd_aux_backtracking_step_min);
  s.ispd_init_kappa_strategy = getOr<int>(ispd, "init_kappa_strategy", s.ispd_init_kappa_strategy);
  s.ispd_kappa_initial = getOr<double>(ispd, "kappa_initial", s.ispd_kappa_initial);
  s.ispd_update_kappa_strategy = getOr<int>(ispd, "update_kappa_strategy", s.ispd_update_kappa_strategy);
  s.ispd_kappa_update_factor = getOr<double>(ispd, "kappa_update_factor", s.ispd_kappa_update_factor);
  s.ispd_kappa_final = getOr<double>(ispd, "kappa_final", s.ispd_kappa_final);
  s.ispd_tau = getOr<double>(ispd, "tau", s.ispd_tau);
  s.ispd_limit_kappa_final = getOr<bool>(ispd, "limit_kappa_final", s.ispd_limit_kappa_final);
  s.ispd_init_slack_strategy = getOr<int>(ispd, "init_slack_strategy", s.ispd_init_slack_strategy);
  s.ispd_slack_variable_initial = getOr<double>(ispd, "slack_variable_initial", s.ispd_slack_variable_initial);
  s.ispd_init_lagrange_strategy = getOr<int>(ispd, "init_lagrange_strategy", s.ispd_init_lagrange_strategy);
  s.ispd_lagrange_multiplier_initial = getOr<double>(ispd, "lagrange_multiplier_initial", s.ispd_lagrange_multiplier_initial);
  s.ispd_ineq_prediction_strategy = getOr<int>(ispd, "ineq_prediction_strategy", s.ispd_ineq_prediction_strategy);
  s.ispd_ineq_prediction_step_strategy = getOr<int>(ispd, "ineq_prediction_step_strategy", s.ispd_ineq_prediction_step_strategy);
  s.ispd_ineq_prediction_param = getOr<double>(ispd, "ineq_prediction_param", s.ispd_ineq_prediction_param);
  s.ispd_keep_aux_positive_strategy = getOr<int>(ispd, "keep_aux_positive_strategy", s.ispd_keep_aux_positive_strategy);
  s.ispd_aux_scaling_factor = getOr<double>(ispd, "aux_scaling_factor", s.ispd_aux_scaling_factor);
  s.ispd_aux_correction_value = getOr<double>(ispd, "aux_correction_value", s.ispd_aux_correction_value);

  return s;
}

void configureAlgorithm(cg2o::SparseOptimizerISPD &o, const SolverSettings &s) {
  o.setStepSizeStrategy(s.ispd_step_size_strategy);
  o.setIneqBacktrackingStepMin(s.ispd_ineq_backtracking_step_min);
  o.setAuxBacktrackingStepMin(s.ispd_aux_backtracking_step_min);
  o.setInitKappaStrategy(s.ispd_init_kappa_strategy);
  o.setKappaInitial(s.ispd_kappa_initial);
  o.setUpdateKappaStrategy(s.ispd_update_kappa_strategy);
  o.setKappaUpdateFactor(s.ispd_kappa_update_factor);
  o.setKappaFinal(s.ispd_kappa_final);
  o.setTau(s.ispd_tau);
  o.setLimitKappaFinal(s.ispd_limit_kappa_final);
  o.setInitSlackStrategy(s.ispd_init_slack_strategy);
  o.setSlackVariableInitialIneq(s.ispd_slack_variable_initial);
  o.setInitLagrangeStrategy(s.ispd_init_lagrange_strategy);
  o.setLagrangeMultiplierInitialIneq(s.ispd_lagrange_multiplier_initial);
  o.setIneqPredictionStrategy(s.ispd_ineq_prediction_strategy);
  o.setIneqPredictionStepStrategy(s.ispd_ineq_prediction_step_strategy);
  o.setIneqPredictionParam(s.ispd_ineq_prediction_param);
  o.setKeepAuxPositiveStrategy(s.ispd_keep_aux_positive_strategy);
  o.setAuxScalingFactor(s.ispd_aux_scaling_factor);
  o.setAuxCorrectionValue(s.ispd_aux_correction_value);

  // warm start (receding horizon); strategy 0 = cold duals with the primal guess of this backend.
  // select() rejects an unknown method; cg2o validates the options when optimize() starts.
  auto &start = o.startPoint();
  start.select(static_cast<::cg2o::ISPDStartMethod>(s.warm_start_strategy));
  auto &w = start.options();
  w.warmPrimal = true;
  w.shift.enabled = s.warm_start_strategy > 0;
  w.shift.steps = 1;
  w.kappa.rule = static_cast<::cg2o::WarmKappaRule>(s.warm_start_kappa_strategy);
  w.kappa.factor = s.warm_start_kappa_factor;
  w.kappa.value = s.warm_start_kappa_value;
  w.kappa.gap = s.warm_start_kappa_gap;
  w.push.both = s.warm_start_push;
  w.push.slack = s.warm_start_slack_push;
  w.push.multiplier = s.warm_start_multiplier_push;
  if (s.warm_start_slack_projection != 0 && s.warm_start_slack_projection != 1)
    throw std::invalid_argument("warm_start.slack_projection: 0 retain, 1 project to -g");
  w.push.projectSlack = s.warm_start_slack_projection == 1;
  w.centering.lo = s.warm_start_centering_lo;
  w.centering.hi = s.warm_start_centering_hi < 0.0 ? std::numeric_limits<double>::infinity()
                                                   : s.warm_start_centering_hi;
  w.centering.mode = static_cast<::cg2o::CenteringMode>(s.warm_start_centering_mode);
  w.recovery.fallback = s.warm_start_fallback;
  w.recovery.retry = s.warm_start_retry;
  w.recovery.divergenceFactor = s.warm_start_divergence_factor;
  w.reset.tolerance = s.warm_start_reset_tolerance;
  w.reset.test = static_cast<::cg2o::ResetTest>(s.warm_start_reset_test);
  w.reset.mode = static_cast<::cg2o::ResetMode>(s.warm_start_reset_mode);
  if (start.method() == ::cg2o::ISPDStartMethod::PreviousMiddle) {
    auto &middle = start.settings<::cg2o::ispd::PreviousMiddle>();
    middle.rule = static_cast<::cg2o::MiddlePointRule>(s.warm_start_middle_strategy);
    middle.gap = s.warm_start_middle_gap;
    middle.kappa = s.warm_start_middle_kappa;
    middle.iteration = s.warm_start_middle_iteration;
  }
}


inline void applyLegacySettings(cg2o::SparseOptimizerISPD &o,const SolverSettings &s) {
  o.setVerbose(s.verbose);
  o.setNumInnerIterationsMax(s.num_inner_iterations_max);
  o.setAlphaBacktracking(s.alpha_backtracking);
  o.setLagrangeMultiplierInitial(s.lagrange_multiplier_initial);
  o.setEqualityRegularization(s.equality_regularization);
  o.setEpsilonConstraint(s.epsilon_constraint);
  o.setEpsilonConvergence(s.epsilon_convergence);
  o.setConvergenceCriterion(s.convergence_criterion);
  configureAlgorithm(o,s);
}
}
