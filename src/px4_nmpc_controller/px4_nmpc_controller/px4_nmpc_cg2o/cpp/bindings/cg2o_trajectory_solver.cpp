// pybind11 wrapper exposing the cg2o trajectory formulation to Python with a call signature that
// px4_nmpc_controller/px4_nmpc_cg2o/solver.py maps onto NmpcOcpSolver.solve_ocp's inputs/outputs.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <memory>
#include <stdexcept>

#include "linear_solvers.h"
#include "mpc_formulation.h"
#include "mpc_parameters.h"
#include "solver_settings.h"

namespace py = pybind11;
using px4nmpc::cg2o::SolveInfo;
using px4nmpc::cg2o::SolverSettings;
using px4nmpc::cg2o::TrajFormulationBase;
using px4nmpc::cg2o::TrajParams;

namespace {

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

py::dict infoDict(const SolveInfo &i) {
  py::dict d;
  d["iterations"] = i.iterations;
  d["converged"] = i.converged;
  d["retried"] = i.retried;
  d["restored"] = i.restored;
  d["warm_used"] = i.warm_used;
  d["reset_rows"] = i.reset_rows;
  d["start_kappa"] = i.start_kappa;
  d["final_kappa"] = i.final_kappa;
  d["primal_warm"] = i.primal_warm;
  return d;
}

class Cg2oTrajectorySolver {
public:
  Cg2oTrajectorySolver(int N, double dt, int K, double g, double vmax, double tilt_max,
                       Eigen::Vector3d jerk_max, double acc_up_max, double acc_down_max,
                       Eigen::Vector3d q_pos, Eigen::Vector3d q_vel, Eigen::Vector3d q_pos_terminal,
                       Eigen::Vector3d q_vel_terminal, double r_jerk, double w_acc,
                       double w_slack_obstacle_l1, double w_slack_obstacle_l2, double w_slack_arena_l1,
                       double w_slack_arena_l2, double w_slack_dynamic_l1, double w_slack_dynamic_l2,
                       double w_terminal_penalty, Eigen::Vector3d r_delta_input_first, int max_iter,
                       py::dict solver_config, bool yaw = false, double yaw_rate_max = 0.8,
                       double yaw_acc_max = 2.0, double w_yaw_lateral = 1.0, double w_yaw_rate = 0.05,
                       double w_yaw_acc = 0.05) {
    if (N < 2 || dt <= 0.0 || K < 1) throw std::runtime_error("N >= 2, dt > 0 and K >= 1 are required");
    _max_iter = max_iter;
    _param = std::make_shared<TrajParams>();
    auto &p = *_param;
    p.N = N; p.dt = dt; p.K = K; p.g = g; p.vmax = vmax; p.tilt_max = tilt_max; p.jerk_max = jerk_max;
    p.acc_up_max = acc_up_max; p.acc_down_max = acc_down_max; p.q_pos = q_pos; p.q_vel = q_vel;
    p.q_pos_terminal = q_pos_terminal; p.q_vel_terminal = q_vel_terminal; p.r_jerk = r_jerk; p.w_acc = w_acc;
    p.w_slack_obstacle_l1 = w_slack_obstacle_l1; p.w_slack_obstacle_l2 = w_slack_obstacle_l2;
    p.w_slack_arena_l1 = w_slack_arena_l1; p.w_slack_arena_l2 = w_slack_arena_l2;
    p.w_slack_dynamic_l1 = w_slack_dynamic_l1; p.w_slack_dynamic_l2 = w_slack_dynamic_l2;
    p.r_delta_input_first = r_delta_input_first; p.w_terminal_penalty = w_terminal_penalty;
    p.yaw = yaw; p.yaw_rate_max = yaw_rate_max; p.yaw_acc_max = yaw_acc_max;
    p.w_yaw_lateral = w_yaw_lateral; p.w_yaw_rate = w_yaw_rate; p.w_yaw_acc = w_yaw_acc;
    p.resizeHorizonBuffers();
    _settings = parseSolverSettings(solver_config);
    _formulation = px4nmpc::cg2o::makeTrajFormulation(_param, _settings);
  }

  // shift_steps < 0: cold start; >= 0: warm start shifted by that many stages
  py::tuple solve(Eigen::VectorXd x0, Eigen::MatrixXd pref, Eigen::MatrixXd vref, Eigen::MatrixXd obstacles,
                  Eigen::Vector3d arena_lb, Eigen::Vector3d arena_ub, Eigen::Vector3d uprev,
                  Eigen::VectorXd hdg_ref = Eigen::VectorXd(), Eigen::VectorXd hdg_w = Eigen::VectorXd(),
                  int shift_steps = 1) {
    const int N = _param->N;
    const int K = _param->K;
    if (pref.rows() != 3 || pref.cols() != N + 1) throw std::runtime_error("pref must be 3 x (N+1)");
    if (vref.rows() != 3 || vref.cols() != N + 1) throw std::runtime_error("vref must be 3 x (N+1)");
    if (obstacles.rows() != 4 || obstacles.cols() != K) throw std::runtime_error("obstacles must be 4 x K");
    const int nx = _param->yaw ? 11 : 9;
    if (x0.size() != nx) throw std::runtime_error("x0 must have " + std::to_string(nx) + " entries");
    _param->x0 = x0.head<9>();
    if (_param->yaw) {
      _param->psi0 = x0[9];
      _param->om0 = x0[10];
      _param->hdg_ref = (hdg_ref.size() == N + 1) ? hdg_ref : Eigen::VectorXd::Constant(N + 1, x0[9]);
      _param->hdg_w = (hdg_w.size() == N + 1) ? hdg_w : Eigen::VectorXd::Zero(N + 1);
    }
    _param->pref = pref;
    _param->vref = vref;
    _param->obstacles = obstacles;
    _param->arena_lb = arena_lb;
    _param->arena_ub = arena_ub;
    _param->uprev = uprev;

    int iters = 0;
    double solve_ms = 0.0;
    {
      // pure C++ from here on: let other Python threads run (the ROS node keeps streaming setpoints to PX4
      // while a long solve is running; PX4 leaves OFFBOARD after COM_OF_LOSS_T without setpoints)
      py::gil_scoped_release release;
      const auto t0 = std::chrono::high_resolution_clock::now();
      iters = _formulation->solve(_max_iter, shift_steps);
      const auto t1 = std::chrono::high_resolution_clock::now();
      solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return py::make_tuple(_formulation->statesMatrix(), _formulation->inputsMatrix(),
                          _formulation->slacksMatrix(), iters, _formulation->lastCost(), solve_ms,
                          infoDict(_formulation->lastInfo()));
  }

  void resetWarmStart() { _formulation->resetWarmStart(); }
  std::string optimizer() const { return _formulation->optimizerName(); }
  std::string linearSolver() const { return _formulation->linearSolverName(); }
  int maxIterations() const { return _max_iter; }

private:
  int _max_iter = 60;
  SolverSettings _settings;
  std::shared_ptr<TrajParams> _param;
  std::unique_ptr<TrajFormulationBase> _formulation;
};

} // namespace

PYBIND11_MODULE(cpp_px4_nmpc_cg2o, m) {
  m.doc() = "cg2o constrained factor-graph backend for PX4 NMPC";
  // all optimizers are compiled in; cg2o.yaml `optimizer:` selects one at run time
  m.attr("optimizers") = py::make_tuple("ispd");
  m.attr("compiled_optimizer") = "runtime";
  m.attr("api_version") = 2;
  m.def("linear_solvers", []() {
    py::list out;
    for (const auto &s : px4nmpc::cg2o::linearSolverCatalog()) {
      py::dict d;
      d["key"] = s.key; d["name"] = s.name; d["needs_regularization"] = s.needs_regularization;
      d["available"] = s.available; d["note"] = s.note;
      out.append(d);
    }
    return out;
  });

  py::class_<Cg2oTrajectorySolver>(m, "Cg2oTrajectorySolver")
      .def(py::init<int, double, int, double, double, double, Eigen::Vector3d, double, double,
                    Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d, double, double,
                    double, double, double, double, double, double, double, Eigen::Vector3d, int,
                    py::dict, bool, double, double, double, double, double>(),
           py::arg("N"), py::arg("dt"), py::arg("K"), py::arg("g"), py::arg("vmax"), py::arg("tilt_max"),
           py::arg("jerk_max"), py::arg("acc_up_max"), py::arg("acc_down_max"), py::arg("q_pos"),
           py::arg("q_vel"), py::arg("q_pos_terminal"), py::arg("q_vel_terminal"), py::arg("r_jerk"),
           py::arg("w_acc"), py::arg("w_slack_obstacle_l1"), py::arg("w_slack_obstacle_l2"),
           py::arg("w_slack_arena_l1"), py::arg("w_slack_arena_l2"), py::arg("w_slack_dynamic_l1"),
           py::arg("w_slack_dynamic_l2"), py::arg("w_terminal_penalty"), py::arg("r_delta_input_first"),
           py::arg("max_iter"), py::arg("solver_config"), py::arg("yaw") = false,
           py::arg("yaw_rate_max") = 0.8, py::arg("yaw_acc_max") = 2.0, py::arg("w_yaw_lateral") = 1.0,
           py::arg("w_yaw_rate") = 0.05, py::arg("w_yaw_acc") = 0.05)
      .def("solve", &Cg2oTrajectorySolver::solve, py::arg("x0"), py::arg("pref"), py::arg("vref"),
           py::arg("obstacles"), py::arg("arena_lb"), py::arg("arena_ub"), py::arg("uprev"),
           py::arg("hdg_ref") = Eigen::VectorXd(), py::arg("hdg_w") = Eigen::VectorXd(),
           py::arg("shift_steps") = 1)
      .def("reset_warm_start", &Cg2oTrajectorySolver::resetWarmStart)
      .def_property_readonly("optimizer", &Cg2oTrajectorySolver::optimizer)
      .def_property_readonly("linear_solver", &Cg2oTrajectorySolver::linearSolver)
      .def_property_readonly("max_iterations", &Cg2oTrajectorySolver::maxIterations);
}
