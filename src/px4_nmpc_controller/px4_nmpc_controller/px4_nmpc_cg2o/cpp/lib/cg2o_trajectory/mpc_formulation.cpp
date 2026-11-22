#include "mpc_formulation.h"

#include "linear_solvers.h"
#include "mpc_edges.h"
#include "mpc_optimizer.h"
#include "mpc_vertices.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace px4nmpc::cg2o {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

Eigen::Matrix<double, 9, 9> stateInfo(const Eigen::Vector3d &qp, const Eigen::Vector3d &qv, double w_acc) {
  Eigen::Matrix<double, 9, 1> d;
  d << qp, qv, w_acc, w_acc, w_acc;
  return d.asDiagonal();
}

void configureAlgorithm(OptimizerISPD &o, const SolverSettings &s) {
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

class TrajFormulation final : public TrajFormulationBase {
public:
  TrajFormulation(std::shared_ptr<TrajParams> param, SolverSettings settings)
      : _param(std::move(param)), _settings(std::move(settings)) {
    _linear_key = canonicalLinearSolverKey(_settings.linear_solver);
    auto block = std::make_unique<g2o::BlockSolverX>(makeLinearSolver(_linear_key));
    // ISPD sets its own step; cg2o rejects Levenberg and Dogleg, which accept a step by the
    // objective alone and damp the multiplier rows
    const std::string alg = lower(_settings.unconstrained_algorithm);
    if (alg != "gn" && alg != "gauss_newton")
      throw std::runtime_error("cg2o unconstrained algorithm must be GN (Gauss-Newton), got: " +
                               _settings.unconstrained_algorithm);
    _algorithm = std::make_unique<g2o::OptimizationAlgorithmGaussNewton>(std::move(block));

    _optimizer = std::make_unique<OptimizerISPD>();
    _optimizer->setVerbose(_settings.verbose);
    _optimizer->setAlgorithm(_algorithm.get());
    _optimizer->setNumInnerIterationsMax(_settings.num_inner_iterations_max);
    _optimizer->setAlphaBacktracking(_settings.alpha_backtracking);
    _optimizer->setLagrangeMultiplierInitial(_settings.lagrange_multiplier_initial);
    _optimizer->setEqualityRegularization(_settings.equality_regularization);
    _optimizer->setEpsilonConstraint(_settings.epsilon_constraint);
    _optimizer->setEpsilonConvergence(_settings.epsilon_convergence);
    _optimizer->setConvergenceCriterion(_settings.convergence_criterion);
    configureAlgorithm(*_optimizer, _settings);

    buildGraph();
    registerShiftSequences();
    // The per-solve reference is not available yet: the cold start is seeded in solve(), after
    // the Python wrapper has refreshed x0/pref/vref/obstacles.
    _optimizer->initializeOptimization();
  }

  int solve(int max_iterations, int shift_steps) override {
    _info = SolveInfo();
    const bool cold = !_solved_once || shift_steps < 0;
    auto &start = _optimizer->startPoint();
    if (cold) {
      seedColdStart();
      start.invalidate();
    } else {
      const int n = std::clamp(shift_steps, 0, _param->N - 1);
      const bool ispd_warm = _settings.warm_start_strategy > 0 && start.available();
      if (ispd_warm) {
        // cg2o restores x, nu, lambda, s, kappa of the stored iterate and shifts them n stages
        auto &shift = start.options().shift;
        shift.enabled = n > 0;
        if (n > 0) shift.steps = n;
      }
      if (!ispd_warm) {
        // primal warm start: shift the last returned solution (duals/slacks start cold)
        shiftPrimal(n);
        if (_settings.feasible_start) liftSlacksToFeasible();
      }
      _info.primal_warm = true;
    }
    _solved_once = true;
    setFixedInitialState();

    _optimizer->optimize(max_iterations);
    const auto &report = start.lastReport();
    _info.iterations = _optimizer->lastIterationCount();
    _info.converged = _optimizer->lastConverged();
    _info.retried = report.retried;
    _info.restored = report.restored;
    _info.warm_used = static_cast<int>(report.used);
    _info.reset_rows = report.resetRows;
    _info.start_kappa = report.kappa;
    _info.final_kappa = _optimizer->lastKappa();
    return _info.iterations;
  }

  void resetWarmStart() override {
    // The next solve() seeds a fresh rollout AFTER the new x0/reference has been loaded.
    _solved_once = false;
    _optimizer->startPoint().invalidate();
  }

  Eigen::MatrixXd statesMatrix() const override {
    const int N = _param->N;
    Eigen::MatrixXd X(_param->yaw ? 11 : 9, N + 1);
    for (int k = 0; k <= N; ++k) {
      X.block<3, 1>(0, k) = _vp[k]->estimate();
      X.block<3, 1>(3, k) = _vv[k]->estimate();
      X.block<3, 1>(6, k) = _va[k]->estimate();
      if (_param->yaw) {
        X(9, k) = _vpsi[k]->estimate()[0];
        X(10, k) = _vom[k]->estimate()[0];
      }
    }
    return X;
  }

  Eigen::MatrixXd inputsMatrix() const override {
    const int N = _param->N;
    Eigen::MatrixXd U(_param->yaw ? 4 : 3, N);
    for (int k = 0; k < N; ++k) {
      U.block<3, 1>(0, k) = _vj[k]->estimate();
      if (_param->yaw) U(3, k) = _val[k]->estimate()[0];
    }
    return U;
  }

  Eigen::MatrixXd slacksMatrix() const override {
    Eigen::MatrixXd S(3, _param->N);
    for (int k = 0; k < _param->N; ++k) S.col(k) = _vs[k]->estimate();
    return S;
  }

  double lastCost() override {
    double cost = 0.0;
    for (auto *e : _cost_edges) {
      // chi2() returns the error cached at the last linearisation, i.e. one Newton step behind
      // the returned point: re-evaluate at the current vertex values first.
      e->computeError();
      cost += e->chi2();
    }
    // Remove the shifted-slack constants (one slack cost for k = 1..N-1) so the objective is the
    // IPOPT one: l1*s + l2*s^2 = l2*(s + l1/(2 l2))^2 - l1^2/(4 l2).
    auto c = [](double l1, double l2) { return (l2 > 0.0) ? l1 * l1 / (4.0 * l2) : 0.0; };
    cost -= (_param->N - 1) * (c(_param->w_slack_obstacle_l1, _param->w_slack_obstacle_l2) +
                               c(_param->w_slack_arena_l1, _param->w_slack_arena_l2) +
                               c(_param->w_slack_dynamic_l1, _param->w_slack_dynamic_l2));
    // IPOPT also evaluates the (constant) stage-0 state cost of the fixed x0.
    const Eigen::Vector3d ep0 = _param->x0.segment<3>(0) - _param->pref.col(0);
    const Eigen::Vector3d ev0 = _param->x0.segment<3>(3) - _param->vref.col(0);
    cost += (_param->q_pos.array() * ep0.array().square()).sum();
    cost += (_param->q_vel.array() * ev0.array().square()).sum();
    cost += _param->w_acc * _param->x0.segment<3>(6).squaredNorm();
    if (_param->yaw) {
      const double hw0 = std::max(0.0, _param->hdg_w[0]);
      const double e0 = _param->psi0 - _param->hdg_ref[0];
      const double lat0 = _param->x0[3] * std::sin(_param->psi0) - _param->x0[4] * std::cos(_param->psi0);
      cost += hw0 * e0 * e0 + _param->w_yaw_lateral * lat0 * lat0 + _param->w_yaw_rate * _param->om0 * _param->om0;
    }
    return cost;
  }

  const SolveInfo &lastInfo() const override { return _info; }
  std::string optimizerName() const override { return "ispd"; }
  std::string linearSolverName() const override { return _linear_key; }

private:
  void buildGraph();
  void registerShiftSequences();
  void seedColdStart();
  void shiftPrimal(int n);
  void rolloutTail(int first_stage);
  void liftSlacksToFeasible();
  void setFixedInitialState();

  std::shared_ptr<TrajParams> _param;
  SolverSettings _settings;
  std::string _linear_key;
  // Destruction order (reverse of declaration): edges, vertices, optimizer, algorithm.
  std::unique_ptr<g2o::OptimizationAlgorithm> _algorithm;
  std::unique_ptr<OptimizerISPD> _optimizer;

  std::vector<std::unique_ptr<Vertex_p>> _vp;
  std::vector<std::unique_ptr<Vertex_v>> _vv;
  std::vector<std::unique_ptr<Vertex_a>> _va;
  std::vector<std::unique_ptr<Vertex_j>> _vj;
  std::vector<std::unique_ptr<Vertex_s>> _vs;
  std::vector<std::unique_ptr<Vertex_psi>> _vpsi;
  std::vector<std::unique_ptr<Vertex_om>> _vom;
  std::vector<std::unique_ptr<Vertex_al>> _val;

  std::vector<std::shared_ptr<g2o::OptimizableGraph::Edge>> _edges;
  std::vector<g2o::OptimizableGraph::Edge *> _cost_edges;
  // stage sequences (stage order) of the edges whose multipliers/slacks are shifted (ISPD warm start)
  std::vector<EdgeDynamics *> _e_dyn;
  std::vector<EdgeYawDynamics *> _e_yawdyn;
  std::vector<EdgeIneqJerkBounds *> _e_jerk;
  std::vector<EdgeIneqYawAccBounds *> _e_yawacc;
  std::vector<EdgeIneqSlackNonneg *> _e_snn;
  std::vector<std::vector<EdgeIneqObstacle *>> _e_obs;  // [slot][stage]
  std::vector<std::vector<EdgeIneqArena *>> _e_arena;   // [row][stage]
  std::vector<EdgeIneqVelMax *> _e_vel;
  std::vector<EdgeIneqTiltCone *> _e_tilt;
  std::vector<EdgeIneqAccDown *> _e_accd;
  std::vector<EdgeIneqAccUp *> _e_accu;
  std::vector<EdgeIneqYawRate *> _e_yawrate;

  bool _solved_once = false;
  SolveInfo _info;
};

void TrajFormulation::buildGraph() {
  const int N = _param->N;
  const int K = _param->K;
  const bool yaw = _param->yaw;

  const int id_p = 0;
  const int id_v = id_p + (N + 1);
  const int id_a = id_v + (N + 1);
  const int id_j = id_a + (N + 1);
  const int id_s = id_j + N;
  const int id_psi = id_s + N;
  const int id_om = id_psi + (yaw ? N + 1 : 0);
  const int id_al = id_om + (yaw ? N + 1 : 0);
  const int id_lag = id_al + (yaw ? N : 0);
  // Lagrange-multiplier ids: translation dynamics use id_lag + k, yaw dynamics id_lag + N + k.

  _vp.resize(N + 1);
  _vv.resize(N + 1);
  _va.resize(N + 1);
  _vj.resize(N);
  _vs.resize(N);
  for (int k = 0; k <= N; ++k) {
    _vp[k] = std::make_unique<Vertex_p>();
    _vv[k] = std::make_unique<Vertex_v>();
    _va[k] = std::make_unique<Vertex_a>();
    _vp[k]->setId(id_p + k);
    _vv[k]->setId(id_v + k);
    _va[k]->setId(id_a + k);
    _vp[k]->setToOrigin();
    _vv[k]->setToOrigin();
    _va[k]->setToOrigin();
    _optimizer->addVertex(_vp[k].get());
    _optimizer->addVertex(_vv[k].get());
    _optimizer->addVertex(_va[k].get());
    if (k == 0) {
      _vp[k]->setFixed(true);
      _vv[k]->setFixed(true);
      _va[k]->setFixed(true);
    }
  }
  for (int k = 0; k < N; ++k) {
    _vj[k] = std::make_unique<Vertex_j>();
    _vs[k] = std::make_unique<Vertex_s>();
    _vj[k]->setId(id_j + k);
    _vs[k]->setId(id_s + k);
    _vj[k]->setToOrigin();
    _vs[k]->setToOrigin();
    _optimizer->addVertex(_vj[k].get());
    _optimizer->addVertex(_vs[k].get());
    if (k == 0) _vs[k]->setFixed(true); // stage-0 slack unused, pinned to zero
  }
  if (yaw) {
    _vpsi.resize(N + 1);
    _vom.resize(N + 1);
    _val.resize(N);
    for (int k = 0; k <= N; ++k) {
      _vpsi[k] = std::make_unique<Vertex_psi>();
      _vom[k] = std::make_unique<Vertex_om>();
      _vpsi[k]->setId(id_psi + k);
      _vom[k]->setId(id_om + k);
      _vpsi[k]->setToOrigin();
      _vom[k]->setToOrigin();
      _optimizer->addVertex(_vpsi[k].get());
      _optimizer->addVertex(_vom[k].get());
      if (k == 0) {
        _vpsi[k]->setFixed(true);
        _vom[k]->setFixed(true);
      }
    }
    for (int k = 0; k < N; ++k) {
      _val[k] = std::make_unique<Vertex_al>();
      _val[k]->setId(id_al + k);
      _val[k]->setToOrigin();
      _optimizer->addVertex(_val[k].get());
    }
  }

  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  _e_obs.assign(K, {});
  _e_arena.assign(6, {});

  auto keep = [this](const auto &e) { _edges.push_back(e); };
  auto cost = [this](const auto &e) {
    _optimizer->addEdge(e.get());
    _edges.push_back(e);
    _cost_edges.push_back(e.get());
  };

  for (int k = 0; k < N; ++k) {
    // -- dynamics equality: x_{k+1} = f(x_k, j_k) --
    auto edyn = std::make_shared<EdgeDynamics>(_param);
    edyn->setVertex(0, _vp[k + 1].get());
    edyn->setVertex(1, _vv[k + 1].get());
    edyn->setVertex(2, _va[k + 1].get());
    edyn->setVertex(3, _vp[k].get());
    edyn->setVertex(4, _vv[k].get());
    edyn->setVertex(5, _va[k].get());
    edyn->setVertex(6, _vj[k].get());
    edyn->setVertexLagrangeMultiplierId(id_lag + k);
    _optimizer->addEdgeEq(edyn.get());
    keep(edyn);
    _e_dyn.push_back(edyn.get());

    // -- input regularisation r_u * ||j_k||^2 --
    auto ecost_u = std::make_shared<EdgeCostInput>();
    ecost_u->setVertex(0, _vj[k].get());
    ecost_u->setInformation(_param->r_jerk * I3);
    cost(ecost_u);
    if (k == 0 && _param->r_delta_input_first.maxCoeff() > 0.0) {
      auto edelta = std::make_shared<EdgeCostFirstInputDelta>(_param);
      edelta->setVertex(0, _vj[k].get());
      edelta->setInformation(_param->r_delta_input_first.asDiagonal());
      cost(edelta);
    }

    if (yaw) {
      auto eyd = std::make_shared<EdgeYawDynamics>(_param);
      eyd->setVertex(0, _vpsi[k + 1].get());
      eyd->setVertex(1, _vom[k + 1].get());
      eyd->setVertex(2, _vpsi[k].get());
      eyd->setVertex(3, _vom[k].get());
      eyd->setVertex(4, _val[k].get());
      eyd->setVertexLagrangeMultiplierId(id_lag + N + k);
      _optimizer->addEdgeEq(eyd.get());
      keep(eyd);
      _e_yawdyn.push_back(eyd.get());

      auto eau = std::make_shared<EdgeCostYawInput>();
      eau->setVertex(0, _val[k].get());
      eau->setInformation(Eigen::Matrix<double, 1, 1>(_param->w_yaw_acc));
      cost(eau);

      auto eab = std::make_shared<EdgeIneqYawAccBounds>(_param);
      eab->setVertex(0, _val[k].get());
      _optimizer->addEdgeIneq(eab.get());
      keep(eab);
      _e_yawacc.push_back(eab.get());
    }

    // -- hard jerk bounds --
    auto eineq_j = std::make_shared<EdgeIneqJerkBounds>(_param);
    eineq_j->setVertex(0, _vj[k].get());
    _optimizer->addEdgeIneq(eineq_j.get());
    keep(eineq_j);
    _e_jerk.push_back(eineq_j.get());

    if (k >= 1 && k <= N - 1) {
      auto eineq_s = std::make_shared<EdgeIneqSlackNonneg>();
      eineq_s->setVertex(0, _vs[k].get());
      _optimizer->addEdgeIneq(eineq_s.get());
      keep(eineq_s);
      _e_snn.push_back(eineq_s.get());

      auto ecost_s = std::make_shared<EdgeCostSlack>(_param);
      ecost_s->setVertex(0, _vs[k].get());
      Eigen::Vector3d sw(_param->w_slack_obstacle_l2, _param->w_slack_arena_l2, _param->w_slack_dynamic_l2);
      ecost_s->setInformation(sw.asDiagonal());
      cost(ecost_s);

      for (int j = 0; j < K; ++j) {
        auto eobs = std::make_shared<EdgeIneqObstacle>(_param, j);
        eobs->setVertex(0, _vp[k].get());
        eobs->setVertex(1, _vs[k].get());
        _optimizer->addEdgeIneq(eobs.get());
        keep(eobs);
        _e_obs[j].push_back(eobs.get());
      }

      int row = 0;
      for (int axis = 0; axis < 3; ++axis) {
        for (bool is_upper : {false, true}) {
          auto earena = std::make_shared<EdgeIneqArena>(_param, axis, is_upper);
          earena->setVertex(0, _vp[k].get());
          earena->setVertex(1, _vs[k].get());
          _optimizer->addEdgeIneq(earena.get());
          keep(earena);
          _e_arena[row++].push_back(earena.get());
        }
      }

      auto evel = std::make_shared<EdgeIneqVelMax>(_param);
      evel->setVertex(0, _vv[k].get());
      evel->setVertex(1, _vs[k].get());
      _optimizer->addEdgeIneq(evel.get());
      keep(evel);
      _e_vel.push_back(evel.get());

      auto etilt = std::make_shared<EdgeIneqTiltCone>(_param);
      etilt->setVertex(0, _va[k].get());
      etilt->setVertex(1, _vs[k].get());
      _optimizer->addEdgeIneq(etilt.get());
      keep(etilt);
      _e_tilt.push_back(etilt.get());

      auto eaccd = std::make_shared<EdgeIneqAccDown>(_param);
      eaccd->setVertex(0, _va[k].get());
      eaccd->setVertex(1, _vs[k].get());
      _optimizer->addEdgeIneq(eaccd.get());
      keep(eaccd);
      _e_accd.push_back(eaccd.get());

      auto eaccu = std::make_shared<EdgeIneqAccUp>(_param);
      eaccu->setVertex(0, _va[k].get());
      eaccu->setVertex(1, _vs[k].get());
      _optimizer->addEdgeIneq(eaccu.get());
      keep(eaccu);
      _e_accu.push_back(eaccu.get());

      if (yaw) {
        auto eyr = std::make_shared<EdgeIneqYawRate>(_param);
        eyr->setVertex(0, _vom[k].get());
        eyr->setVertex(1, _vs[k].get());
        _optimizer->addEdgeIneq(eyr.get());
        keep(eyr);
        _e_yawrate.push_back(eyr.get());

        auto ehd = std::make_shared<EdgeCostHeading>(_param, k);
        ehd->setVertex(0, _vv[k].get());
        ehd->setVertex(1, _vpsi[k].get());
        ehd->setVertex(2, _vom[k].get());
        ehd->setInformation(Eigen::Matrix3d::Identity());
        cost(ehd);
      }

      auto estate = std::make_shared<EdgeCostState>(_param, k);
      estate->setVertex(0, _vp[k].get());
      estate->setVertex(1, _vv[k].get());
      estate->setVertex(2, _va[k].get());
      estate->setInformation(stateInfo(_param->q_pos, _param->q_vel, _param->w_acc));
      cost(estate);
    }
  }

  // -- terminal cost at x_N --
  auto eterm = std::make_shared<EdgeCostState>(_param, N);
  eterm->setVertex(0, _vp[N].get());
  eterm->setVertex(1, _vv[N].get());
  eterm->setVertex(2, _va[N].get());
  eterm->setInformation(stateInfo(_param->q_pos_terminal, _param->q_vel_terminal, _param->w_acc));
  cost(eterm);
  if (yaw) {
    auto ehd = std::make_shared<EdgeCostHeading>(_param, N);
    ehd->setVertex(0, _vv[N].get());
    ehd->setVertex(1, _vpsi[N].get());
    ehd->setVertex(2, _vom[N].get());
    ehd->setInformation(Eigen::Matrix3d::Identity());
    cost(ehd);
  }
  // -- terminal smooth-hinge obstacle/arena penalty at p_N (no slack there) --
  for (int j = 0; j < K; ++j) {
    auto ehinge = std::make_shared<EdgeCostHingeObstacle>(_param, j);
    ehinge->setVertex(0, _vp[N].get());
    ehinge->setInformation(Eigen::Matrix<double, 1, 1>(_param->w_terminal_penalty));
    cost(ehinge);
  }
  for (int axis = 0; axis < 3; ++axis) {
    for (bool is_upper : {false, true}) {
      auto ehinge = std::make_shared<EdgeCostHingeArena>(_param, axis, is_upper);
      ehinge->setVertex(0, _vp[N].get());
      ehinge->setInformation(Eigen::Matrix<double, 1, 1>(_param->w_terminal_penalty));
      cost(ehinge);
    }
  }
}

void TrajFormulation::registerShiftSequences() {
  auto raw = [](const auto &owned) {
    using T = typename std::decay_t<decltype(owned)>::value_type::element_type;
    std::vector<T *> out;
    for (const auto &p : owned) out.push_back(p.get());
    return out;
  };
  auto &start = _optimizer->startPoint();
  start.addShiftSequence(raw(_vp));
  start.addShiftSequence(raw(_vv));
  start.addShiftSequence(raw(_va));
  start.addShiftSequence(raw(_vj));
  start.addShiftSequence(raw(_vs));
  start.addShiftSequence(_e_dyn);
  start.addShiftSequence(_e_jerk);
  start.addShiftSequence(_e_snn);
  for (auto &seq : _e_obs) start.addShiftSequence(seq);
  for (auto &seq : _e_arena) start.addShiftSequence(seq);
  start.addShiftSequence(_e_vel);
  start.addShiftSequence(_e_tilt);
  start.addShiftSequence(_e_accd);
  start.addShiftSequence(_e_accu);
  if (_param->yaw) {
    start.addShiftSequence(raw(_vpsi));
    start.addShiftSequence(raw(_vom));
    start.addShiftSequence(raw(_val));
    start.addShiftSequence(_e_yawdyn);
    start.addShiftSequence(_e_yawacc);
    start.addShiftSequence(_e_yawrate);
  }
  if (_settings.warm_start_terminal_rollout) {
    start.setPostShiftCallback([this]() {
      rolloutTail(_param->N - std::max(1, _optimizer->startPoint().options().shift.steps));
    });
  }
}

void TrajFormulation::seedColdStart() {
  // The cold start of all three backends: a dynamically consistent rollout of the same PD
  // reference-tracking law (PointMassJerkModel.guess_input / TrajectoryTNLP::coldStart). In the
  // infeasible-slalom experiment this rollout deliberately violates the obstacle inequalities.
  const int N = _param->N;
  const double dt = _param->dt;
  Eigen::Vector3d p = _param->x0.segment<3>(0);
  Eigen::Vector3d v = _param->x0.segment<3>(3);
  Eigen::Vector3d a = _param->x0.segment<3>(6);
  double psi = _param->psi0;
  double om = _param->om0;
  for (int k = 0; k <= N; ++k) {
    _vp[k]->setEstimate(p);
    _vv[k]->setEstimate(v);
    _va[k]->setEstimate(a);
    if (_param->yaw) {
      _vpsi[k]->setEstimate(Eigen::Matrix<double, 1, 1>(psi));
      _vom[k]->setEstimate(Eigen::Matrix<double, 1, 1>(om));
    }
    if (k >= N) continue;
    Eigen::Vector3d a_des = 1.2 * (_param->pref.col(k + 1) - p) + 2.0 * (_param->vref.col(k + 1) - v);
    const double an = a_des.norm();
    if (an > 4.0) a_des *= 4.0 / an;
    Eigen::Vector3d j = (a_des - a) / 0.3;
    for (int d = 0; d < 3; ++d) j[d] = std::clamp(j[d], -_param->jerk_max[d], _param->jerk_max[d]);
    _vj[k]->setEstimate(j);
    _vs[k]->setEstimate(Eigen::Vector3d::Constant(k == 0 ? 0.0 : 0.01));
    p += v * dt + a * (0.5 * dt * dt) + j * (dt * dt * dt / 6.0);
    v += a * dt + j * (0.5 * dt * dt);
    a += j * dt;
    if (_param->yaw) {
      const double ref = (k + 1 < _param->hdg_ref.size()) ? _param->hdg_ref[k + 1] : psi;
      const double al = std::clamp(6.0 * (ref - psi) - 3.5 * om, -_param->yaw_acc_max, _param->yaw_acc_max);
      _val[k]->setEstimate(Eigen::Matrix<double, 1, 1>(al));
      psi += om * dt + al * 0.5 * dt * dt;
      om += al * dt;
    }
  }
  if (_settings.feasible_start) liftSlacksToFeasible();
}

void TrajFormulation::shiftPrimal(int n) {
  // stage k takes stage k + n; the last n stages take the last one, then (optionally) the tail is
  // rolled out with the dynamics. Same rule as the IPOPT backends' primal warm start.
  const int N = _param->N;
  if (n <= 0) return;
  for (int k = 0; k <= N; ++k) {
    const int src = std::min(k + n, N);
    _vp[k]->setEstimate(_vp[src]->estimate());
    _vv[k]->setEstimate(_vv[src]->estimate());
    _va[k]->setEstimate(_va[src]->estimate());
    if (_param->yaw) {
      _vpsi[k]->setEstimate(_vpsi[src]->estimate());
      _vom[k]->setEstimate(_vom[src]->estimate());
    }
  }
  for (int k = 0; k < N; ++k) {
    const int src = std::min(k + n, N - 1);
    _vj[k]->setEstimate(_vj[src]->estimate());
    if (k > 0) _vs[k]->setEstimate(_vs[std::max(1, src)]->estimate());
    if (_param->yaw) _val[k]->setEstimate(_val[src]->estimate());
  }
  _vs[0]->setEstimate(Eigen::Vector3d::Zero());
  if (_settings.warm_start_terminal_rollout) rolloutTail(N - n);
}

void TrajFormulation::rolloutTail(int first_stage) {
  // x_{k+1} = f(x_k, u_k) for the stages that were replicated by a shift (u_k = last input)
  const int N = _param->N;
  const double dt = _param->dt;
  for (int k = std::max(0, first_stage); k < N; ++k) {
    const Eigen::Vector3d p = _vp[k]->estimate(), v = _vv[k]->estimate(), a = _va[k]->estimate();
    const Eigen::Vector3d j = _vj[k]->estimate();
    if (!_vp[k + 1]->fixed()) {
      _vp[k + 1]->setEstimate(p + v * dt + a * (0.5 * dt * dt) + j * (dt * dt * dt / 6.0));
      _vv[k + 1]->setEstimate(v + a * dt + j * (0.5 * dt * dt));
      _va[k + 1]->setEstimate(a + j * dt);
    }
    if (_param->yaw && !_vpsi[k + 1]->fixed()) {
      const double psi = _vpsi[k]->estimate()[0], om = _vom[k]->estimate()[0], al = _val[k]->estimate()[0];
      _vpsi[k + 1]->setEstimate(Eigen::Matrix<double, 1, 1>(psi + om * dt + al * 0.5 * dt * dt));
      _vom[k + 1]->setEstimate(Eigen::Matrix<double, 1, 1>(om + al * dt));
    }
  }
}

void TrajFormulation::liftSlacksToFeasible() {
  // Every state inequality of the OCP is softened by a stage slack, so a strictly feasible point
  // always exists: keep the primal guess and raise each stage slack above the largest violation
  // of its rows (+ margin); keep inputs strictly inside their boxes (optional, default off).
  const int N = _param->N;
  const int K = _param->K;
  const double m = std::max(1e-6, _settings.feasible_start_margin);
  const double eps = _param->obstacle_eps;
  const double tan2 = std::tan(_param->tilt_max) * std::tan(_param->tilt_max);
  const double g = _param->g;
  for (int k = 0; k < N; ++k) {
    Eigen::Vector3d j = _vj[k]->estimate();
    for (int d = 0; d < 3; ++d) j[d] = std::clamp(j[d], -0.999 * _param->jerk_max[d], 0.999 * _param->jerk_max[d]);
    _vj[k]->setEstimate(j);
    if (_param->yaw) {
      const double al = std::clamp(_val[k]->estimate()[0], -0.999 * _param->yaw_acc_max, 0.999 * _param->yaw_acc_max);
      _val[k]->setEstimate(Eigen::Matrix<double, 1, 1>(al));
    }
  }
  for (int k = 1; k < N; ++k) {
    const Eigen::Vector3d p = _vp[k]->estimate(), v = _vv[k]->estimate(), a = _va[k]->estimate();
    double so = 0.0, sa = 0.0, sd = 0.0;
    for (int jo = 0; jo < K; ++jo) {
      const double dist = std::sqrt((p - _param->obstacles.block<3, 1>(0, jo)).squaredNorm() + eps * eps);
      so = std::max(so, _param->obstacles(3, jo) - dist);
    }
    for (int d = 0; d < 3; ++d) {
      sa = std::max(sa, _param->arena_lb[d] - p[d]);
      sa = std::max(sa, p[d] - _param->arena_ub[d]);
    }
    sd = std::max(sd, (v.squaredNorm() - _param->vmax * _param->vmax) / (2.0 * _param->vmax));
    sd = std::max(sd, (a[0] * a[0] + a[1] * a[1] - tan2 * (g - a[2]) * (g - a[2])) / (2.0 * g));
    sd = std::max(sd, a[2] - _param->acc_down_max);
    sd = std::max(sd, -a[2] - _param->acc_up_max);
    if (_param->yaw) {
      const double om = _vom[k]->estimate()[0];
      sd = std::max(sd, (om * om - _param->yaw_rate_max * _param->yaw_rate_max) / (2.0 * _param->yaw_rate_max));
    }
    _vs[k]->setEstimate(Eigen::Vector3d(so + m, sa + m, sd + m));
  }
}

void TrajFormulation::setFixedInitialState() {
  _vp[0]->setEstimate(_param->x0.segment<3>(0));
  _vv[0]->setEstimate(_param->x0.segment<3>(3));
  _va[0]->setEstimate(_param->x0.segment<3>(6));
  _vs[0]->setEstimate(Eigen::Vector3d::Zero());
  if (_param->yaw) {
    _vpsi[0]->setEstimate(Eigen::Matrix<double, 1, 1>(_param->psi0));
    _vom[0]->setEstimate(Eigen::Matrix<double, 1, 1>(_param->om0));
  }
}

} // namespace

std::unique_ptr<TrajFormulationBase> makeTrajFormulation(std::shared_ptr<TrajParams> param,
                                                         const SolverSettings &settings) {
  const std::string opt = lower(settings.optimizer);
  if (opt == "ispd") return std::make_unique<TrajFormulation>(std::move(param), settings);
  throw std::runtime_error("the cg2o backend uses ISPD only (optimizer: ispd), got '" + settings.optimizer + "'");
}

} // namespace px4nmpc::cg2o
