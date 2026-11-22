#ifndef PX4_NMPC_CG2O_MPC_PARAMETERS_H_
#define PX4_NMPC_CG2O_MPC_PARAMETERS_H_




#include <Eigen/Dense>

#ifdef CG2O_USE_LOCAL_G2O_NAMESPACE
namespace g2o {}
namespace cg2o = ::g2o;
#endif

namespace px4nmpc::cg2o {

// Mirrors the numeric content of px4_nmpc_core.ocp.NmpcOcpSolver for the
// 'trajectory' control interface (point-mass/jerk model): same weights,
// same soft-constraint semantics, same K-nearest-obstacle parameter slots.
// One instance is shared (by pointer) across every vertex/edge of one
// factor graph and is refreshed in place before each solve() call, so the
// graph topology itself never has to be rebuilt between NMPC solves.
class TrajParams {
public:
  // ---- horizon (fixed at construction) ---------------------------------
  int N = 25;
  double dt = 0.10;
  int K = 8; // max_obstacles

  // ---- vehicle / constraint limits --------------------------------------
  double g = 9.81;
  double vmax = 3.0;
  double tilt_max = 0.5236; // 30 deg
  Eigen::Vector3d jerk_max = Eigen::Vector3d(12.0, 12.0, 8.0);
  double acc_up_max = 4.0;
  double acc_down_max = 3.0;
  double obstacle_eps = 0.05; // matches ocp.py's `eps` softening term

  // ---- weights -----------------------------------------------------------
  Eigen::Vector3d q_pos = Eigen::Vector3d(12.0, 12.0, 16.0);
  Eigen::Vector3d q_vel = Eigen::Vector3d(2.0, 2.0, 3.0);
  Eigen::Vector3d q_pos_terminal = Eigen::Vector3d(72.0, 72.0, 96.0);
  Eigen::Vector3d q_vel_terminal = Eigen::Vector3d(8.0, 8.0, 12.0);
  double r_jerk = 0.02;
  double w_acc = 0.3;
  // Exact IPOPT slack objective l1*s + l2*s^2 is represented by
  // l2*(s + l1/(2*l2))^2 up to an additive constant.
  double w_slack_obstacle_l1 = 2000.0;
  double w_slack_obstacle_l2 = 4000.0;
  double w_slack_arena_l1 = 1000.0;
  double w_slack_arena_l2 = 2000.0;
  double w_slack_dynamic_l1 = 200.0;
  double w_slack_dynamic_l2 = 400.0;
  double w_terminal_penalty = 4000.0;

  // ---- optimised yaw (PointMassJerkYawModel); `yaw=false` leaves the graph exactly as before ----
  bool yaw = false;
  double yaw_rate_max = 0.8;   // [rad/s], soft (shares the dynamic-limit slack)
  double yaw_acc_max = 2.0;    // [rad/s^2], hard box on the input
  double w_yaw_lateral = 1.0;
  double w_yaw_rate = 0.05;
  double w_yaw_acc = 0.05;

  // ---- per-solve data (refreshed every solve_ocp() call) -----------------
  Eigen::Matrix<double, 9, 1> x0 = Eigen::Matrix<double, 9, 1>::Zero();
  double psi0 = 0.0;                // yaw states of the fixed initial vertices (yaw only)
  double om0 = 0.0;
  Eigen::VectorXd hdg_ref;          // N+1 heading references [rad, unwrapped] (yaw only)
  Eigen::VectorXd hdg_w;            // N+1 heading weights (yaw only)
  Eigen::MatrixXd pref;             // 3 x (N+1)
  Eigen::MatrixXd vref;             // 3 x (N+1)
  Eigen::MatrixXd obstacles;        // 4 x K: rows [n, e, d, radius]
  Eigen::Vector3d arena_lb = Eigen::Vector3d(-1.0e4, -1.0e4, -1.0e4);
  Eigen::Vector3d arena_ub = Eigen::Vector3d(1.0e4, 1.0e4, 1.0e4);
  Eigen::Vector3d uprev = Eigen::Vector3d::Zero();
  Eigen::Vector3d r_delta_input_first = Eigen::Vector3d::Zero();

  void resizeHorizonBuffers() {
    pref = Eigen::MatrixXd::Zero(3, N + 1);
    vref = Eigen::MatrixXd::Zero(3, N + 1);
    hdg_ref = Eigen::VectorXd::Zero(N + 1);
    hdg_w = Eigen::VectorXd::Zero(N + 1);
    obstacles = Eigen::MatrixXd::Zero(4, K);
    for (int j = 0; j < K; ++j) {
      obstacles(3, j) = 0.1; // harmless default radius, matches ocp.py
    }
  }
};

} // namespace px4nmpc::cg2o

#endif // PX4_NMPC_CG2O_MPC_PARAMETERS_H_
