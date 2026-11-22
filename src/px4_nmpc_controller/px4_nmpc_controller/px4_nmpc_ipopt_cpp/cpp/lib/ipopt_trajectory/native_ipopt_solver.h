#pragma once

#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>
#include <Eigen/Core>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace px4nmpc::ipoptcpp {

struct ProblemSettings {
  int N = 25;
  double dt = 0.1;
  int K = 8;
  double g = 9.81;
  double vmax = 3.0;
  double tilt_max = 0.5235987755982988;
  Eigen::Vector3d jerk_max = Eigen::Vector3d(12.0, 12.0, 8.0);
  double acc_up_max = 4.0;
  double acc_down_max = 3.0;
  Eigen::Vector3d q_pos = Eigen::Vector3d(12.0, 12.0, 16.0);
  Eigen::Vector3d q_vel = Eigen::Vector3d(2.0, 2.0, 3.0);
  Eigen::Vector3d q_pos_terminal = Eigen::Vector3d(72.0, 72.0, 96.0);
  Eigen::Vector3d q_vel_terminal = Eigen::Vector3d(8.0, 8.0, 12.0);
  double r_jerk = 0.02;
  double w_acc = 0.3;
  double w_slack_obstacle_l1 = 2000.0;
  double w_slack_obstacle_l2 = 4000.0;
  double w_slack_arena_l1 = 1000.0;
  double w_slack_arena_l2 = 2000.0;
  double w_slack_dynamic_l1 = 200.0;
  double w_slack_dynamic_l2 = 400.0;
  double w_terminal_penalty = 4000.0;
  Eigen::Vector3d r_delta_input_first = Eigen::Vector3d::Zero();
  // Optimised yaw (see PointMassJerkYawModel): adds the states [psi, omega] and the input alpha.
  bool yaw = false;
  double yaw_rate_max = 0.8;
  double yaw_acc_max = 2.0;
  double w_yaw_lateral = 1.0;
  double w_yaw_rate = 0.05;
  double w_yaw_acc = 0.05;
  int max_iter = 150;
  double tol = 1e-4;
  double acceptable_tol = 1e-3;
  int acceptable_iter = 3;
  double max_cpu_time = 0.25;
  std::string mu_strategy = "adaptive";
  std::string linear_solver = "mumps";
  std::string hessian_approximation = "exact";
  int print_level = 0;
  // Any further IPOPT option (ipopt_cpp.yaml `options:`), applied after the fields above.
  std::map<std::string, std::string> string_options;
  std::map<std::string, int> integer_options;
  std::map<std::string, double> numeric_options;
  // Receding-horizon warm start: "primal" (shifted primal guess, IPOPT initialises the duals) or
  // "primal_dual" (shifted primal + constraint and bound multipliers, warm_start_init_point=yes).
  std::string warm_start = "primal";
  // roll the replicated tail out with the dynamics after a shift (same as the cg2o backend)
  bool terminal_rollout = true;
};

struct SolveOutput {
  Eigen::MatrixXd X;
  Eigen::MatrixXd U;
  Eigen::MatrixXd S;
  int iterations = -1;
  double cost = 0.0;
  double solve_ms = 0.0;
  std::string status;
  bool success = false;
  bool warm_primal = false;
  bool warm_duals = false;
};

class TrajectoryTNLP final : public Ipopt::TNLP {
public:
  explicit TrajectoryTNLP(ProblemSettings settings);
  ~TrajectoryTNLP() override = default;

  void setProblemData(const Eigen::VectorXd& x0,
                      const Eigen::MatrixXd& pref,
                      const Eigen::MatrixXd& vref,
                      const Eigen::MatrixXd& obstacles,
                      const Eigen::Vector3d& arena_lb,
                      const Eigen::Vector3d& arena_ub,
                      const Eigen::Vector3d& uprev,
                      const Eigen::VectorXd& hdg_ref,
                      const Eigen::VectorXd& hdg_w,
                      int warm_shift);
  void resetWarmStart();

  const Eigen::MatrixXd& states() const { return X_result_; }
  const Eigen::MatrixXd& inputs() const { return U_result_; }
  const Eigen::MatrixXd& slacks() const { return S_result_; }
  double finalCost() const { return final_cost_; }
  bool hasSolution() const { return has_solution_; }
  //! a stored primal-dual point that the next solve can start from (needs warm_shift >= 0)
  bool dualWarmAvailable() const { return has_warm_ && has_warm_duals_ && warm_shift_ >= 0; }
  bool primalWarmAvailable() const { return has_warm_ && warm_shift_ >= 0; }
  void setUseWarmDuals(bool use) { use_warm_duals_ = use; }

  bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                    Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                    IndexStyleEnum& index_style) override;
  bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                       Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u) override;
  bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                          bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
                          Ipopt::Index m, bool init_lambda, Ipopt::Number* lambda) override;
  bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
              Ipopt::Number& obj_value) override;
  bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                   Ipopt::Number* grad_f) override;
  bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
              Ipopt::Index m, Ipopt::Number* g) override;
  bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                  Ipopt::Index m, Ipopt::Index nele_jac,
                  Ipopt::Index* iRow, Ipopt::Index* jCol,
                  Ipopt::Number* values) override;
  bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
              Ipopt::Number obj_factor, Ipopt::Index m,
              const Ipopt::Number* lambda, bool new_lambda,
              Ipopt::Index nele_hess, Ipopt::Index* iRow,
              Ipopt::Index* jCol, Ipopt::Number* values) override;
  void finalize_solution(Ipopt::SolverReturn status,
                         Ipopt::Index n, const Ipopt::Number* x,
                         const Ipopt::Number* z_L, const Ipopt::Number* z_U,
                         Ipopt::Index m, const Ipopt::Number* g,
                         const Ipopt::Number* lambda,
                         Ipopt::Number obj_value,
                         const Ipopt::IpoptData* ip_data,
                         Ipopt::IpoptCalculatedQuantities* ip_cq) override;

  int nVars() const { return n_vars_; }
  int nCons() const { return n_cons_; }
  int nnzJac() const { return static_cast<int>(jac_i_.size()); }
  int nnzHess() const { return static_cast<int>(hess_i_.size()); }

private:
  // nx_ = 9 (+2 with yaw: psi, omega), nu_ = 3 (+1: alpha)
  int idxX(int k, int i) const { return nx_ * k + i; }
  int idxU(int k, int i) const { return nx_ * (settings_.N + 1) + nu_ * k + i; }
  int idxS(int k, int i) const { return nx_ * (settings_.N + 1) + nu_ * settings_.N + 3 * k + i; }
  int nDyn() const { return settings_.yaw ? 5 : 4; }  // soft dynamic-limit rows per stage

  void buildJacobianStructure();
  void buildHessianStructure();
  void coldStart(std::vector<double>& z) const;
  void warmStart(std::vector<double>& z) const;
  void rolloutTail(std::vector<double>& z, int first_stage) const;
  int shiftedVariable(int index, int shift) const;    // index of the variable that index takes after a shift
  int shiftedConstraint(int row, int shift) const;
  void unpack(const Ipopt::Number* x, Eigen::MatrixXd& X, Eigen::MatrixXd& U, Eigen::MatrixXd& S) const;
  double objective(const Ipopt::Number* x) const;
  double headingCost(const Ipopt::Number* x, int k) const;
  void headingGradient(const Ipopt::Number* x, int k, Ipopt::Number* grad) const;

  ProblemSettings settings_;
  int nx_ = 9;
  int nu_ = 3;
  int n_vars_ = 0;
  int n_cons_ = 0;
  std::vector<Ipopt::Index> jac_i_;
  std::vector<Ipopt::Index> jac_j_;
  std::vector<Ipopt::Index> hess_i_;
  std::vector<Ipopt::Index> hess_j_;

  Eigen::VectorXd x0_;
  Eigen::VectorXd hdg_ref_;
  Eigen::VectorXd hdg_w_;
  Eigen::MatrixXd pref_;
  Eigen::MatrixXd vref_;
  Eigen::MatrixXd obstacles_;
  Eigen::Vector3d arena_lb_ = Eigen::Vector3d::Constant(-1e4);
  Eigen::Vector3d arena_ub_ = Eigen::Vector3d::Constant(1e4);
  Eigen::Vector3d uprev_ = Eigen::Vector3d::Zero();
  int warm_shift_ = -1;

  bool has_warm_ = false;
  bool has_warm_duals_ = false;
  bool use_warm_duals_ = false;
  std::vector<double> lambda_warm_;
  std::vector<double> zl_warm_;
  std::vector<double> zu_warm_;
  Eigen::MatrixXd X_warm_;
  Eigen::MatrixXd U_warm_;
  Eigen::MatrixXd S_warm_;

  bool has_solution_ = false;
  Eigen::MatrixXd X_result_;
  Eigen::MatrixXd U_result_;
  Eigen::MatrixXd S_result_;
  double final_cost_ = 0.0;
};

class NativeIpoptTrajectorySolver {
public:
  explicit NativeIpoptTrajectorySolver(ProblemSettings settings);
  SolveOutput solve(const Eigen::VectorXd& x0,
                    const Eigen::MatrixXd& pref,
                    const Eigen::MatrixXd& vref,
                    const Eigen::MatrixXd& obstacles,
                    const Eigen::Vector3d& arena_lb,
                    const Eigen::Vector3d& arena_ub,
                    const Eigen::Vector3d& uprev,
                    const Eigen::VectorXd& hdg_ref,
                    const Eigen::VectorXd& hdg_w,
                    int warm_shift);
  void resetWarmStart();
  int nVars() const;
  int nCons() const;

private:
  static std::string statusString(Ipopt::ApplicationReturnStatus status);
  static bool statusSuccess(Ipopt::ApplicationReturnStatus status);

  ProblemSettings settings_;
  Ipopt::SmartPtr<Ipopt::IpoptApplication> app_;
  Ipopt::SmartPtr<TrajectoryTNLP> nlp_;
};

} // namespace px4nmpc::ipoptcpp
