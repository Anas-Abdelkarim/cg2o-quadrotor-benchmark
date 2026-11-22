#include "native_ipopt_solver.h"

#include <IpSolveStatistics.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace px4nmpc::ipoptcpp {
namespace {
constexpr double kInf = 2.0e19;
constexpr double kObsEps = 0.05;

inline double sq(double x) { return x * x; }
}

TrajectoryTNLP::TrajectoryTNLP(ProblemSettings settings)
    : settings_(std::move(settings)) {
  if (settings_.N < 2 || settings_.dt <= 0.0 || settings_.K < 1) {
    throw std::runtime_error("N >= 2, dt > 0 and K >= 1 are required");
  }
  nx_ = settings_.yaw ? 11 : 9;
  nu_ = settings_.yaw ? 4 : 3;
  n_vars_ = nx_ * (settings_.N + 1) + (nu_ + 3) * settings_.N;
  n_cons_ = nx_ + nx_ * settings_.N + (settings_.N - 1) * (settings_.K + 6 + nDyn());
  x0_ = Eigen::VectorXd::Zero(nx_);
  hdg_ref_ = Eigen::VectorXd::Zero(settings_.N + 1);
  hdg_w_ = Eigen::VectorXd::Zero(settings_.N + 1);
  pref_.resize(3, settings_.N + 1);
  vref_.resize(3, settings_.N + 1);
  obstacles_.resize(4, settings_.K);
  X_result_.resize(nx_, settings_.N + 1);
  U_result_.resize(nu_, settings_.N);
  S_result_.resize(3, settings_.N);
  X_warm_.resize(nx_, settings_.N + 1);
  U_warm_.resize(nu_, settings_.N);
  S_warm_.resize(3, settings_.N);
  buildJacobianStructure();
  buildHessianStructure();
}

void TrajectoryTNLP::setProblemData(const Eigen::VectorXd& x0,
                                    const Eigen::MatrixXd& pref,
                                    const Eigen::MatrixXd& vref,
                                    const Eigen::MatrixXd& obstacles,
                                    const Eigen::Vector3d& arena_lb,
                                    const Eigen::Vector3d& arena_ub,
                                    const Eigen::Vector3d& uprev,
                                    const Eigen::VectorXd& hdg_ref,
                                    const Eigen::VectorXd& hdg_w,
                                    int warm_shift) {
  if (x0.size() != nx_ ||
      pref.rows() != 3 || pref.cols() != settings_.N + 1 ||
      vref.rows() != 3 || vref.cols() != settings_.N + 1 ||
      obstacles.rows() != 4 || obstacles.cols() != settings_.K ||
      (settings_.yaw && (hdg_ref.size() != settings_.N + 1 || hdg_w.size() != settings_.N + 1))) {
    throw std::runtime_error("invalid native IPOPT problem dimensions");
  }
  x0_ = x0;
  if (settings_.yaw) { hdg_ref_ = hdg_ref; hdg_w_ = hdg_w; }
  pref_ = pref;
  vref_ = vref;
  obstacles_ = obstacles;
  arena_lb_ = arena_lb;
  arena_ub_ = arena_ub;
  uprev_ = uprev;
  warm_shift_ = warm_shift;
  has_solution_ = false;
}

void TrajectoryTNLP::resetWarmStart() {
  has_warm_ = false;
  has_warm_duals_ = false;
  warm_shift_ = -1;
}

bool TrajectoryTNLP::get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                                  Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                                  IndexStyleEnum& index_style) {
  n = n_vars_;
  m = n_cons_;
  nnz_jac_g = static_cast<Ipopt::Index>(jac_i_.size());
  nnz_h_lag = static_cast<Ipopt::Index>(hess_i_.size());
  index_style = TNLP::C_STYLE;
  return true;
}

bool TrajectoryTNLP::get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                                     Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u) {
  if (n != n_vars_ || m != n_cons_) return false;
  for (int i = 0; i < n_vars_; ++i) { x_l[i] = -kInf; x_u[i] = kInf; }
  for (int k = 0; k < settings_.N; ++k) {
    for (int d = 0; d < 3; ++d) {
      x_l[idxU(k,d)] = -settings_.jerk_max[d];
      x_u[idxU(k,d)] =  settings_.jerk_max[d];
      x_l[idxS(k,d)] = 0.0;
      x_u[idxS(k,d)] = (k == 0) ? 0.0 : kInf;
    }
    if (settings_.yaw) { x_l[idxU(k,3)] = -settings_.yaw_acc_max; x_u[idxU(k,3)] = settings_.yaw_acc_max; }
  }
  int r = 0;
  for (int i = 0; i < nx_; ++i, ++r) g_l[r] = g_u[r] = 0.0;
  for (int k = 0; k < settings_.N; ++k)
    for (int i = 0; i < nx_; ++i, ++r) g_l[r] = g_u[r] = 0.0;
  for (int k = 1; k < settings_.N; ++k) {
    for (int j = 0; j < settings_.K; ++j, ++r) { g_l[r] = 0.0; g_u[r] = kInf; }
    for (int j = 0; j < 6; ++j, ++r) { g_l[r] = 0.0; g_u[r] = kInf; }
    for (int j = 0; j < nDyn(); ++j, ++r) { g_l[r] = -kInf; g_u[r] = 0.0; }
  }
  return r == n_cons_;
}

void TrajectoryTNLP::coldStart(std::vector<double>& z) const {
  std::fill(z.begin(), z.end(), 0.0);
  Eigen::Matrix<double, 9, 1> x = x0_.head<9>();
  double psi = settings_.yaw ? x0_[9] : 0.0, om = settings_.yaw ? x0_[10] : 0.0;
  for (int i = 0; i < nx_; ++i) z[idxX(0,i)] = x0_[i];
  const double dt = settings_.dt;
  for (int k = 0; k < settings_.N; ++k) {
    Eigen::Vector3d a_des = 1.2 * (pref_.col(k + 1) - x.segment<3>(0))
                          + 2.0 * (vref_.col(k + 1) - x.segment<3>(3));
    const double an = a_des.norm();
    if (an > 4.0) a_des *= 4.0 / an;
    Eigen::Vector3d j = (a_des - x.segment<3>(6)) / 0.3;
    for (int d = 0; d < 3; ++d) {
      j[d] = std::clamp(j[d], -settings_.jerk_max[d], settings_.jerk_max[d]);
      z[idxU(k,d)] = j[d];
      z[idxS(k,d)] = (k == 0) ? 0.0 : 0.01;
    }
    Eigen::Matrix<double, 9, 1> xn;
    xn.segment<3>(0) = x.segment<3>(0) + x.segment<3>(3)*dt + x.segment<3>(6)*(0.5*dt*dt) + j*(dt*dt*dt/6.0);
    xn.segment<3>(3) = x.segment<3>(3) + x.segment<3>(6)*dt + j*(0.5*dt*dt);
    xn.segment<3>(6) = x.segment<3>(6) + j*dt;
    x = xn;
    for (int i = 0; i < 9; ++i) z[idxX(k+1,i)] = x[i];
    if (settings_.yaw) {
      // critically damped PD law towards the heading reference, rolled out like the translation
      const double al = std::clamp(6.0 * (hdg_ref_[k + 1] - psi) - 3.5 * om, -settings_.yaw_acc_max, settings_.yaw_acc_max);
      z[idxU(k,3)] = al;
      const double psi_n = psi + om * dt + al * 0.5 * dt * dt;
      om = om + al * dt;
      psi = psi_n;
      z[idxX(k+1,9)] = psi;
      z[idxX(k+1,10)] = om;
    }
  }
}

void TrajectoryTNLP::warmStart(std::vector<double>& z) const {
  const int shift = std::clamp(warm_shift_, 0, settings_.N - 1);
  for (int k = 0; k <= settings_.N; ++k) {
    int src = std::min(k + shift, settings_.N);
    for (int i = 0; i < nx_; ++i) z[idxX(k,i)] = X_warm_(i,src);
  }
  for (int k = 0; k < settings_.N; ++k) {
    int src = std::min(k + shift, settings_.N - 1);
    for (int i = 0; i < nu_; ++i) z[idxU(k,i)] = U_warm_(i,src);
    for (int i = 0; i < 3; ++i) z[idxS(k,i)] = (k == 0) ? 0.0 : std::max(0.0, S_warm_(i,src));
  }
  for (int i = 0; i < nx_; ++i) z[idxX(0,i)] = x0_[i];
  if (settings_.terminal_rollout && shift > 0) rolloutTail(z, settings_.N - shift);
}

void TrajectoryTNLP::rolloutTail(std::vector<double>& z, int first_stage) const {
  // x_{k+1} = f(x_k, u_k) for the stages that were replicated by the shift (same as cg2o)
  const double dt = settings_.dt;
  for (int k = std::max(0, first_stage); k < settings_.N; ++k) {
    for (int d = 0; d < 3; ++d) {
      const double p = z[idxX(k,d)], v = z[idxX(k,3+d)], a = z[idxX(k,6+d)], j = z[idxU(k,d)];
      z[idxX(k+1,d)] = p + v*dt + a*(0.5*dt*dt) + j*(dt*dt*dt/6.0);
      z[idxX(k+1,3+d)] = v + a*dt + j*(0.5*dt*dt);
      z[idxX(k+1,6+d)] = a + j*dt;
    }
    if (settings_.yaw) {
      const double psi = z[idxX(k,9)], om = z[idxX(k,10)], al = z[idxU(k,3)];
      z[idxX(k+1,9)] = psi + om*dt + al*0.5*dt*dt;
      z[idxX(k+1,10)] = om + al*dt;
    }
  }
}

int TrajectoryTNLP::shiftedVariable(int index, int shift) const {
  const int N = settings_.N;
  const int nX = nx_ * (N + 1), nU = nu_ * N;
  if (index < nX) {
    const int k = index / nx_, i = index % nx_;
    return idxX(std::min(k + shift, N), i);
  }
  if (index < nX + nU) {
    const int k = (index - nX) / nu_, i = (index - nX) % nu_;
    return idxU(std::min(k + shift, N - 1), i);
  }
  const int k = (index - nX - nU) / 3, i = (index - nX - nU) % 3;
  return idxS(std::min(k + shift, N - 1), i);
}

int TrajectoryTNLP::shiftedConstraint(int row, int shift) const {
  // rows: [initial condition (nx)] [dynamics k = 0..N-1 (nx each)] [path rows k = 1..N-1 (K + 6 + nDyn each)]
  const int N = settings_.N;
  if (row < nx_) return row;
  const int dyn0 = nx_, path0 = nx_ + nx_ * N;
  if (row < path0) {
    const int k = (row - dyn0) / nx_, i = (row - dyn0) % nx_;
    return dyn0 + std::min(k + shift, N - 1) * nx_ + i;
  }
  const int w = settings_.K + 6 + nDyn();
  const int k = 1 + (row - path0) / w, i = (row - path0) % w;
  return path0 + (std::min(k + shift, N - 1) - 1) * w + i;
}

bool TrajectoryTNLP::get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                                        bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
                                        Ipopt::Index m, bool init_lambda, Ipopt::Number* lambda) {
  if (n != n_vars_ || m != n_cons_ || !init_x) return false;
  std::vector<double> z(n_vars_, 0.0);
  if (has_warm_ && warm_shift_ >= 0) warmStart(z); else coldStart(z);
  std::copy(z.begin(), z.end(), x);
  if (init_z || init_lambda) {
    // only requested with warm_start_init_point=yes, which solve() sets only when a stored
    // primal-dual point exists (primal_dual warm start)
    if (!(use_warm_duals_ && dualWarmAvailable())) return false;
    const int shift = std::clamp(warm_shift_, 0, settings_.N - 1);
    if (init_z) {
      for (int i = 0; i < n_vars_; ++i) {
        const int src = shiftedVariable(i, shift);
        z_L[i] = zl_warm_[src];
        z_U[i] = zu_warm_[src];
      }
    }
    if (init_lambda) {
      for (int r = 0; r < n_cons_; ++r) lambda[r] = lambda_warm_[shiftedConstraint(r, shift)];
    }
  }
  return true;
}

// Heading cost of stage k (identical to PointMassJerkYawModel.heading_cost):
//   hw_k (psi - psi_ref_k)^2 + w_lat (vN sin psi - vE cos psi)^2 + w_rate omega^2
double TrajectoryTNLP::headingCost(const Ipopt::Number* x, int k) const {
  const double psi = x[idxX(k,9)], om = x[idxX(k,10)];
  const double lat = x[idxX(k,3)] * std::sin(psi) - x[idxX(k,4)] * std::cos(psi);
  return hdg_w_[k] * sq(psi - hdg_ref_[k]) + settings_.w_yaw_lateral * sq(lat) + settings_.w_yaw_rate * sq(om);
}

void TrajectoryTNLP::headingGradient(const Ipopt::Number* x, int k, Ipopt::Number* grad) const {
  const double psi = x[idxX(k,9)], om = x[idxX(k,10)];
  const double vN = x[idxX(k,3)], vE = x[idxX(k,4)];
  const double s = std::sin(psi), c = std::cos(psi);
  const double lat = vN * s - vE * c;
  const double w = settings_.w_yaw_lateral;
  grad[idxX(k,3)] += 2.0 * w * lat * s;
  grad[idxX(k,4)] -= 2.0 * w * lat * c;
  grad[idxX(k,9)] += 2.0 * hdg_w_[k] * (psi - hdg_ref_[k]) + 2.0 * w * lat * (vN * c + vE * s);
  grad[idxX(k,10)] += 2.0 * settings_.w_yaw_rate * om;
}

double TrajectoryTNLP::objective(const Ipopt::Number* x) const {
  double J = 0.0;
  for (int k = 0; k < settings_.N; ++k) {
    for (int d = 0; d < 3; ++d) {
      J += settings_.q_pos[d] * sq(x[idxX(k,d)] - pref_(d,k));
      J += settings_.q_vel[d] * sq(x[idxX(k,3+d)] - vref_(d,k));
      J += settings_.w_acc * sq(x[idxX(k,6+d)]);
      J += settings_.r_jerk * sq(x[idxU(k,d)]);
      if (k == 0) J += settings_.r_delta_input_first[d] * sq(x[idxU(k,d)] - uprev_[d]);
    }
    if (k >= 1) {
      const double so=x[idxS(k,0)], sa=x[idxS(k,1)], sd=x[idxS(k,2)];
      J += settings_.w_slack_obstacle_l1*so + settings_.w_slack_obstacle_l2*so*so;
      J += settings_.w_slack_arena_l1*sa + settings_.w_slack_arena_l2*sa*sa;
      J += settings_.w_slack_dynamic_l1*sd + settings_.w_slack_dynamic_l2*sd*sd;
    }
    if (settings_.yaw) {
      J += headingCost(x, k);
      J += settings_.w_yaw_acc * sq(x[idxU(k,3)]);
    }
  }
  const int N = settings_.N;
  if (settings_.yaw) J += headingCost(x, N);
  for (int d = 0; d < 3; ++d) {
    J += settings_.q_pos_terminal[d] * sq(x[idxX(N,d)] - pref_(d,N));
    J += settings_.q_vel_terminal[d] * sq(x[idxX(N,3+d)] - vref_(d,N));
    J += settings_.w_acc * sq(x[idxX(N,6+d)]);
  }
  Eigen::Vector3d p;
  for (int d=0; d<3; ++d) p[d]=x[idxX(N,d)];
  for (int j=0; j<settings_.K; ++j) {
    const Eigen::Vector3d diff = p - obstacles_.block<3,1>(0,j);
    const double h = std::sqrt(diff.squaredNorm() + kObsEps*kObsEps) - obstacles_(3,j);
    if (h < 0.0) J += settings_.w_terminal_penalty*h*h;
  }
  for (int d=0; d<3; ++d) {
    double h = p[d]-arena_lb_[d]; if (h<0.0) J += settings_.w_terminal_penalty*h*h;
    h = arena_ub_[d]-p[d]; if (h<0.0) J += settings_.w_terminal_penalty*h*h;
  }
  return J;
}

bool TrajectoryTNLP::eval_f(Ipopt::Index, const Ipopt::Number* x, bool, Ipopt::Number& obj_value) {
  obj_value = objective(x); return true;
}

bool TrajectoryTNLP::eval_grad_f(Ipopt::Index, const Ipopt::Number* x, bool, Ipopt::Number* grad) {
  std::fill(grad, grad+n_vars_, 0.0);
  for (int k=0;k<settings_.N;++k) {
    for (int d=0;d<3;++d) {
      grad[idxX(k,d)] += 2.0*settings_.q_pos[d]*(x[idxX(k,d)]-pref_(d,k));
      grad[idxX(k,3+d)] += 2.0*settings_.q_vel[d]*(x[idxX(k,3+d)]-vref_(d,k));
      grad[idxX(k,6+d)] += 2.0*settings_.w_acc*x[idxX(k,6+d)];
      grad[idxU(k,d)] += 2.0*settings_.r_jerk*x[idxU(k,d)];
      if (k==0) grad[idxU(k,d)] += 2.0*settings_.r_delta_input_first[d]*(x[idxU(k,d)]-uprev_[d]);
    }
    if (k>=1) {
      grad[idxS(k,0)] += settings_.w_slack_obstacle_l1 + 2.0*settings_.w_slack_obstacle_l2*x[idxS(k,0)];
      grad[idxS(k,1)] += settings_.w_slack_arena_l1 + 2.0*settings_.w_slack_arena_l2*x[idxS(k,1)];
      grad[idxS(k,2)] += settings_.w_slack_dynamic_l1 + 2.0*settings_.w_slack_dynamic_l2*x[idxS(k,2)];
    }
    if (settings_.yaw) {
      headingGradient(x, k, grad);
      grad[idxU(k,3)] += 2.0*settings_.w_yaw_acc*x[idxU(k,3)];
    }
  }
  const int N=settings_.N;
  if (settings_.yaw) headingGradient(x, N, grad);
  for (int d=0;d<3;++d) {
    grad[idxX(N,d)] += 2.0*settings_.q_pos_terminal[d]*(x[idxX(N,d)]-pref_(d,N));
    grad[idxX(N,3+d)] += 2.0*settings_.q_vel_terminal[d]*(x[idxX(N,3+d)]-vref_(d,N));
    grad[idxX(N,6+d)] += 2.0*settings_.w_acc*x[idxX(N,6+d)];
  }
  Eigen::Vector3d p; for(int d=0;d<3;++d)p[d]=x[idxX(N,d)];
  for(int j=0;j<settings_.K;++j){
    Eigen::Vector3d diff=p-obstacles_.block<3,1>(0,j);
    double dist=std::sqrt(diff.squaredNorm()+kObsEps*kObsEps);
    double h=dist-obstacles_(3,j);
    if(h<0.0){ Eigen::Vector3d dh=diff/dist; for(int d=0;d<3;++d)grad[idxX(N,d)]+=2.0*settings_.w_terminal_penalty*h*dh[d]; }
  }
  for(int d=0;d<3;++d){
    double h=p[d]-arena_lb_[d]; if(h<0.0)grad[idxX(N,d)]+=2.0*settings_.w_terminal_penalty*h;
    h=arena_ub_[d]-p[d]; if(h<0.0)grad[idxX(N,d)]-=2.0*settings_.w_terminal_penalty*h;
  }
  return true;
}

bool TrajectoryTNLP::eval_g(Ipopt::Index, const Ipopt::Number* x, bool, Ipopt::Index, Ipopt::Number* g) {
  int r=0;
  for(int i=0;i<nx_;++i) g[r++]=x[idxX(0,i)]-x0_[i];
  const double dt=settings_.dt;
  for(int k=0;k<settings_.N;++k){
    for(int d=0;d<3;++d) g[r++]=x[idxX(k+1,d)]-(x[idxX(k,d)]+x[idxX(k,3+d)]*dt+x[idxX(k,6+d)]*0.5*dt*dt+x[idxU(k,d)]*dt*dt*dt/6.0);
    for(int d=0;d<3;++d) g[r++]=x[idxX(k+1,3+d)]-(x[idxX(k,3+d)]+x[idxX(k,6+d)]*dt+x[idxU(k,d)]*0.5*dt*dt);
    for(int d=0;d<3;++d) g[r++]=x[idxX(k+1,6+d)]-(x[idxX(k,6+d)]+x[idxU(k,d)]*dt);
    if (settings_.yaw) {
      g[r++]=x[idxX(k+1,9)]-(x[idxX(k,9)]+x[idxX(k,10)]*dt+x[idxU(k,3)]*0.5*dt*dt);
      g[r++]=x[idxX(k+1,10)]-(x[idxX(k,10)]+x[idxU(k,3)]*dt);
    }
  }
  const double tan2=std::tan(settings_.tilt_max)*std::tan(settings_.tilt_max);
  for(int k=1;k<settings_.N;++k){
    Eigen::Vector3d p; for(int d=0;d<3;++d)p[d]=x[idxX(k,d)];
    for(int j=0;j<settings_.K;++j){
      Eigen::Vector3d diff=p-obstacles_.block<3,1>(0,j);
      g[r++]=std::sqrt(diff.squaredNorm()+kObsEps*kObsEps)-obstacles_(3,j)+x[idxS(k,0)];
    }
    for(int d=0;d<3;++d) g[r++]=p[d]-arena_lb_[d]+x[idxS(k,1)];
    for(int d=0;d<3;++d) g[r++]=arena_ub_[d]-p[d]+x[idxS(k,1)];
    double v2=0.0; for(int d=0;d<3;++d)v2+=sq(x[idxX(k,3+d)]);
    g[r++]=(v2-settings_.vmax*settings_.vmax)/(2.0*settings_.vmax)-x[idxS(k,2)];
    const double a0=x[idxX(k,6)],a1=x[idxX(k,7)],a2=x[idxX(k,8)];
    g[r++]=(a0*a0+a1*a1-tan2*sq(settings_.g-a2))/(2.0*settings_.g)-x[idxS(k,2)];
    g[r++]=a2-settings_.acc_down_max-x[idxS(k,2)];
    g[r++]=-a2-settings_.acc_up_max-x[idxS(k,2)];
    if (settings_.yaw) g[r++]=(sq(x[idxX(k,10)])-sq(settings_.yaw_rate_max))/(2.0*settings_.yaw_rate_max)-x[idxS(k,2)];
  }
  return r==n_cons_;
}

void TrajectoryTNLP::buildJacobianStructure() {
  jac_i_.clear(); jac_j_.clear();
  auto add=[this](int r,int c){jac_i_.push_back(r);jac_j_.push_back(c);};
  int r=0;
  for(int i=0;i<nx_;++i,++r)add(r,idxX(0,i));
  for(int k=0;k<settings_.N;++k){
    for(int d=0;d<3;++d,++r){ add(r,idxX(k+1,d)); add(r,idxX(k,d)); add(r,idxX(k,3+d)); add(r,idxX(k,6+d)); add(r,idxU(k,d)); }
    for(int d=0;d<3;++d,++r){ add(r,idxX(k+1,3+d)); add(r,idxX(k,3+d)); add(r,idxX(k,6+d)); add(r,idxU(k,d)); }
    for(int d=0;d<3;++d,++r){ add(r,idxX(k+1,6+d)); add(r,idxX(k,6+d)); add(r,idxU(k,d)); }
    if (settings_.yaw) {
      add(r,idxX(k+1,9)); add(r,idxX(k,9)); add(r,idxX(k,10)); add(r,idxU(k,3)); ++r;
      add(r,idxX(k+1,10)); add(r,idxX(k,10)); add(r,idxU(k,3)); ++r;
    }
  }
  for(int k=1;k<settings_.N;++k){
    for(int j=0;j<settings_.K;++j,++r){ for(int d=0;d<3;++d)add(r,idxX(k,d)); add(r,idxS(k,0)); }
    for(int d=0;d<3;++d,++r){add(r,idxX(k,d));add(r,idxS(k,1));}
    for(int d=0;d<3;++d,++r){add(r,idxX(k,d));add(r,idxS(k,1));}
    for(int d=0;d<3;++d)add(r,idxX(k,3+d)); add(r,idxS(k,2)); ++r;
    for(int d=0;d<3;++d)add(r,idxX(k,6+d)); add(r,idxS(k,2)); ++r;
    add(r,idxX(k,8)); add(r,idxS(k,2)); ++r;
    add(r,idxX(k,8)); add(r,idxS(k,2)); ++r;
    if (settings_.yaw) { add(r,idxX(k,10)); add(r,idxS(k,2)); ++r; }
  }
  if(r!=n_cons_) throw std::runtime_error("internal Jacobian row count mismatch");
}

bool TrajectoryTNLP::eval_jac_g(Ipopt::Index, const Ipopt::Number* x, bool,
                                Ipopt::Index, Ipopt::Index nele,
                                Ipopt::Index* iRow, Ipopt::Index* jCol,
                                Ipopt::Number* values) {
  if(nele!=static_cast<Ipopt::Index>(jac_i_.size()))return false;
  if(values==nullptr){std::copy(jac_i_.begin(),jac_i_.end(),iRow);std::copy(jac_j_.begin(),jac_j_.end(),jCol);return true;}
  int q=0;
  for(int i=0;i<nx_;++i)values[q++]=1.0;
  const double dt=settings_.dt;
  for(int k=0;k<settings_.N;++k){
    for(int d=0;d<3;++d){values[q++]=1.;values[q++]=-1.;values[q++]=-dt;values[q++]=-0.5*dt*dt;values[q++]=-dt*dt*dt/6.;}
    for(int d=0;d<3;++d){values[q++]=1.;values[q++]=-1.;values[q++]=-dt;values[q++]=-0.5*dt*dt;}
    for(int d=0;d<3;++d){values[q++]=1.;values[q++]=-1.;values[q++]=-dt;}
    if (settings_.yaw) {
      values[q++]=1.;values[q++]=-1.;values[q++]=-dt;values[q++]=-0.5*dt*dt;
      values[q++]=1.;values[q++]=-1.;values[q++]=-dt;
    }
  }
  const double tan2=std::tan(settings_.tilt_max)*std::tan(settings_.tilt_max);
  for(int k=1;k<settings_.N;++k){
    Eigen::Vector3d p;for(int d=0;d<3;++d)p[d]=x[idxX(k,d)];
    for(int j=0;j<settings_.K;++j){Eigen::Vector3d diff=p-obstacles_.block<3,1>(0,j);double dist=std::sqrt(diff.squaredNorm()+kObsEps*kObsEps);for(int d=0;d<3;++d)values[q++]=diff[d]/dist;values[q++]=1.;}
    for(int d=0;d<3;++d){values[q++]=1.;values[q++]=1.;}
    for(int d=0;d<3;++d){values[q++]=-1.;values[q++]=1.;}
    for(int d=0;d<3;++d)values[q++]=x[idxX(k,3+d)]/settings_.vmax;values[q++]=-1.;
    const double a0=x[idxX(k,6)],a1=x[idxX(k,7)],a2=x[idxX(k,8)];
    values[q++]=a0/settings_.g; values[q++]=a1/settings_.g; values[q++]=tan2*(settings_.g-a2)/settings_.g; values[q++]=-1.;
    values[q++]=1.;values[q++]=-1.;
    values[q++]=-1.;values[q++]=-1.;
    if (settings_.yaw) { values[q++]=x[idxX(k,10)]/settings_.yaw_rate_max; values[q++]=-1.; }
  }
  return q==nele;
}

void TrajectoryTNLP::buildHessianStructure() {
  hess_i_.clear(); hess_j_.clear();
  // Every variable can receive a diagonal objective/constraint contribution.
  for (int i=0; i<n_vars_; ++i) { hess_i_.push_back(i); hess_j_.push_back(i); }
  // Obstacle distance Hessians couple the three position coordinates. Store
  // only the lower-triangular off-diagonal terms required by IPOPT.
  for (int k=1; k<=settings_.N; ++k) {
    hess_i_.push_back(idxX(k,1)); hess_j_.push_back(idxX(k,0));
    hess_i_.push_back(idxX(k,2)); hess_j_.push_back(idxX(k,0));
    hess_i_.push_back(idxX(k,2)); hess_j_.push_back(idxX(k,1));
  }
  // Heading cost couples the horizontal velocity and the yaw angle (lower triangle: psi row).
  if (settings_.yaw) {
    for (int k=0; k<=settings_.N; ++k) {
      hess_i_.push_back(idxX(k,9)); hess_j_.push_back(idxX(k,3));
      hess_i_.push_back(idxX(k,9)); hess_j_.push_back(idxX(k,4));
    }
  }
}

bool TrajectoryTNLP::eval_h(Ipopt::Index, const Ipopt::Number* x, bool,
                            Ipopt::Number obj_factor, Ipopt::Index,
                            const Ipopt::Number* lambda, bool,
                            Ipopt::Index nele, Ipopt::Index* iRow,
                            Ipopt::Index* jCol, Ipopt::Number* values) {
  if (nele != static_cast<Ipopt::Index>(hess_i_.size())) return false;
  if (values == nullptr) {
    std::copy(hess_i_.begin(), hess_i_.end(), iRow);
    std::copy(hess_j_.begin(), hess_j_.end(), jCol);
    return true;
  }
  std::fill(values, values + nele, 0.0);
  auto add_diag = [&](int var, double v) { values[var] += v; };
  auto add_pos_hess = [&](int k, const Eigen::Matrix3d& H, double scale) {
    for (int d=0; d<3; ++d) add_diag(idxX(k,d), scale*H(d,d));
    const int base = n_vars_ + (k-1)*3;
    values[base+0] += scale*H(1,0);
    values[base+1] += scale*H(2,0);
    values[base+2] += scale*H(2,1);
  };
  auto distance_hess = [](const Eigen::Vector3d& diff) {
    const double dist = std::sqrt(diff.squaredNorm() + kObsEps*kObsEps);
    return Eigen::Matrix3d::Identity()/dist - (diff*diff.transpose())/(dist*dist*dist);
  };

  // Heading cost Hessian of stage k (vN, vE, psi block + diagonal omega) times `scale`.
  auto add_heading_hess = [&](int k, double scale) {
    const double psi = x[idxX(k,9)];
    const double vN = x[idxX(k,3)], vE = x[idxX(k,4)];
    const double s = std::sin(psi), c = std::cos(psi);
    const double lat = vN * s - vE * c;
    const double lpsi = vN * c + vE * s;          // d lat / d psi
    const double w = settings_.w_yaw_lateral;
    add_diag(idxX(k,3), scale * 2.0 * w * s * s);
    add_diag(idxX(k,4), scale * 2.0 * w * c * c);
    add_diag(idxX(k,9), scale * (2.0 * hdg_w_[k] + 2.0 * w * (lpsi * lpsi - lat * lat)));
    add_diag(idxX(k,10), scale * 2.0 * settings_.w_yaw_rate);
    const int base = n_vars_ + 3 * settings_.N + 2 * k;
    values[base + 0] += scale * 2.0 * w * (s * lpsi + lat * c);     // (psi, vN)
    values[base + 1] += scale * 2.0 * w * (-c * lpsi + lat * s);    // (psi, vE)
  };

  // Objective Hessian.
  for (int k=0; k<settings_.N; ++k) {
    for (int d=0; d<3; ++d) {
      add_diag(idxX(k,d), obj_factor*2.0*settings_.q_pos[d]);
      add_diag(idxX(k,3+d), obj_factor*2.0*settings_.q_vel[d]);
      add_diag(idxX(k,6+d), obj_factor*2.0*settings_.w_acc);
      double hu = 2.0*settings_.r_jerk;
      if (k==0) hu += 2.0*settings_.r_delta_input_first[d];
      add_diag(idxU(k,d), obj_factor*hu);
    }
    if (k>=1) {
      add_diag(idxS(k,0), obj_factor*2.0*settings_.w_slack_obstacle_l2);
      add_diag(idxS(k,1), obj_factor*2.0*settings_.w_slack_arena_l2);
      add_diag(idxS(k,2), obj_factor*2.0*settings_.w_slack_dynamic_l2);
    }
    if (settings_.yaw) {
      add_heading_hess(k, obj_factor);
      add_diag(idxU(k,3), obj_factor*2.0*settings_.w_yaw_acc);
    }
  }
  const int N=settings_.N;
  if (settings_.yaw) add_heading_hess(N, obj_factor);
  for (int d=0; d<3; ++d) {
    add_diag(idxX(N,d), obj_factor*2.0*settings_.q_pos_terminal[d]);
    add_diag(idxX(N,3+d), obj_factor*2.0*settings_.q_vel_terminal[d]);
    add_diag(idxX(N,6+d), obj_factor*2.0*settings_.w_acc);
  }
  Eigen::Vector3d pN; for(int d=0;d<3;++d)pN[d]=x[idxX(N,d)];
  for(int j=0;j<settings_.K;++j){
    Eigen::Vector3d diff=pN-obstacles_.block<3,1>(0,j);
    double dist=std::sqrt(diff.squaredNorm()+kObsEps*kObsEps);
    double h=dist-obstacles_(3,j);
    if(h<0.0){
      Eigen::Vector3d dh=diff/dist;
      Eigen::Matrix3d H=2.0*settings_.w_terminal_penalty*(dh*dh.transpose()+h*distance_hess(diff));
      add_pos_hess(N,H,obj_factor);
    }
  }
  for(int d=0;d<3;++d){
    if(pN[d]-arena_lb_[d]<0.0) add_diag(idxX(N,d),obj_factor*2.0*settings_.w_terminal_penalty);
    if(arena_ub_[d]-pN[d]<0.0) add_diag(idxX(N,d),obj_factor*2.0*settings_.w_terminal_penalty);
  }

  // Constraint Hessians multiplied by IPOPT's lambda.
  int r = nx_ + nx_*settings_.N; // initial condition + linear dynamics
  const double tan2=std::tan(settings_.tilt_max)*std::tan(settings_.tilt_max);
  for(int k=1;k<settings_.N;++k){
    Eigen::Vector3d p; for(int d=0;d<3;++d)p[d]=x[idxX(k,d)];
    for(int j=0;j<settings_.K;++j,++r){
      Eigen::Vector3d diff=p-obstacles_.block<3,1>(0,j);
      add_pos_hess(k,distance_hess(diff),lambda[r]);
    }
    r += 6; // arena constraints are linear
    const double lv=lambda[r++];
    for(int d=0;d<3;++d)add_diag(idxX(k,3+d),lv/settings_.vmax);
    const double lt=lambda[r++];
    add_diag(idxX(k,6),lt/settings_.g);
    add_diag(idxX(k,7),lt/settings_.g);
    add_diag(idxX(k,8),-lt*tan2/settings_.g);
    r += 2; // vertical acceleration inequalities are linear
    if (settings_.yaw) {
      const double lw=lambda[r++];
      add_diag(idxX(k,10), lw/settings_.yaw_rate_max);
    }
  }
  return true;
}

void TrajectoryTNLP::unpack(const Ipopt::Number* x, Eigen::MatrixXd& X, Eigen::MatrixXd& U, Eigen::MatrixXd& S) const {
  X.resize(nx_,settings_.N+1);U.resize(nu_,settings_.N);S.resize(3,settings_.N);
  for(int k=0;k<=settings_.N;++k)for(int i=0;i<nx_;++i)X(i,k)=x[idxX(k,i)];
  for(int k=0;k<settings_.N;++k){for(int i=0;i<nu_;++i)U(i,k)=x[idxU(k,i)];for(int i=0;i<3;++i)S(i,k)=x[idxS(k,i)];}
}

void TrajectoryTNLP::finalize_solution(Ipopt::SolverReturn,
                                       Ipopt::Index n, const Ipopt::Number* x,
                                       const Ipopt::Number* z_L, const Ipopt::Number* z_U,
                                       Ipopt::Index m, const Ipopt::Number*, const Ipopt::Number* lambda,
                                       Ipopt::Number obj_value,
                                       const Ipopt::IpoptData*, Ipopt::IpoptCalculatedQuantities*) {
  unpack(x,X_result_,U_result_,S_result_);
  final_cost_=obj_value;has_solution_=true;
  X_warm_=X_result_;U_warm_=U_result_;S_warm_=S_result_;has_warm_=true;
  has_warm_duals_ = (z_L != nullptr && z_U != nullptr && lambda != nullptr && n == n_vars_ && m == n_cons_);
  if (has_warm_duals_) {
    zl_warm_.assign(z_L, z_L + n);
    zu_warm_.assign(z_U, z_U + n);
    lambda_warm_.assign(lambda, lambda + m);
  }
}

NativeIpoptTrajectorySolver::NativeIpoptTrajectorySolver(ProblemSettings settings)
    // `new IpoptApplication()` works with every IPOPT release (3.11 on Ubuntu 22.04
    // has no IpoptApplicationFactory(), which was only added in 3.12).
    : settings_(std::move(settings)), app_(new Ipopt::IpoptApplication()), nlp_(new TrajectoryTNLP(settings_)) {
  if (Ipopt::IsNull(app_)) throw std::runtime_error("could not create IpoptApplication");
  app_->Options()->SetIntegerValue("print_level", settings_.print_level);
  app_->Options()->SetStringValue("sb", "yes");
  app_->Options()->SetIntegerValue("max_iter", settings_.max_iter);
  app_->Options()->SetNumericValue("tol", settings_.tol);
  app_->Options()->SetNumericValue("acceptable_tol", settings_.acceptable_tol);
  app_->Options()->SetIntegerValue("acceptable_iter", settings_.acceptable_iter);
  app_->Options()->SetStringValue("mu_strategy", settings_.mu_strategy);
  app_->Options()->SetStringValue("hessian_approximation", settings_.hessian_approximation);
  if (!settings_.linear_solver.empty()) app_->Options()->SetStringValue("linear_solver", settings_.linear_solver);
  if (settings_.max_cpu_time > 0.0) app_->Options()->SetNumericValue("max_cpu_time", settings_.max_cpu_time);
  for (const auto& [k, v] : settings_.string_options)
    if (!app_->Options()->SetStringValue(k, v)) throw std::runtime_error("invalid IPOPT string option " + k + "=" + v);
  for (const auto& [k, v] : settings_.integer_options)
    if (!app_->Options()->SetIntegerValue(k, v) && !app_->Options()->SetNumericValue(k, static_cast<double>(v)))
      throw std::runtime_error("invalid IPOPT option " + k);
  for (const auto& [k, v] : settings_.numeric_options)
    if (!app_->Options()->SetNumericValue(k, v)) throw std::runtime_error("invalid IPOPT numeric option " + k);
  if (settings_.warm_start != "primal" && settings_.warm_start != "primal_dual")
    throw std::runtime_error("ipopt_cpp warm_start must be primal or primal_dual");
  const auto status=app_->Initialize();
  if(status!=Ipopt::Solve_Succeeded) throw std::runtime_error("IPOPT Initialize failed: "+statusString(status));
}

SolveOutput NativeIpoptTrajectorySolver::solve(const Eigen::VectorXd& x0,
                                               const Eigen::MatrixXd& pref,
                                               const Eigen::MatrixXd& vref,
                                               const Eigen::MatrixXd& obstacles,
                                               const Eigen::Vector3d& arena_lb,
                                               const Eigen::Vector3d& arena_ub,
                                               const Eigen::Vector3d& uprev,
                                               const Eigen::VectorXd& hdg_ref,
                                               const Eigen::VectorXd& hdg_w,
                                               int warm_shift) {
  nlp_->setProblemData(x0,pref,vref,obstacles,arena_lb,arena_ub,uprev,hdg_ref,hdg_w,warm_shift);
  const bool warm_duals = settings_.warm_start == "primal_dual" && nlp_->dualWarmAvailable();
  const bool warm_primal = nlp_->primalWarmAvailable();
  nlp_->setUseWarmDuals(warm_duals);
  app_->Options()->SetStringValue("warm_start_init_point", warm_duals ? "yes" : "no");
  const auto t0=std::chrono::high_resolution_clock::now();
  const auto st=app_->OptimizeTNLP(nlp_);
  const auto t1=std::chrono::high_resolution_clock::now();
  SolveOutput out;
  out.warm_primal=warm_primal;out.warm_duals=warm_duals;
  out.solve_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
  out.status=statusString(st);out.success=statusSuccess(st);
  if(Ipopt::IsValid(app_->Statistics()))out.iterations=static_cast<int>(app_->Statistics()->IterationCount());
  if(nlp_->hasSolution()){out.X=nlp_->states();out.U=nlp_->inputs();out.S=nlp_->slacks();out.cost=nlp_->finalCost();}
  if(!out.success) nlp_->resetWarmStart();
  return out;
}

void NativeIpoptTrajectorySolver::resetWarmStart(){nlp_->resetWarmStart();}
int NativeIpoptTrajectorySolver::nVars() const{return nlp_->nVars();}
int NativeIpoptTrajectorySolver::nCons() const{return nlp_->nCons();}

bool NativeIpoptTrajectorySolver::statusSuccess(Ipopt::ApplicationReturnStatus s){
  return s==Ipopt::Solve_Succeeded || s==Ipopt::Solved_To_Acceptable_Level || s==Ipopt::Feasible_Point_Found;
}
std::string NativeIpoptTrajectorySolver::statusString(Ipopt::ApplicationReturnStatus s){
  switch(s){
    case Ipopt::Solve_Succeeded:return "Solve_Succeeded";
    case Ipopt::Solved_To_Acceptable_Level:return "Solved_To_Acceptable_Level";
    case Ipopt::Infeasible_Problem_Detected:return "Infeasible_Problem_Detected";
    case Ipopt::Search_Direction_Becomes_Too_Small:return "Search_Direction_Becomes_Too_Small";
    case Ipopt::Diverging_Iterates:return "Diverging_Iterates";
    case Ipopt::User_Requested_Stop:return "User_Requested_Stop";
    case Ipopt::Feasible_Point_Found:return "Feasible_Point_Found";
    case Ipopt::Maximum_Iterations_Exceeded:return "Maximum_Iterations_Exceeded";
    case Ipopt::Restoration_Failed:return "Restoration_Failed";
    case Ipopt::Error_In_Step_Computation:return "Error_In_Step_Computation";
    case Ipopt::Maximum_CpuTime_Exceeded:return "Maximum_CpuTime_Exceeded";
    case Ipopt::Not_Enough_Degrees_Of_Freedom:return "Not_Enough_Degrees_Of_Freedom";
    case Ipopt::Invalid_Problem_Definition:return "Invalid_Problem_Definition";
    case Ipopt::Invalid_Option:return "Invalid_Option";
    case Ipopt::Invalid_Number_Detected:return "Invalid_Number_Detected";
    case Ipopt::Unrecoverable_Exception:return "Unrecoverable_Exception";
    case Ipopt::NonIpopt_Exception_Thrown:return "NonIpopt_Exception_Thrown";
    case Ipopt::Insufficient_Memory:return "Insufficient_Memory";
    case Ipopt::Internal_Error:return "Internal_Error";
    default:return "Ipopt_Status_"+std::to_string(static_cast<int>(s));
  }
}

} // namespace px4nmpc::ipoptcpp
