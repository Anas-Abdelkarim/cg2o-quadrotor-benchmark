#pragma once
// Nonlinear quadrotor MPC with collective thrust and body rates as inputs,
// progress along an infeasible guide line, and an attitude-aware five-sphere
// airframe. Every function below is shared by the cg2o graph and the IPOPT
// TNLP, so both backends solve the same nonlinear program.
//
// Variables, z = [x_1..x_N, u_0..u_{N-1}, slack_0..slack_{N-1}]
//   x = [pN pE pD vN vE vD roll pitch yaw p q r s]  (NED, FRD body, ZYX Euler,
//                                                    measured body rates)
//   u = [T p_sp q_sp r_sp vs]                        (N, rate setpoints sent to
//                                                    PX4, guide progress/s)
// PX4's rate loop is a first-order lag per axis: d(omega)/dt = (omega_sp - omega)/tau.
// Inequalities are written g(z) <= 0.
#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <unsupported/Eigen/AutoDiff>
#include <vector>

namespace challenging {
using Eigen::MatrixXd;
using Eigen::VectorXd;
template <class T> using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
constexpr int NX = 13, NU = 5, NB = 5;   // state, input, body spheres
constexpr int IW = 9, IS = 12;            // first body-rate index, progress index   // state, input, body spheres
constexpr int STATE_ROWS = 9, INPUT_ROWS = 10, PATH_RES = 9;

enum ObstacleType { Sphere = 0, Post = 1, Bar = 2 };

struct Params {
  int N = 40;
  double dt = .1, mass = 2.0643076923, gravity = 9.81;
  Eigen::Vector3d drag{0., 0., 0.};          // body-frame linear rotor drag [1/s]
  Eigen::Vector3d tau_rate{.095, .070, .390}; // PX4 rate-loop lag [s], identified in Gazebo
  double tmin = 5., tmax = 34., rate_xy = 2.5, rate_z = 1.5;
  double vmax = 5., cos_tilt = std::cos(40. * M_PI / 180.);
  double vs_max = 2.5, s_goal = 14., vs_cold = 2.5;
  // Guide line: N = s, E = e_amp sin(e_freq s + e_phase),
  // altitude h = alt0 + alt_amp sin(alt_freq s + alt_phase), D = -h.
  double e_amp = 0., e_freq = 0., e_phase = 0.;
  double alt0 = 2.5, alt_amp = 0., alt_freq = 0., alt_phase = 0.;
  // Airframe: NB body points (FRD, m) and their sphere radii.
  MatrixXd body = (MatrixXd(NB, 3) << 0., 0., 0., .177, .177, 0., .177, -.177,
                   0., -.177, .177, 0., -.177, -.177, 0.)
                      .finished();
  VectorXd body_radius = (VectorXd(NB) << .12, .16, .16, .16, .16).finished();
  double margin = .15;                      // tracking reserve on every obstacle row
  Eigen::Vector3d lower{-16.55, -4.55, -5.55}, upper{16.55, 4.55, -.3};
  // Weights (least-squares residuals).
  double w_lag = 40., w_lag_terminal = 200., w_contour = 1.,
         w_contour_terminal = 10., w_heading = 1., w_attitude = .1;
  double w_thrust = .01, w_rate = .05, w_progress = 2., v_bar = 2.5;
  double w_goal = 10., goal_delta = 2.;     // linear terminal progress reward
  VectorXd w_slew = (VectorXd(NU) << .01, .3, .3, .3, 1.).finished();
  double w_slack = 1e4;
  // Obstacles, one per row: [type, a, b, c, radius].
  //   Sphere: centre (N,E,D).  Post: vertical axis through (N,E).
  //   Bar: horizontal axis along East through (N,D) stored in (a,c).
  MatrixXd obstacles;
  double hover() const { return mass * gravity; }
};

// ---------------------------------------------------------------- geometry
template <class T> Eigen::Matrix<T, 3, 1> path(const T &s, const Params &p) {
  using std::sin;
  Eigen::Matrix<T, 3, 1> c;
  c[0] = s;
  c[1] = p.e_amp * sin(p.e_freq * s + p.e_phase);
  c[2] = -(p.alt0 + p.alt_amp * sin(p.alt_freq * s + p.alt_phase));
  return c;
}
template <class T>
Eigen::Matrix<T, 3, 1> path_derivative(const T &s, const Params &p) {
  using std::cos;
  Eigen::Matrix<T, 3, 1> c;
  c[0] = T(1.);
  c[1] = p.e_amp * p.e_freq * cos(p.e_freq * s + p.e_phase);
  c[2] = -p.alt_amp * p.alt_freq * cos(p.alt_freq * s + p.alt_phase);
  return c;
}
// ZYX active rotation, body FRD to NED.
template <class T>
Eigen::Matrix<T, 3, 3> rotation(const T &phi, const T &theta, const T &psi) {
  using std::cos;
  using std::sin;
  T sp = sin(phi), cp = cos(phi), st = sin(theta), ct = cos(theta),
    sy = sin(psi), cy = cos(psi);
  Eigen::Matrix<T, 3, 3> R;
  R << cy * ct, cy * st * sp - sy * cp, cy * st * cp + sy * sp,
       sy * ct, sy * st * sp + cy * cp, sy * st * cp - cy * sp,
       -st, ct * sp, ct * cp;
  return R;
}
// Squared distance from a point to obstacle j's axis/centre.
template <class T>
T obstacle_distance2(const Eigen::Matrix<T, 3, 1> &w, const Params &p, int j) {
  const int type = int(p.obstacles(j, 0));
  T dn = w[0] - p.obstacles(j, 1), de = w[1] - p.obstacles(j, 2),
    dd = w[2] - p.obstacles(j, 3);
  if (type == Sphere) return dn * dn + de * de + dd * dd;
  if (type == Post) return dn * dn + de * de;
  if (type == Bar) return dn * dn + dd * dd;
  throw std::invalid_argument("unknown obstacle type");
}

// ---------------------------------------------------------------- dynamics
template <class T>
Vec<T> rhs(const Vec<T> &x, const Vec<T> &u, const Params &p) {
  using std::cos;
  using std::sin;
  using std::tan;
  auto R = rotation<T>(x[6], x[7], x[8]);
  Eigen::Matrix<T, 3, 1> v = x.template segment<3>(3);
  Eigen::Matrix<T, 3, 1> a = -(u[0] / p.mass) * R.col(2);
  a[2] += p.gravity;
  if (p.drag.squaredNorm() > 0.) {
    Eigen::Matrix<T, 3, 1> vb = R.transpose() * v;
    for (int i = 0; i < 3; i++) vb[i] *= p.drag[i];
    a -= R * vb;
  }
  T sp = sin(x[6]), cp = cos(x[6]), ct = cos(x[7]), tt = tan(x[7]);
  const T wp = x[IW], wq = x[IW + 1], wr = x[IW + 2];
  Vec<T> y(NX);
  y.template head<3>() = v;
  y.template segment<3>(3) = a;
  y[6] = wp + sp * tt * wq + cp * tt * wr;
  y[7] = cp * wq - sp * wr;
  y[8] = (sp * wq + cp * wr) / ct;
  for (int i = 0; i < 3; i++) y[IW + i] = (u[1 + i] - x[IW + i]) / p.tau_rate[i];
  y[IS] = u[4];
  return y;
}
template <class T>
Vec<T> step(const Vec<T> &x, const Vec<T> &u, const Params &p, double h) {
  // Scalars as T: nested AD has no double * vector<T> product.
  const T half(.5 * h), full(h), sixth(h / 6.), two(2.);
  Vec<T> a = rhs(x, u, p);
  Vec<T> b = rhs(Vec<T>(x + half * a), u, p);
  Vec<T> c = rhs(Vec<T>(x + half * b), u, p);
  Vec<T> d = rhs(Vec<T>(x + full * c), u, p);
  return x + sixth * (a + two * b + two * c + d);
}
template <class T>
Vec<T> step(const Vec<T> &x, const Vec<T> &u, const Params &p) {
  return step(x, u, p, p.dt);
}

// ---------------------------------------------------------------- factors
enum Kind { Dynamics, PathCost, InputCost, SlewCost, StateIneq, InputIneq,
            ObstacleIneq, SlackCost, SlackIneq };
inline bool is_cost(int k) {
  return k == PathCost || k == InputCost || k == SlewCost || k == SlackCost;
}
inline int rows_of(int kind) {
  switch (kind) {
  case Dynamics: return NX;
  case PathCost: return PATH_RES;
  case InputCost: return NU;
  case SlewCost: return NU;
  case StateIneq: return STATE_ROWS;
  case InputIneq: return INPUT_ROWS;
  case ObstacleIneq: return NB;
  default: return 1;
  }
}
struct Factor {
  Kind kind;
  std::vector<int> indices;   // -1: fixed initial state component
  int stage = 0, obstacle = -1;
  bool terminal = false, first = false;
  VectorXd uprev = VectorXd::Zero(NU);
  bool cost() const { return is_cost(kind); }
  // Local variables that enter the residual nonlinearly. Second derivatives
  // vanish outside this set: position enters the dynamics linearly, the
  // progress speed linearly, input bounds and slack rows are linear.
  std::vector<int> curved() const {
    switch (kind) {
    // Rates and Euler angles couple nonlinearly. Within an RK4 step the rate
    // setpoints reach the intermediate rate states, which multiply trig terms
    // of the angles, so the setpoints are curved too; vs and p are linear.
    case Dynamics: return {3, 4, 5, 6, 7, 8, 9, 10, 11, NX, NX + 1, NX + 2, NX + 3};
    case PathCost: return {0, 1, 2, 8, IS};
    case StateIneq: return {3, 4, 5, 6, 7};
    case ObstacleIneq: return {0, 1, 2, 6, 7, 8};
    default: return {};
    }
  }
  // Pairs (a >= b, local indices) with possibly nonzero Lagrangian Hessian.
  std::vector<std::pair<int, int>> hessian_pairs() const {
    std::vector<int> set;
    if (cost()) for (int i = 0; i < int(indices.size()); i++) set.push_back(i);  // J^T J
    else set = curved();
    std::vector<std::pair<int, int>> out;
    for (size_t a = 0; a < set.size(); a++)
      for (size_t b = 0; b <= a; b++) out.emplace_back(set[a], set[b]);
    return out;
  }
};

template <class T>
Vec<T> value(const Factor &f, const Vec<T> &z, const Params &p) {
  using std::cos;
  using std::sin;
  using std::sqrt;
  switch (f.kind) {
  case Dynamics:
    return Vec<T>(z.segment(NX + NU, NX)) -
           step(Vec<T>(z.head(NX)), Vec<T>(z.segment(NX, NU)), p);
  case PathCost: {
    const T s = z[IS];
    Eigen::Matrix<T, 3, 1> e = z.template head<3>() - path<T>(s, p);
    Eigen::Matrix<T, 3, 1> d = path_derivative<T>(s, p);
    Eigen::Matrix<T, 3, 1> t = d / sqrt(d.squaredNorm());
    T lag = t.dot(e);
    Eigen::Matrix<T, 3, 1> contour = e - lag * t;
    T hxy = sqrt(d[0] * d[0] + d[1] * d[1]);
    const double wl = f.terminal ? p.w_lag_terminal : p.w_lag;
    const double wc = f.terminal ? p.w_contour_terminal : p.w_contour;
    Vec<T> r(PATH_RES);
    r[0] = sqrt(wl) * lag;
    for (int i = 0; i < 3; i++) r[1 + i] = sqrt(wc) * contour[i];
    r[4] = sqrt(p.w_heading) * (cos(z[8]) - d[0] / hxy);
    r[5] = sqrt(p.w_heading) * (sin(z[8]) - d[1] / hxy);
    r[6] = sqrt(p.w_attitude) * z[6];
    r[7] = sqrt(p.w_attitude) * z[7];
    // Linear terminal progress reward in least-squares form:
    // r^2 = w_goal * softplus(s_goal + goal_delta - s_N), ~linear for s_N <= s_goal.
    if (f.terminal) {
      T a = p.s_goal + p.goal_delta - s;
      T arg = .5 * (a + sqrt(a * a + .01));
      r[8] = sqrt(p.w_goal * arg);
    } else {
      r[8] = T(0.) * s;
    }
    return r;
  }
  case InputCost: {
    Vec<T> r(NU);
    r[0] = sqrt(p.w_thrust) * (z[0] - p.hover());
    for (int i = 1; i < 4; i++) r[i] = sqrt(p.w_rate) * z[i];
    r[4] = sqrt(p.w_progress) * (z[4] - p.v_bar);
    return r;
  }
  case SlewCost: {
    Vec<T> r(NU);
    for (int i = 0; i < NU; i++)
      r[i] = sqrt(p.w_slew[i]) * (z[i] - (f.first ? T(f.uprev[i]) : z[NU + i]));
    return r;
  }
  case StateIneq: {
    Vec<T> r(STATE_ROWS);
    r[0] = z.template segment<3>(3).squaredNorm() / (p.vmax * p.vmax) - 1.;
    // Exact tilt cone: angle between body z and vertical <= tilt limit.
    r[1] = (p.cos_tilt - cos(z[6]) * cos(z[7])) / (1. - p.cos_tilt);
    for (int i = 0; i < 3; i++) {
      r[2 + 2 * i] = p.lower[i] - z[i];
      r[3 + 2 * i] = z[i] - p.upper[i];
    }
    r[8] = z[IS] - p.s_goal;
    return r;
  }
  case InputIneq: {
    Vec<T> r(INPUT_ROWS);
    r[0] = (p.tmin - z[0]) / p.hover();
    r[1] = (z[0] - p.tmax) / p.hover();
    for (int i = 1; i < 4; i++) {
      double b = i == 3 ? p.rate_z : p.rate_xy;
      r[2 * i] = -1. - z[i] / b;
      r[2 * i + 1] = z[i] / b - 1.;
    }
    r[8] = -z[4] / p.vs_max;
    r[9] = z[4] / p.vs_max - 1.;
    return r;
  }
  case ObstacleIneq: {
    // z = [x (10), stage slack (1)]
    auto R = rotation<T>(z[6], z[7], z[8]);
    Eigen::Matrix<T, 3, 1> pos = z.template head<3>();
    Vec<T> r(NB);
    for (int i = 0; i < NB; i++) {
      Eigen::Matrix<T, 3, 1> w = pos + R * p.body.row(i).transpose().template cast<T>();
      double rho = p.obstacles(f.obstacle, 4) + p.body_radius[i] + p.margin;
      r[i] = 1. - obstacle_distance2<T>(w, p, f.obstacle) / (rho * rho) - z[NX];
    }
    return r;
  }
  case SlackCost: {
    Vec<T> r(1);
    r[0] = sqrt(p.w_slack) * z[0];
    return r;
  }
  case SlackIneq: {
    Vec<T> r(1);
    r[0] = -z[0];
    return r;
  }
  }
  throw std::invalid_argument("unknown factor kind");
}

// Value and Jacobian of one factor (forward-mode AD, at most 25 local variables).
constexpr int MAX_LOCAL = NX + NU + NX;
inline std::pair<VectorXd, MatrixXd>
derivative(const Factor &f, const VectorXd &z, const Params &p) {
  if (z.size() > MAX_LOCAL) throw std::invalid_argument("local derivative width");
  using AD = Eigen::AutoDiffScalar<Eigen::Matrix<double, MAX_LOCAL, 1>>;
  Vec<AD> a(z.size());
  for (int j = 0; j < z.size(); j++) {
    a[j].value() = z[j];
    a[j].derivatives().setZero();
    a[j].derivatives()[j] = 1.;
  }
  Vec<AD> b = value(f, a, p);
  VectorXd r(b.size());
  MatrixXd J(b.size(), z.size());
  for (int i = 0; i < b.size(); i++) {
    r[i] = b[i].value();
    J.row(i) = b[i].derivatives().head(z.size()).transpose();
  }
  return {r, J};
}

// Local Hessian (indices.size() square) of
//   cost:        sigma*|r|^2  = 2 sigma (J^T J + sum_i r_i Hess r_i)
//   constraint:  w^T g        =          sum_i w_i Hess g_i
// The curvature terms use nested forward AD seeded only on f.curved().
// Used by IPOPT's exact Lagrangian Hessian; cg2o uses its Gauss-Newton model.
constexpr int MAX_CURVED = 13;
inline MatrixXd hessian(const Factor &f, const VectorXd &z, const Params &p,
                        const double *weights, double sigma) {
  const int n = int(z.size());
  MatrixXd H = MatrixXd::Zero(n, n);
  VectorXd w;
  if (f.cost()) {
    auto [r, J] = derivative(f, z, p);
    H = 2. * sigma * J.transpose() * J;
    w = 2. * sigma * r;
  } else {
    w = Eigen::Map<const VectorXd>(weights, rows_of(f.kind));
  }
  const std::vector<int> c = f.curved();
  const int m = int(c.size());
  if (m == 0) return H;
  if (m > MAX_CURVED) throw std::invalid_argument("hessian width");
  using D1 = Eigen::Matrix<double, MAX_CURVED, 1>;
  using AD1 = Eigen::AutoDiffScalar<D1>;
  using AD2 = Eigen::AutoDiffScalar<Eigen::Matrix<AD1, MAX_CURVED, 1>>;
  Vec<AD2> a(n);
  for (int j = 0; j < n; j++) {
    a[j].value() = AD1(z[j], D1::Zero());
    for (int k = 0; k < MAX_CURVED; k++) a[j].derivatives()[k] = AD1(0., D1::Zero());
  }
  for (int k = 0; k < m; k++) {
    a[c[k]].value().derivatives()[k] = 1.;
    a[c[k]].derivatives()[k] = AD1(1., D1::Zero());
  }
  Vec<AD2> v = value(f, a, p);
  AD2 y(AD1(0., D1::Zero()));
  for (int k = 0; k < MAX_CURVED; k++) y.derivatives()[k] = AD1(0., D1::Zero());
  for (int i = 0; i < v.size(); i++) y = y + w[i] * v[i];
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++) H(c[i], c[j]) += y.derivatives()[i].derivatives()[j];
  return H;
}

// ---------------------------------------------------------------- problem
struct Problem {
  Params p;
  VectorXd x0 = VectorXd::Zero(NX);
  VectorXd uprev = VectorXd::Zero(NU);
  std::vector<Factor> factors;
  int K() const { return int(p.obstacles.rows()); }
  int n() const { return (NX + NU + 1) * p.N; }
  int nx(int k, int j) const { return k == 0 ? -1 : NX * (k - 1) + j; }
  int nu(int k, int j) const { return NX * p.N + NU * k + j; }
  int ns(int k) const { return (NX + NU) * p.N + k; }
  VectorXd local(const Factor &f, const VectorXd &z) const {
    VectorXd a(f.indices.size());
    for (int j = 0; j < a.size(); j++)
      a[j] = f.indices[j] < 0 ? x0[j] : z[f.indices[j]];
    return a;
  }
  void build() {
    if ((p.tau_rate.array() <= 0.).any())
      throw std::invalid_argument("tau_rate must be positive");
    if (p.N < 1 || p.dt <= 0 || p.mass <= 0 || p.gravity <= 0 || p.vmax <= 0 ||
        !(p.cos_tilt > 0. && p.cos_tilt < 1.) || p.tmin < 0 || p.tmax <= p.tmin ||
        p.rate_xy <= 0 || p.rate_z <= 0 || p.vs_max <= 0 || p.margin < 0)
      throw std::invalid_argument("invalid physical parameters");
    if (p.body.rows() != NB || p.body.cols() != 3 || p.body_radius.size() != NB ||
        p.w_slew.size() != NU)
      throw std::invalid_argument("invalid airframe or weight arrays");
    if (p.obstacles.cols() != 5 || !p.obstacles.allFinite())
      throw std::invalid_argument("obstacles must be [type,a,b,c,radius] rows");
    for (int j = 0; j < K(); j++) {
      int t = int(p.obstacles(j, 0));
      if (double(t) != p.obstacles(j, 0) || t < 0 || t > 2 || p.obstacles(j, 4) <= 0)
        throw std::invalid_argument("invalid obstacle row");
    }
    if (x0.size() != NX || uprev.size() != NU || !x0.allFinite() || !uprev.allFinite())
      throw std::invalid_argument("invalid problem arrays");
    factors.clear();
    auto xs = [&](int k) {
      std::vector<int> a;
      for (int j = 0; j < NX; j++) a.push_back(nx(k, j));
      return a;
    };
    auto us = [&](int k) {
      std::vector<int> a;
      for (int j = 0; j < NU; j++) a.push_back(nu(k, j));
      return a;
    };
    for (int k = 0; k <= p.N; k++) {
      Factor f;
      f.stage = k;
      if (k > 0) {
        f.kind = PathCost;
        f.indices = xs(k);
        f.terminal = k == p.N;
        factors.push_back(f);
        f.terminal = false;
        f.kind = StateIneq;
        factors.push_back(f);
        for (int j = 0; j < K(); j++) {
          Factor o = f;
          o.kind = ObstacleIneq;
          o.obstacle = j;
          o.indices.push_back(ns(k - 1));
          factors.push_back(o);
        }
      }
      if (k == p.N) continue;
      f.kind = Dynamics;
      f.indices = xs(k);
      auto u = us(k), next = xs(k + 1);
      f.indices.insert(f.indices.end(), u.begin(), u.end());
      f.indices.insert(f.indices.end(), next.begin(), next.end());
      factors.push_back(f);
      f.indices = u;
      f.kind = InputCost;
      factors.push_back(f);
      f.kind = InputIneq;
      factors.push_back(f);
      f.kind = SlewCost;
      f.first = k == 0;
      f.uprev = uprev;
      if (k > 0) {
        auto old = us(k - 1);
        f.indices.insert(f.indices.end(), old.begin(), old.end());
      }
      factors.push_back(f);
      f.first = false;
      f.indices = {ns(k)};
      f.kind = SlackCost;
      factors.push_back(f);
      f.kind = SlackIneq;
      factors.push_back(f);
    }
  }
  MatrixXd states(const VectorXd &z) const {
    MatrixXd X(NX, p.N + 1);
    X.col(0) = x0;
    for (int k = 1; k <= p.N; k++) X.col(k) = z.segment(NX * (k - 1), NX);
    return X;
  }
  MatrixXd inputs(const VectorXd &z) const {
    MatrixXd U(NU, p.N);
    for (int k = 0; k < p.N; k++) U.col(k) = z.segment(NX * p.N + NU * k, NU);
    return U;
  }
  VectorXd slacks(const VectorXd &z) const { return z.segment((NX + NU) * p.N, p.N); }
  VectorXd pack(const MatrixXd &X, const MatrixXd &U, const VectorXd &S) const {
    VectorXd z(n());
    for (int k = 1; k <= p.N; k++) z.segment(NX * (k - 1), NX) = X.col(k);
    for (int k = 0; k < p.N; k++) z.segment(NX * p.N + NU * k, NU) = U.col(k);
    z.segment((NX + NU) * p.N, p.N) = S;
    return z;
  }
};

// Closest guide-line parameter to a position (Gauss-Newton on |p - c(s)|^2,
// started at s = pN, which is exact for a straight line). Deterministic.
inline double project(const Eigen::Vector3d &pos, const Params &p) {
  double s = pos[0];
  for (int it = 0; it < 8; it++) {
    Eigen::Vector3d e = pos - path<double>(s, p), d = path_derivative<double>(s, p);
    double ds = d.dot(e) / d.squaredNorm();
    s += ds;
    if (std::abs(ds) < 1e-12) break;
  }
  return std::min(s, p.s_goal);
}

// Obstacle-ignorant cold start shared bit-for-bit by both backends: states on
// the guide line at the cold progress speed, level attitude, yaw along the
// tangent, hover thrust, zero rates, zero obstacle slacks. It violates the
// dynamics (it starts from the guide line, not from x0) and, where the guide
// line crosses obstacles, the obstacle rows.
inline VectorXd cold_start(const Problem &b) {
  const Params &p = b.p;
  MatrixXd X(NX, p.N + 1), U(NU, p.N);
  X.col(0) = b.x0;
  double s = std::min(b.x0[IS], p.s_goal);
  for (int k = 0; k < p.N; k++) {
    double next = std::min(s + p.vs_cold * p.dt, p.s_goal);
    double vs = (next - s) / p.dt;
    Eigen::Vector3d c = path<double>(next, p), d = path_derivative<double>(next, p);
    X.col(k + 1) << c, d * vs, 0., 0., std::atan2(d[1], d[0]), 0., 0., 0., next;
    U.col(k) << p.hover(), 0., 0., 0., vs;
    s = next;
  }
  return b.pack(X, U, VectorXd::Zero(p.N));
}
} // namespace challenging
