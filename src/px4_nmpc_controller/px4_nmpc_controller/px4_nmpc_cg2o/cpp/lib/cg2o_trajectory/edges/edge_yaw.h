#pragma once
// Edges of the optimised-yaw extension (see PointMassJerkYawModel in px4_nmpc_core/models.py).
//   state  psi (yaw, unwrapped), omega (yaw rate)      input alpha (yaw acceleration)
//   psi+ = psi + omega*dt + alpha*dt^2/2               omega+ = omega + alpha*dt
// All quantities are scalars, so each vertex is one-dimensional.
#include "mpc_parameters.h"
#include "cg2o/core/base_fixed_sized_edge_eq.h"
#include "cg2o/core/base_fixed_sized_edge_ineq.h"
#include "mpc_vertices.h"
#include <Eigen/Core>
#include <g2o/core/base_fixed_sized_edge.h>
#include <cmath>
#include <memory>

namespace px4nmpc::cg2o {

// ================================================== yaw dynamics (equality)
// Vertex order: psi_{k+1}, omega_{k+1}, psi_k, omega_k, alpha_k.
class EdgeYawDynamics : public ::cg2o::BaseFixedSizedEdgeEq<
                            2, double, Vertex_psi, Vertex_om, Vertex_psi,
                            Vertex_om, Vertex_al> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit EdgeYawDynamics(std::shared_ptr<TrajParams> param)
      : _param(std::move(param)) {}

  void computeEq() override {
    const double dt = _param->dt;
    const double psi1 = static_cast<const Vertex_psi *>(_vertices[0])->estimate()[0];
    const double om1 = static_cast<const Vertex_om *>(_vertices[1])->estimate()[0];
    const double psi0 = static_cast<const Vertex_psi *>(_vertices[2])->estimate()[0];
    const double om0 = static_cast<const Vertex_om *>(_vertices[3])->estimate()[0];
    const double al0 = static_cast<const Vertex_al *>(_vertices[4])->estimate()[0];
    _eq[0] = psi0 + om0 * dt + al0 * dt * dt / 2.0 - psi1;
    _eq[1] = om0 + al0 * dt - om1;
  }

  void linearizeOplus() override {
    initializeJacobians();
    const double dt = _param->dt;
    const int D = _eq.size();
    auto &&Jpsi1 = std::get<0>(_jacobianOplus).topRows(D);
    auto &&Jom1 = std::get<1>(_jacobianOplus).topRows(D);
    auto &&Jpsi0 = std::get<2>(_jacobianOplus).topRows(D);
    auto &&Jom0 = std::get<3>(_jacobianOplus).topRows(D);
    auto &&Jal0 = std::get<4>(_jacobianOplus).topRows(D);
    Jpsi1(0, 0) = -1.0;
    Jpsi0(0, 0) = 1.0;
    Jom0(0, 0) = dt;
    Jal0(0, 0) = dt * dt / 2.0;
    Jom1(1, 0) = -1.0;
    Jom0(1, 0) = 1.0;
    Jal0(1, 0) = dt;
  }

private:
  std::shared_ptr<TrajParams> _param;
};

// ============================================== yaw-rate limit (soft ineq)
// (omega^2 - rate_max^2) / (2 rate_max) - s_dynamic <= 0, shares the dynamic-limit slack.
class EdgeIneqYawRate
    : public ::cg2o::BaseFixedSizedEdgeIneq<1, double, Vertex_om, Vertex_s> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit EdgeIneqYawRate(std::shared_ptr<TrajParams> param)
      : _param(std::move(param)) {}

  void computeIneq() override {
    const double om = static_cast<const Vertex_om *>(_vertices[0])->estimate()[0];
    const auto &s = static_cast<const Vertex_s *>(_vertices[1])->estimate();
    const double m = _param->yaw_rate_max;
    _ineq[0] = (om * om - m * m) / (2.0 * m) - s[2];
  }
  void linearizeOplus() override {
    const double om = static_cast<const Vertex_om *>(_vertices[0])->estimate()[0];
    auto &Jo = std::get<0>(_jacobianOplus);
    auto &Js = std::get<1>(_jacobianOplus);
    Jo(0, 0) = om / _param->yaw_rate_max;
    Js << 0.0, 0.0, -1.0;
  }

private:
  std::shared_ptr<TrajParams> _param;
};

// =============================================== yaw acceleration box bound
class EdgeIneqYawAccBounds
    : public ::cg2o::BaseFixedSizedEdgeIneq<2, double, Vertex_al> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit EdgeIneqYawAccBounds(std::shared_ptr<TrajParams> param)
      : _param(std::move(param)) {}

  void computeIneq() override {
    const double al = static_cast<const Vertex_al *>(_vertices[0])->estimate()[0];
    _ineq[0] = al - _param->yaw_acc_max;
    _ineq[1] = -al - _param->yaw_acc_max;
  }
  void linearizeOplus() override {
    auto &Ja = std::get<0>(_jacobianOplus);
    Ja(0, 0) = 1.0;
    Ja(1, 0) = -1.0;
  }

private:
  std::shared_ptr<TrajParams> _param;
};

// ================================================ cost: yaw acceleration input
// w_acc_yaw * alpha^2 (information matrix = weight).
class EdgeCostYawInput : public g2o::BaseFixedSizedEdge<1, double, Vertex_al> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void computeError() override {
    _error[0] = static_cast<const Vertex_al *>(_vertices[0])->estimate()[0];
  }
  void linearizeOplus() override { std::get<0>(_jacobianOplus)(0, 0) = 1.0; }
  bool read(std::istream &) override { return true; }
  bool write(std::ostream &) const override { return true; }
};

// ======================================================== cost: heading terms
// Residuals (information = identity):
//   sqrt(hw_k) (psi - psi_ref_k)                  follow the travel / goal heading
//   sqrt(w_lat) (vN sin psi - vE cos psi)         sideways body-frame velocity
//   sqrt(w_rate) omega
// hw_k and psi_ref_k are per-solve data (TrajParams::hdg_w / hdg_ref), so they are applied inside
// the residual and the graph topology never changes between solves.
class EdgeCostHeading
    : public g2o::BaseFixedSizedEdge<3, double, Vertex_v, Vertex_psi, Vertex_om> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeCostHeading(std::shared_ptr<TrajParams> param, int ref_col)
      : _param(std::move(param)), _ref_col(ref_col) {}

  void computeError() override {
    const auto &v = static_cast<const Vertex_v *>(_vertices[0])->estimate();
    const double psi = static_cast<const Vertex_psi *>(_vertices[1])->estimate()[0];
    const double om = static_cast<const Vertex_om *>(_vertices[2])->estimate()[0];
    const double hw = std::max(0.0, _param->hdg_w[_ref_col]);
    _error[0] = std::sqrt(hw) * (psi - _param->hdg_ref[_ref_col]);
    _error[1] = std::sqrt(_param->w_yaw_lateral) * (v[0] * std::sin(psi) - v[1] * std::cos(psi));
    _error[2] = std::sqrt(_param->w_yaw_rate) * om;
  }

  void linearizeOplus() override {
    const auto &v = static_cast<const Vertex_v *>(_vertices[0])->estimate();
    const double psi = static_cast<const Vertex_psi *>(_vertices[1])->estimate()[0];
    const double hw = std::max(0.0, _param->hdg_w[_ref_col]);
    const double sl = std::sqrt(_param->w_yaw_lateral);
    const double s = std::sin(psi), c = std::cos(psi);
    auto &Jv = std::get<0>(_jacobianOplus);
    auto &Jpsi = std::get<1>(_jacobianOplus);
    auto &Jom = std::get<2>(_jacobianOplus);
    Jv.setZero();
    Jpsi.setZero();
    Jom.setZero();
    Jv(1, 0) = sl * s;
    Jv(1, 1) = -sl * c;
    Jpsi(0, 0) = std::sqrt(hw);
    Jpsi(1, 0) = sl * (v[0] * c + v[1] * s);
    Jom(2, 0) = std::sqrt(_param->w_yaw_rate);
  }

  bool read(std::istream &) override { return true; }
  bool write(std::ostream &) const override { return true; }

private:
  std::shared_ptr<TrajParams> _param;
  int _ref_col;
};

} // namespace px4nmpc::cg2o
