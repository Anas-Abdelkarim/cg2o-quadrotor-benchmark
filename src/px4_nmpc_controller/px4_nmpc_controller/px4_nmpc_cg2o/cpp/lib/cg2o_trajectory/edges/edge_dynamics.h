#pragma once
#include "mpc_parameters.h"
#include "cg2o/core/base_fixed_sized_edge_eq.h"
#include "cg2o/core/base_fixed_sized_edge_ineq.h"
#include "mpc_vertices.h"
#include <Eigen/Core>
#include <g2o/core/base_fixed_sized_edge.h>
#include <algorithm>
#include <cmath>
#include <memory>

namespace px4nmpc::cg2o {


// ============================================================ dynamics (Eq)
// Exact triple-integrator discretisation; residual is affine, so the
// Jacobian is constant. Vertex order: p_{k+1}, v_{k+1}, a_{k+1}, p_k, v_k,
// a_k, j_k (matches PointMassJerkModel.discrete in models.py).
class EdgeDynamics : public ::cg2o::BaseFixedSizedEdgeEq<
                         9, double, Vertex_p, Vertex_v, Vertex_a, Vertex_p,
                         Vertex_v, Vertex_a, Vertex_j> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit EdgeDynamics(std::shared_ptr<TrajParams> param)
      : _param(std::move(param)) {}

  void computeEq() override {
    const double dt = _param->dt;
    const auto *vp1 = static_cast<const Vertex_p *>(_vertices[0]);
    const auto *vv1 = static_cast<const Vertex_v *>(_vertices[1]);
    const auto *va1 = static_cast<const Vertex_a *>(_vertices[2]);
    const auto *vp0 = static_cast<const Vertex_p *>(_vertices[3]);
    const auto *vv0 = static_cast<const Vertex_v *>(_vertices[4]);
    const auto *va0 = static_cast<const Vertex_a *>(_vertices[5]);
    const auto *vj0 = static_cast<const Vertex_j *>(_vertices[6]);

    const Eigen::Vector3d &p0 = vp0->estimate();
    const Eigen::Vector3d &v0 = vv0->estimate();
    const Eigen::Vector3d &a0 = va0->estimate();
    const Eigen::Vector3d &j0 = vj0->estimate();

    _eq.segment<3>(0) =
        p0 + v0 * dt + a0 * dt * dt / 2.0 + j0 * dt * dt * dt / 6.0 -
        vp1->estimate();
    _eq.segment<3>(3) = v0 + a0 * dt + j0 * dt * dt / 2.0 - vv1->estimate();
    _eq.segment<3>(6) = a0 + j0 * dt - va1->estimate();
  }

  void linearizeOplus() override {
    initializeJacobians();
    const double dt = _param->dt;
    const int D = _eq.size();
    auto &&Jp1 = std::get<0>(_jacobianOplus).topRows(D);
    auto &&Jv1 = std::get<1>(_jacobianOplus).topRows(D);
    auto &&Ja1 = std::get<2>(_jacobianOplus).topRows(D);
    auto &&Jp0 = std::get<3>(_jacobianOplus).topRows(D);
    auto &&Jv0 = std::get<4>(_jacobianOplus).topRows(D);
    auto &&Ja0 = std::get<5>(_jacobianOplus).topRows(D);
    auto &&Jj0 = std::get<6>(_jacobianOplus).topRows(D);

    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    Jp1.block<3, 3>(0, 0) = -I3;
    Jp0.block<3, 3>(0, 0) = I3;
    Jv0.block<3, 3>(0, 0) = dt * I3;
    Ja0.block<3, 3>(0, 0) = (dt * dt / 2.0) * I3;
    Jj0.block<3, 3>(0, 0) = (dt * dt * dt / 6.0) * I3;

    Jv1.block<3, 3>(3, 0) = -I3;
    Jv0.block<3, 3>(3, 0) = I3;
    Ja0.block<3, 3>(3, 0) = dt * I3;
    Jj0.block<3, 3>(3, 0) = (dt * dt / 2.0) * I3;

    Ja1.block<3, 3>(6, 0) = -I3;
    Ja0.block<3, 3>(6, 0) = I3;
    Jj0.block<3, 3>(6, 0) = dt * I3;
  }

private:
  std::shared_ptr<TrajParams> _param;
};

// ========================================================== cost: state/accel
// Combines position+velocity tracking with the acceleration regularizer
// (extra_stage_cost in models.py). Reused for both the mid-horizon stage
// cost and the terminal cost by passing different weights/reference column.

} // namespace px4nmpc::cg2o
