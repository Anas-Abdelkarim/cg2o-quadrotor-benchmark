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


// ==================================================== ineq: dynamic limits
// Velocity magnitude, tilt cone and vertical-accel limits, all softened by
// the shared s_dynamic slack. Mirrors PointMassJerkModel.soft_path_constraints.
class EdgeIneqVelMax
    : public ::cg2o::BaseFixedSizedEdgeIneq<1, double, Vertex_v, Vertex_s> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit EdgeIneqVelMax(std::shared_ptr<TrajParams> param)
      : _param(std::move(param)) {}

  void computeIneq() override {
    const auto &v = static_cast<const Vertex_v *>(_vertices[0])->estimate();
    const auto &s = static_cast<const Vertex_s *>(_vertices[1])->estimate();
    const double vmax = _param->vmax;
    _ineq[0] = (v.squaredNorm() - vmax * vmax) / (2.0 * vmax) - s[2];
  }
  void linearizeOplus() override {
    const auto &v = static_cast<const Vertex_v *>(_vertices[0])->estimate();
    auto &Jv = std::get<0>(_jacobianOplus);
    auto &Js = std::get<1>(_jacobianOplus);
    Jv = (v / _param->vmax).transpose();
    Js << 0.0, 0.0, -1.0;
  }

private:
  std::shared_ptr<TrajParams> _param;
};

} // namespace px4nmpc::cg2o
