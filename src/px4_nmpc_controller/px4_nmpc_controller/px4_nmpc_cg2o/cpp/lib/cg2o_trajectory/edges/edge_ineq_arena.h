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


// ================================================================ ineq: arena
// axis in {0,1,2} = N,E,D. is_upper selects p[axis] <= ub vs lb <= p[axis].
class EdgeIneqArena
    : public ::cg2o::BaseFixedSizedEdgeIneq<1, double, Vertex_p, Vertex_s> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeIneqArena(std::shared_ptr<TrajParams> param, int axis, bool is_upper)
      : _param(std::move(param)), _axis(axis), _is_upper(is_upper) {}

  void computeIneq() override {
    const auto &p = static_cast<const Vertex_p *>(_vertices[0])->estimate();
    const auto &s = static_cast<const Vertex_s *>(_vertices[1])->estimate();
    if (_is_upper) {
      _ineq[0] = p[_axis] - _param->arena_ub[_axis] - s[1];
    } else {
      _ineq[0] = _param->arena_lb[_axis] - p[_axis] - s[1];
    }
  }

  void linearizeOplus() override {
    auto &Jp = std::get<0>(_jacobianOplus);
    auto &Js = std::get<1>(_jacobianOplus);
    Jp.setZero();
    Jp[_axis] = _is_upper ? 1.0 : -1.0;
    Js << 0.0, -1.0, 0.0;
  }

private:
  std::shared_ptr<TrajParams> _param;
  int _axis;
  bool _is_upper;
};

// ==================================================== ineq: dynamic limits
// Velocity magnitude, tilt cone and vertical-accel limits, all softened by
// the shared s_dynamic slack. Mirrors PointMassJerkModel.soft_path_constraints.

} // namespace px4nmpc::cg2o
