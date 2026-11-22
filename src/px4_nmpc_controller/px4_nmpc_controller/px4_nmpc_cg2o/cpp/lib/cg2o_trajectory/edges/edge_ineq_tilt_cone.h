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


class EdgeIneqTiltCone
    : public ::cg2o::BaseFixedSizedEdgeIneq<1, double, Vertex_a, Vertex_s> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit EdgeIneqTiltCone(std::shared_ptr<TrajParams> param)
      : _param(std::move(param)) {}

  void computeIneq() override {
    const auto &a = static_cast<const Vertex_a *>(_vertices[0])->estimate();
    const auto &s = static_cast<const Vertex_s *>(_vertices[1])->estimate();
    const double g = _param->g;
    const double tan2 = std::tan(_param->tilt_max) * std::tan(_param->tilt_max);
    const double axy2 = a[0] * a[0] + a[1] * a[1];
    _ineq[0] = (axy2 - tan2 * (g - a[2]) * (g - a[2])) / (2.0 * g) - s[2];
  }
  void linearizeOplus() override {
    const auto &a = static_cast<const Vertex_a *>(_vertices[0])->estimate();
    const double g = _param->g;
    const double tan2 = std::tan(_param->tilt_max) * std::tan(_param->tilt_max);
    auto &Ja = std::get<0>(_jacobianOplus);
    auto &Js = std::get<1>(_jacobianOplus);
    Ja[0] = a[0] / g;
    Ja[1] = a[1] / g;
    Ja[2] = tan2 * (g - a[2]) / g;
    Js << 0.0, 0.0, -1.0;
  }

private:
  std::shared_ptr<TrajParams> _param;
};

} // namespace px4nmpc::cg2o
