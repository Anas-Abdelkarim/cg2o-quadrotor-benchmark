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


// =============================================================== ineq: jerk
// |j| <= jerk_max, hard bound (always satisfiable, matches ocp.py's u_bounds).
class EdgeIneqJerkBounds
    : public ::cg2o::BaseFixedSizedEdgeIneq<6, double, Vertex_j> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit EdgeIneqJerkBounds(std::shared_ptr<TrajParams> param)
      : _param(std::move(param)) {}

  void computeIneq() override {
    const auto &j = static_cast<const Vertex_j *>(_vertices[0])->estimate();
    _ineq.segment<3>(0) = j - _param->jerk_max;
    _ineq.segment<3>(3) = -j - _param->jerk_max;
  }
  void linearizeOplus() override {
    auto &Jj = std::get<0>(_jacobianOplus);
    Jj.setZero();
    Jj.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Jj.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
  }

private:
  std::shared_ptr<TrajParams> _param;
};

// ============================================================= ineq: slack>=0

} // namespace px4nmpc::cg2o
