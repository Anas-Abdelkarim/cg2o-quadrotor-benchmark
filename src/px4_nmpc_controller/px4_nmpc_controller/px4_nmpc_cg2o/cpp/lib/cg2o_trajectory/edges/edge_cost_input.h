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


// ================================================================ cost: input
class EdgeCostInput : public g2o::BaseFixedSizedEdge<3, double, Vertex_j> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void computeError() override {
    _error = static_cast<const Vertex_j *>(_vertices[0])->estimate();
  }
  void linearizeOplus() override {
    std::get<0>(_jacobianOplus) = Eigen::Matrix3d::Identity();
  }
  bool read(std::istream &) override { return true; }
  bool write(std::ostream &) const override { return true; }
};

// ================================================================ cost: slack
// Exact IPOPT penalty l1*s + l2*s^2 represented as shifted least squares:
//   l2 * (s + l1/(2*l2))^2 - l1^2/(4*l2).
// The constant is removed in TrajFormulation::lastCost() for comparable logs.

} // namespace px4nmpc::cg2o
