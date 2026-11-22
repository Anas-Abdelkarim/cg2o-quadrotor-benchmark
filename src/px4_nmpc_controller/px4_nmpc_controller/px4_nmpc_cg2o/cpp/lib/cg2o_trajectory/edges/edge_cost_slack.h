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


// ================================================================ cost: slack
// Exact IPOPT penalty l1*s + l2*s^2 represented as shifted least squares:
//   l2 * (s + l1/(2*l2))^2 - l1^2/(4*l2).
// The constant is removed in TrajFormulation::lastCost() for comparable logs.
class EdgeCostSlack : public g2o::BaseFixedSizedEdge<3, double, Vertex_s> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit EdgeCostSlack(std::shared_ptr<TrajParams> param) : _param(std::move(param)) {}

  void computeError() override {
    const auto &s = static_cast<const Vertex_s *>(_vertices[0])->estimate();
    Eigen::Vector3d shift;
    shift << _param->w_slack_obstacle_l1 / (2.0 * _param->w_slack_obstacle_l2),
             _param->w_slack_arena_l1 / (2.0 * _param->w_slack_arena_l2),
             _param->w_slack_dynamic_l1 / (2.0 * _param->w_slack_dynamic_l2);
    _error = s + shift;
  }
  void linearizeOplus() override {
    std::get<0>(_jacobianOplus) = Eigen::Matrix3d::Identity();
  }
  bool read(std::istream &) override { return true; }
  bool write(std::ostream &) const override { return true; }
private:
  std::shared_ptr<TrajParams> _param;
};

// =============================================== cost: first input increment

} // namespace px4nmpc::cg2o
