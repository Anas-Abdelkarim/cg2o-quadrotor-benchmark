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


// =============================================== cost: first input increment
class EdgeCostFirstInputDelta : public g2o::BaseFixedSizedEdge<3, double, Vertex_j> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit EdgeCostFirstInputDelta(std::shared_ptr<TrajParams> param) : _param(std::move(param)) {}
  void computeError() override {
    _error = static_cast<const Vertex_j *>(_vertices[0])->estimate() - _param->uprev;
  }
  void linearizeOplus() override { std::get<0>(_jacobianOplus) = Eigen::Matrix3d::Identity(); }
  bool read(std::istream &) override { return true; }
  bool write(std::ostream &) const override { return true; }
private:
  std::shared_ptr<TrajParams> _param;
};

// ============================================================ ineq: obstacle
// -(dist(p, obstacle_j) - radius_j + s_obstacle) <= 0, one edge per obstacle
// slot per stage. `j` indexes TrajParams::obstacles (4 x K: n,e,d,radius).

} // namespace px4nmpc::cg2o
