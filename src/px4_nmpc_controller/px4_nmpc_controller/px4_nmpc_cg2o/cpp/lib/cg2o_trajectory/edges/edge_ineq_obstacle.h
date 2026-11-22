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


// ============================================================ ineq: obstacle
// -(dist(p, obstacle_j) - radius_j + s_obstacle) <= 0, one edge per obstacle
// slot per stage. `j` indexes TrajParams::obstacles (4 x K: n,e,d,radius).
class EdgeIneqObstacle
    : public ::cg2o::BaseFixedSizedEdgeIneq<1, double, Vertex_p, Vertex_s> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeIneqObstacle(std::shared_ptr<TrajParams> param, int j)
      : _param(std::move(param)), _j(j) {}

  void computeIneq() override {
    const auto &p = static_cast<const Vertex_p *>(_vertices[0])->estimate();
    const auto &s = static_cast<const Vertex_s *>(_vertices[1])->estimate();
    const Eigen::Vector3d o = _param->obstacles.block<3, 1>(0, _j);
    const double r = _param->obstacles(3, _j);
    const double eps = _param->obstacle_eps;
    _diff = p - o;
    _dist = std::sqrt(_diff.squaredNorm() + eps * eps);
    _ineq[0] = r - _dist - s[0];
  }

  void linearizeOplus() override {
    auto &Jp = std::get<0>(_jacobianOplus);
    auto &Js = std::get<1>(_jacobianOplus);
    Jp = (-_diff / std::max(_dist, 1e-9)).transpose();
    Js << -1.0, 0.0, 0.0; // d(ineq)/ds[0] (obstacle slack); s[1],s[2] unused here
  }

private:
  std::shared_ptr<TrajParams> _param;
  int _j;
  Eigen::Vector3d _diff = Eigen::Vector3d::Zero();
  double _dist = 1.0;
};

// ================================================================ ineq: arena
// axis in {0,1,2} = N,E,D. is_upper selects p[axis] <= ub vs lb <= p[axis].

} // namespace px4nmpc::cg2o
