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


// ===================================================== terminal hinge cost
// Smooth exterior penalty at the terminal stage (no slack there): mirrors
// `w_terminal_penalty * fmin(h, 0)**2` in ocp.py. Cost-only, not a hard
// constraint, so it is a plain g2o cost edge, not an Ineq edge.
class EdgeCostHingeObstacle
    : public g2o::BaseFixedSizedEdge<1, double, Vertex_p> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeCostHingeObstacle(std::shared_ptr<TrajParams> param, int j)
      : _param(std::move(param)), _j(j) {}

  void computeError() override {
    const auto &p = static_cast<const Vertex_p *>(_vertices[0])->estimate();
    const Eigen::Vector3d o = _param->obstacles.block<3, 1>(0, _j);
    const double r = _param->obstacles(3, _j);
    const double eps = _param->obstacle_eps;
    _diff = p - o;
    _dist = std::sqrt(_diff.squaredNorm() + eps * eps);
    _error[0] = std::min(_dist - r, 0.0);
  }

  void linearizeOplus() override {
    auto &Jp = std::get<0>(_jacobianOplus);
    if (_dist - _param->obstacles(3, _j) >= 0.0) {
      Jp.setZero();
    } else {
      Jp = (_diff / std::max(_dist, 1e-9)).transpose();
    }
  }

  bool read(std::istream &) override { return true; }
  bool write(std::ostream &) const override { return true; }

private:
  std::shared_ptr<TrajParams> _param;
  int _j;
  Eigen::Vector3d _diff = Eigen::Vector3d::Zero();
  double _dist = 1.0;
};

} // namespace px4nmpc::cg2o
