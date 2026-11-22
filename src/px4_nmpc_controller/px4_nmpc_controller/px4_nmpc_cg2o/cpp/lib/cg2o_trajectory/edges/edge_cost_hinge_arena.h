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


class EdgeCostHingeArena : public g2o::BaseFixedSizedEdge<1, double, Vertex_p> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeCostHingeArena(std::shared_ptr<TrajParams> param, int axis, bool is_upper)
      : _param(std::move(param)), _axis(axis), _is_upper(is_upper) {}

  void computeError() override {
    const auto &p = static_cast<const Vertex_p *>(_vertices[0])->estimate();
    const double h = _is_upper ? (_param->arena_ub[_axis] - p[_axis])
                                : (p[_axis] - _param->arena_lb[_axis]);
    _error[0] = std::min(h, 0.0);
    _active = h < 0.0;
  }

  void linearizeOplus() override {
    auto &Jp = std::get<0>(_jacobianOplus);
    Jp.setZero();
    if (_active) {
      Jp[_axis] = _is_upper ? -1.0 : 1.0;
    }
  }

  bool read(std::istream &) override { return true; }
  bool write(std::ostream &) const override { return true; }

private:
  std::shared_ptr<TrajParams> _param;
  int _axis;
  bool _is_upper;
  bool _active = false;
};

} // namespace px4nmpc::cg2o
