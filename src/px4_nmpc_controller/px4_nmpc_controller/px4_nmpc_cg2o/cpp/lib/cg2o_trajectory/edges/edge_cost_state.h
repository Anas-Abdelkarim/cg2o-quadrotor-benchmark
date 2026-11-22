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


// ========================================================== cost: state/accel
// Combines position+velocity tracking with the acceleration regularizer
// (extra_stage_cost in models.py). Reused for both the mid-horizon stage
// cost and the terminal cost by passing different weights/reference column.
class EdgeCostState
    : public g2o::BaseFixedSizedEdge<9, double, Vertex_p, Vertex_v, Vertex_a> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeCostState(std::shared_ptr<TrajParams> param, int ref_col)
      : _param(std::move(param)), _ref_col(ref_col) {}

  void computeError() override {
    const auto *vp = static_cast<const Vertex_p *>(_vertices[0]);
    const auto *vv = static_cast<const Vertex_v *>(_vertices[1]);
    const auto *va = static_cast<const Vertex_a *>(_vertices[2]);
    _error.segment<3>(0) = vp->estimate() - _param->pref.col(_ref_col);
    _error.segment<3>(3) = vv->estimate() - _param->vref.col(_ref_col);
    _error.segment<3>(6) = va->estimate();
  }

  void linearizeOplus() override {
    auto &Jp = std::get<0>(_jacobianOplus);
    auto &Jv = std::get<1>(_jacobianOplus);
    auto &Ja = std::get<2>(_jacobianOplus);
    Jp.setZero();
    Jv.setZero();
    Ja.setZero();
    Jp.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Jv.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();
    Ja.block<3, 3>(6, 0) = Eigen::Matrix3d::Identity();
  }

  bool read(std::istream &) override { return true; }
  bool write(std::ostream &) const override { return true; }

private:
  std::shared_ptr<TrajParams> _param;
  int _ref_col;
};

// ================================================================ cost: input

} // namespace px4nmpc::cg2o
