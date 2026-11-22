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


// ============================================================= ineq: slack>=0
class EdgeIneqSlackNonneg
    : public ::cg2o::BaseFixedSizedEdgeIneq<3, double, Vertex_s> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void computeIneq() override {
    _ineq = -static_cast<const Vertex_s *>(_vertices[0])->estimate();
  }
  void linearizeOplus() override {
    std::get<0>(_jacobianOplus) = -Eigen::Matrix3d::Identity();
  }
};

// ===================================================== terminal hinge cost
// Smooth exterior penalty at the terminal stage (no slack there): mirrors
// `w_terminal_penalty * fmin(h, 0)**2` in ocp.py. Cost-only, not a hard
// constraint, so it is a plain g2o cost edge, not an Ineq edge.

} // namespace px4nmpc::cg2o
