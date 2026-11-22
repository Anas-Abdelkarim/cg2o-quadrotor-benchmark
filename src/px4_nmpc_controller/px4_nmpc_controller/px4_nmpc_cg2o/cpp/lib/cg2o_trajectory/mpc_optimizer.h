#pragma once
// The constrained optimizer of the cg2o backend: ISPD, cg2o's infeasible-start primal-dual
// interior-point method (see makeTrajFormulation in mpc_formulation.cpp).
#include "cg2o/core/sparse_optimizer_ispd.h"

namespace px4nmpc::cg2o {
using OptimizerISPD = ::cg2o::SparseOptimizerISPD;
} // namespace px4nmpc::cg2o
