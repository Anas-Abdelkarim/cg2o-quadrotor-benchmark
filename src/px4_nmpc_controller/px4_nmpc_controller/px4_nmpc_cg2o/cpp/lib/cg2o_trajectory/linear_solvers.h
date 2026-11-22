#pragma once
// Linear solvers for the cg2o KKT systems, selected by key at run time (cg2o.yaml:
// linear_solver.type). Only solvers that can factor the
// indefinite KKT matrix of a constrained factor graph (or its sigma-regularised quasi-definite
// version) are listed.
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>

#include <memory>
#include <string>
#include <vector>

namespace px4nmpc::cg2o {

using BlockMatrix = g2o::BlockSolverX::PoseMatrixType;
using LinearSolverBase = g2o::LinearSolver<BlockMatrix>;

struct LinearSolverInfo {
  std::string key;
  std::string name;
  bool needs_regularization = false;  // LDL^T / Cholesky without pivoting: needs sigma > 0
  bool available = false;
  std::string note;
};

const std::vector<LinearSolverInfo> &linearSolverCatalog();
//! accepts a key ("qdldlt") or the legacy names "eigenumfpacklu" / "14" (UMFPACK)
std::string canonicalLinearSolverKey(const std::string &key_or_alias);
std::unique_ptr<LinearSolverBase> makeLinearSolver(const std::string &key);

} // namespace px4nmpc::cg2o
