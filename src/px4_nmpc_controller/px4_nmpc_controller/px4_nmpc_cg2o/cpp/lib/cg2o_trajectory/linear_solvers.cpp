#include "linear_solvers.h"

#ifdef PX4_NMPC_HAVE_G2O_CHOLMOD
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#endif
#ifdef PX4_NMPC_HAVE_G2O_CSPARSE
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#endif
#ifdef PX4_NMPC_HAVE_CG2O_EIGEN
#include <cg2o/solvers/eigen/linear_solver_eigen_ldlt.h>
#include <cg2o/solvers/eigen/linear_solver_eigen_lu.h>
#endif
#ifdef PX4_NMPC_HAVE_CG2O_UMFPACK
#include <cg2o/solvers/umfpack/linear_solver_eigen_umfpack_lu.h>
#include <cg2o/solvers/umfpack/linear_solver_umfpack_fast.h>
#endif
#ifdef PX4_NMPC_HAVE_CG2O_CHOLMOD_FAST
#include <cg2o/solvers/cholmod/linear_solver_cholmod_fast.h>
#endif
#ifdef PX4_NMPC_HAVE_CG2O_QDLDLT
#include <cg2o/solvers/eigen/linear_solver_eigen_qdldlt.h>
#endif

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace px4nmpc::cg2o {
namespace {
std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}
constexpr bool kG2oCholmod =
#ifdef PX4_NMPC_HAVE_G2O_CHOLMOD
    true;
#else
    false;
#endif
constexpr bool kG2oCsparse =
#ifdef PX4_NMPC_HAVE_G2O_CSPARSE
    true;
#else
    false;
#endif
constexpr bool kEigen =
#ifdef PX4_NMPC_HAVE_CG2O_EIGEN
    true;
#else
    false;
#endif
constexpr bool kUmfpack =
#ifdef PX4_NMPC_HAVE_CG2O_UMFPACK
    true;
#else
    false;
#endif
constexpr bool kCholmodFast =
#ifdef PX4_NMPC_HAVE_CG2O_CHOLMOD_FAST
    true;
#else
    false;
#endif
constexpr bool kQdldlt =
#ifdef PX4_NMPC_HAVE_CG2O_QDLDLT
    true;
#else
    false;
#endif
} // namespace

const std::vector<LinearSolverInfo> &linearSolverCatalog() {
  static const std::vector<LinearSolverInfo> catalog = {
      {"umfpack", "UMFPACK LU (cg2o default)", false, kUmfpack, "pivoted LU, exact KKT step"},
      {"umfpack_fast", "UMFPACK LU, cached symbolic", false, kUmfpack, "cg2o dev"},
      {"qdldlt", "quasi-definite LDL^T, delta 1e-9", false, kQdldlt,
       "own regularisation on zero diagonals + refinement + LU fallback (exact KKT step); "
       "damped step (sigma from 1e-6) only when the KKT matrix is singular"},
      {"qdldlt_s7", "quasi-definite LDL^T, delta 1e-7", false, kQdldlt, "refinement + LU fallback"},
      {"qdldlt_s7_ref", "quasi-definite LDL^T, delta 1e-7, always refine", false, kQdldlt, ""},
      {"qdldlt_nodamp", "quasi-definite LDL^T, delta 1e-9, no adaptive damping", false, kQdldlt,
       "qdldlt before the adaptive damping, for comparisons"},
      {"eigen_lu", "Eigen SparseLU", false, kEigen, ""},
      {"eigen_ldlt", "Eigen SimplicialLDLT", true, kEigen, "needs equality_regularization > 0"},
      {"cholmod_fast", "CHOLMOD (cg2o fast variant)", true, kCholmodFast, "needs equality_regularization > 0"},
      {"cholmod", "CHOLMOD (g2o)", true, kG2oCholmod, "needs equality_regularization > 0"},
      {"csparse", "CSparse (g2o)", true, kG2oCsparse, "needs equality_regularization > 0"},
  };
  return catalog;
}

std::string canonicalLinearSolverKey(const std::string &key_or_alias) {
  const std::string k = lower(key_or_alias);
  if (k == "eigenumfpacklu" || k == "14") return "umfpack";
  if (k == "18") return "qdldlt";
  return k;
}

std::unique_ptr<LinearSolverBase> makeLinearSolver(const std::string &key_in) {
  using M = BlockMatrix;
  const std::string key = canonicalLinearSolverKey(key_in);
  const LinearSolverInfo *info = nullptr;
  for (const auto &s : linearSolverCatalog())
    if (s.key == key) info = &s;
  if (!info) {
    std::string keys;
    for (const auto &s : linearSolverCatalog()) keys += (keys.empty() ? "" : ", ") + s.key;
    throw std::runtime_error("unknown cg2o linear solver '" + key_in + "' (available keys: " + keys + ")");
  }
  if (!info->available)
    throw std::runtime_error("cg2o linear solver '" + key + "' is not compiled into this build");
#ifdef PX4_NMPC_HAVE_CG2O_UMFPACK
  if (key == "umfpack") return std::make_unique<::cg2o::LinearSolverEigenUmfPackLU<M>>();
  if (key == "umfpack_fast") return std::make_unique<::cg2o::LinearSolverUmfpackFast<M>>();
#endif
#ifdef PX4_NMPC_HAVE_CG2O_QDLDLT
  if (key == "qdldlt") return std::make_unique<::cg2o::LinearSolverEigenQDLDLT<M>>();
  if (key == "qdldlt_s7") return std::make_unique<::cg2o::LinearSolverEigenQDLDLT<M>>(1e-7, false);
  if (key == "qdldlt_s7_ref") return std::make_unique<::cg2o::LinearSolverEigenQDLDLT<M>>(1e-7, true, 3);
  if (key == "qdldlt_nodamp") {
    auto solver = std::make_unique<::cg2o::LinearSolverEigenQDLDLT<M>>();
    solver->setAdaptiveDamping(false);
    return solver;
  }
#endif
#ifdef PX4_NMPC_HAVE_CG2O_EIGEN
  if (key == "eigen_lu") return std::make_unique<::cg2o::LinearSolverEigenLU<M>>();
  if (key == "eigen_ldlt") return std::make_unique<::cg2o::LinearSolverEigenLDLT<M>>();
#endif
#ifdef PX4_NMPC_HAVE_CG2O_CHOLMOD_FAST
  if (key == "cholmod_fast") return std::make_unique<::cg2o::LinearSolverCholmodFast<M>>();
#endif
#ifdef PX4_NMPC_HAVE_G2O_CHOLMOD
  if (key == "cholmod") return std::make_unique<g2o::LinearSolverCholmod<M>>();
#endif
#ifdef PX4_NMPC_HAVE_G2O_CSPARSE
  if (key == "csparse") return std::make_unique<g2o::LinearSolverCSparse<M>>();
#endif
  throw std::runtime_error("cg2o linear solver '" + key + "' is not compiled into this build");
}

} // namespace px4nmpc::cg2o
