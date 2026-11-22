#ifndef PX4_NMPC_CG2O_MPC_FORMULATION_H_
#define PX4_NMPC_CG2O_MPC_FORMULATION_H_

#include "mpc_parameters.h"
#include "solver_settings.h"

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace px4nmpc::cg2o {

// The trajectory-interface NMPC factor graph (vertices + edges, see mpc_edges.h), built once
// and re-solved every call: only the shared TrajParams and the fixed x0 vertices change between
// calls, so the graph topology and the sparsity pattern are reused for the whole NMPC run.
//
// The constrained optimizer is cg2o's ISPD; makeTrajFormulation() creates the implementation
// (mpc_formulation.cpp) behind this interface.
class TrajFormulationBase {
public:
  virtual ~TrajFormulationBase() = default;

  // Runs up to `max_iterations` iterations.
  //   shift_steps < 0: cold start (fresh reference-tracking rollout, cold duals);
  //   shift_steps >= 0: warm start from the previous solve, shifted by that many stages
  //                     (primal shift by this class, or ISPD's own primal-dual warm start).
  // Returns the iterations taken.
  virtual int solve(int max_iterations, int shift_steps) = 0;
  virtual void resetWarmStart() = 0;

  // 9 x (N+1): rows [pN,pE,pD,vN,vE,vD,aN,aE,aD]; 11 x (N+1) with optimised yaw (+ psi, omega).
  virtual Eigen::MatrixXd statesMatrix() const = 0;
  // 3 x N jerk inputs (4 x N with optimised yaw: + yaw acceleration).
  virtual Eigen::MatrixXd inputsMatrix() const = 0;
  // 3 x N slack triples [obstacle, arena, dynamic] (column 0 is always 0).
  virtual Eigen::MatrixXd slacksMatrix() const = 0;
  // Objective at the RETURNED point, in the same convention as the IPOPT backends.
  virtual double lastCost() = 0;
  virtual const SolveInfo &lastInfo() const = 0;
  virtual std::string optimizerName() const = 0;
  virtual std::string linearSolverName() const = 0;
};

std::unique_ptr<TrajFormulationBase> makeTrajFormulation(std::shared_ptr<TrajParams> param,
                                                         const SolverSettings &settings);

} // namespace px4nmpc::cg2o

#endif // PX4_NMPC_CG2O_MPC_FORMULATION_H_
