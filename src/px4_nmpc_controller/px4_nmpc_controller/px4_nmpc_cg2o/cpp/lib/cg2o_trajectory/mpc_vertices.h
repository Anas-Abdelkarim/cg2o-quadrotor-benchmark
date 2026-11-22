#ifndef PX4_NMPC_CG2O_MPC_VERTICES_H_
#define PX4_NMPC_CG2O_MPC_VERTICES_H_

// Vertex types for the point-mass/jerk ("trajectory" interface) NMPC factor
// graph. Modeled on the VertexMPC<D> pattern used by the reference drone
// factor graph in learn.zip, simplified because every quantity here is a
// plain Euclidean vector (no quaternion), so we do not need the D==1
// scalar specialisation that pattern used for e.g. thrust.

#include <Eigen/Core>
#include <g2o/core/base_vertex.h>

namespace px4nmpc::cg2o {

template <int D>
class VertexMPC : public g2o::BaseVertex<D, Eigen::Matrix<double, D, 1>> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Base = g2o::BaseVertex<D, Eigen::Matrix<double, D, 1>>;
  using EstimateType = typename Base::EstimateType;

  void setToOriginImpl() override { this->_estimate.setZero(); }

  void oplusImpl(const double *update) override {
    Eigen::Map<const Eigen::Matrix<double, D, 1>> delta(update);
    this->_estimate += delta;
  }

  bool write(std::ostream &os) const override {
    os << this->_estimate.transpose();
    return os.good();
  }

  bool read(std::istream &is) override {
    for (int i = 0; i < D; ++i) {
      is >> this->_estimate[i];
    }
    return is.good();
  }
};

// States: position, velocity, acceleration (all NED, metres/seconds).
class Vertex_p : public VertexMPC<3> {};
class Vertex_v : public VertexMPC<3> {};
class Vertex_a : public VertexMPC<3> {};

// Input: jerk.
class Vertex_j : public VertexMPC<3> {};

// Slack triple per stage: [obstacle, arena, dynamic-limit]. Matches the
// NS=3 slack layout used by px4_nmpc_core.ocp.NmpcOcpSolver.
class Vertex_s : public VertexMPC<3> {};

// Optimised yaw (PointMassJerkYawModel): unwrapped yaw angle, yaw rate (states) and yaw acceleration (input).
class Vertex_psi : public VertexMPC<1> {};
class Vertex_om : public VertexMPC<1> {};
class Vertex_al : public VertexMPC<1> {};

} // namespace px4nmpc::cg2o

#endif // PX4_NMPC_CG2O_MPC_VERTICES_H_
