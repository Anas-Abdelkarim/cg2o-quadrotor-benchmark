// cg2o and IPOPT realizations of the challenging nonlinear MPC (model.h).
// Both evaluate the same factor residuals; IPOPT receives the exact
// Lagrangian Hessian by default (nested forward AD).
#include "linear_solvers.h"
#include "model.h"
#include "legacy_settings.h"
#include <IpIpoptApplication.hpp>
#include <IpIpoptData.hpp>
#include <IpTNLP.hpp>
#include <cg2o/core/base_fixed_sized_edge_eq.h>
#include <cg2o/core/base_fixed_sized_edge_ineq.h>
#include <cg2o/core/sparse_optimizer_ispd.h>
#include <chrono>
#include <functional>
#include <g2o/core/base_fixed_sized_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <map>
#include <memory>
#include <tuple>
#include <variant>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace challenging {
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ------------------------------------------------------------------ cg2o graph
template <int D>
class Vertex : public g2o::BaseVertex<D, Eigen::Matrix<double, D, 1>> {
public:
  static constexpr int dim = D;
  void setToOriginImpl() override { this->_estimate.setZero(); }
  void oplusImpl(const double *u) override {
    this->_estimate += Eigen::Map<const Eigen::Matrix<double, D, 1>>(u);
  }
  bool read(std::istream &) override { return false; }
  bool write(std::ostream &) const override { return false; }
};
using VX = Vertex<NX>;
using VU = Vertex<NU>;
using VS = Vertex<1>;
template <int Tag, int D, class... V> struct EdgeBase;
template <int D, class... V> struct EdgeBase<0, D, V...> {
  using type = g2o::BaseFixedSizedEdge<D, double, V...>;
};
template <int D, class... V> struct EdgeBase<1, D, V...> {
  using type = cg2o::BaseFixedSizedEdgeEq<D, double, V...>;
};
template <int D, class... V> struct EdgeBase<2, D, V...> {
  using type = cg2o::BaseFixedSizedEdgeIneq<D, double, V...>;
};
// Tag 0 cost, 1 equality, 2 inequality.
template <int Tag, int D, class... V>
class Edge : public EdgeBase<Tag, D, V...>::type {
  using Base = typename EdgeBase<Tag, D, V...>::type;
public:
  const Factor *f;
  const Params *p;
  Edge(const Factor *f_, const Params *p_) : f(f_), p(p_) {
    if constexpr (Tag == 0)
      this->setInformation(Eigen::Matrix<double, D, D>::Identity());
  }
  template <size_t... I> VectorXd local(std::index_sequence<I...>) const {
    VectorXd z((V::dim + ...));
    int off = 0;
    ((z.segment(off, V::dim) = static_cast<const V *>(this->_vertices[I])->estimate(),
      off += V::dim), ...);
    return z;
  }
  VectorXd local() const { return local(std::index_sequence_for<V...>{}); }
  void computeEq() {
    if constexpr (Tag == 1) this->_eq = value(*f, local(), *p);
  }
  void computeIneq() {
    if constexpr (Tag == 2) this->_ineq = value(*f, local(), *p);
  }
  void computeError() override {
    if constexpr (Tag == 0) this->_error = value(*f, local(), *p);
    else Base::computeError();
  }
  template <size_t... I> void fill(const MatrixXd &J, std::index_sequence<I...>) {
    int off = 0;
    ((std::get<I>(this->_jacobianOplus).topRows(D) = J.middleCols(off, V::dim),
      off += V::dim), ...);
  }
  void linearizeOplus() override {
    if constexpr (Tag == 1) this->initializeJacobians();
    auto [r, J] = derivative(*f, local(), *p);
    fill(J, std::index_sequence_for<V...>{});
  }
  bool read(std::istream &) override { return false; }
  bool write(std::ostream &) const override { return false; }
};
using EDyn = Edge<1, NX, VX, VU, VX>;
using EPath = Edge<0, PATH_RES, VX>;
using EState = Edge<2, STATE_ROWS, VX>;
using EObs = Edge<2, NB, VX, VS>;
using EInCost = Edge<0, NU, VU>;
using EInIneq = Edge<2, INPUT_ROWS, VU>;
using ESlew0 = Edge<0, NU, VU>;
using ESlew = Edge<0, NU, VU, VU>;
using ESlackCost = Edge<0, 1, VS>;
using ESlackIneq = Edge<2, 1, VS>;

struct Result {
  VectorXd z;
  int iterations = 0, status = 0, warm_used = 0;
  int restoration_iterations = 0, regularized_iterations = 0;
  bool converged = false, warm = false, warm_duals = false;
  double ms = 0, callback_ms = 0, hessian_ms = 0, kappa = 0;   // hessian_ms is part of callback_ms
};

// Persistent graph: the cg2o start point keeps its primal/dual iterate between
// MPC calls and shifts it by registered stage sequences.
class CGSession {
  Problem b;
  std::vector<std::shared_ptr<g2o::HyperGraph::Vertex>> vertices;
  std::vector<std::shared_ptr<g2o::HyperGraph::Edge>> edges;
  std::unique_ptr<g2o::OptimizationAlgorithmGaussNewton> alg;
  cg2o::SparseOptimizerISPD opt;
  std::vector<VX *> X;
  std::vector<VU *> U;
  std::vector<VS *> S;
  bool built = false, configured = false;
  slalom::SolverSettings settings;

  int next_id = 0;
  template <class E, int Tag, class... Vs> E *add(int index, Vs *...v) {
    auto e = std::make_shared<E>(&b.factors[index], &b.p);
    int i = 0;
    ((e->setVertex(i++, v)), ...);
    bool ok;
    if constexpr (Tag == 0) ok = opt.addEdge(e.get());
    else if constexpr (Tag == 1) {
      e->setVertexLagrangeMultiplierId(next_id++);   // must precede addEdgeEq
      ok = opt.addEdgeEq(e.get());
    } else ok = opt.addEdgeIneq(e.get());
    if (!ok) throw std::runtime_error("cg2o rejected an edge");
    edges.push_back(e);
    return e.get();
  }
  void build_graph() {
    auto block = std::make_unique<g2o::BlockSolverX>(
        px4nmpc::cg2o::makeLinearSolver(settings.linear_solver));
    alg = std::make_unique<g2o::OptimizationAlgorithmGaussNewton>(std::move(block));
    opt.setAlgorithm(alg.get());
    int id = 0;
    for (int k = 0; k <= b.p.N; k++) {
      auto v = std::make_shared<VX>();
      v->setId(id++);
      v->setEstimate(b.x0);
      v->setFixed(k == 0);
      opt.addVertex(v.get());
      X.push_back(v.get());
      vertices.push_back(v);
    }
    for (int k = 0; k < b.p.N; k++) {
      auto v = std::make_shared<VU>();
      v->setId(id++);
      opt.addVertex(v.get());
      U.push_back(v.get());
      vertices.push_back(v);
    }
    for (int k = 0; k < b.p.N; k++) {
      auto v = std::make_shared<VS>();
      v->setId(id++);
      opt.addVertex(v.get());
      S.push_back(v.get());
      vertices.push_back(v);
    }
    next_id = id;
    std::vector<EDyn *> dyn;
    std::vector<EState *> state_rows;
    std::vector<std::vector<EObs *>> obstacle_rows(b.K());
    std::vector<EInIneq *> input_rows;
    std::vector<ESlackIneq *> slack_rows;
    for (int i = 0; i < int(b.factors.size()); i++) {
      const Factor &f = b.factors[i];
      const int k = f.stage;
      switch (f.kind) {
      case Dynamics: {
        dyn.push_back(add<EDyn, 1>(i, X[k], U[k], X[k + 1]));
        break;
      }
      case PathCost: add<EPath, 0>(i, X[k]); break;
      case StateIneq: state_rows.push_back(add<EState, 2>(i, X[k])); break;
      case ObstacleIneq:
        obstacle_rows[f.obstacle].push_back(add<EObs, 2>(i, X[k], S[k - 1]));
        break;
      case InputCost: add<EInCost, 0>(i, U[k]); break;
      case InputIneq: input_rows.push_back(add<EInIneq, 2>(i, U[k])); break;
      case SlewCost:
        if (f.first) add<ESlew0, 0>(i, U[k]);
        else add<ESlew, 0>(i, U[k], U[k - 1]);
        break;
      case SlackCost: add<ESlackCost, 0>(i, S[k]); break;
      case SlackIneq: slack_rows.push_back(add<ESlackIneq, 2>(i, S[k])); break;
      }
    }
    auto &start = opt.startPoint();
    start.addShiftSequence(X);
    start.addShiftSequence(U);
    start.addShiftSequence(S);
    start.addShiftSequence(dyn);
    start.addShiftSequence(state_rows);
    for (auto &rows : obstacle_rows) start.addShiftSequence(rows);
    start.addShiftSequence(input_rows);
    start.addShiftSequence(slack_rows);
    // After the shift: the new measurement replaces x0 and only the
    // replicated tail is rolled out with the dynamics. The shifted states are
    // kept, so the start has a dynamics defect at the first stage, which the
    // infeasible-start method handles. Re-rolling the whole horizon open loop
    // from x0 amplifies a small attitude error into metres of position error
    // over 4 s and made warm starts worse than cold ones in closed loop.
    start.setPostShiftCallback([this]() {
      X[0]->setEstimate(b.x0);
      const int n = std::max(1, opt.startPoint().options().shift.steps);
      for (int k = std::max(0, b.p.N - n); k < b.p.N; ++k)
        X[k + 1]->setEstimate(step(VectorXd(X[k]->estimate()), VectorXd(U[k]->estimate()), b.p));
    });
    built = true;
  }

public:
  void configure(const py::dict &cfg) {
    settings = slalom::parseSolverSettings(cfg);
    if (settings.optimizer != "ispd" || settings.unconstrained_algorithm != "GN")
      throw std::invalid_argument("challenging MPC: cg2o ISPD with GN only");
    configured = true;
  }
  void reset() { opt.startPoint().invalidate(); }
  // warm=false: cold duals from the given primal point. warm=true: cg2o's own
  // stored iterate shifted by `shift` stages (the given point is ignored).
  Result solve(const Problem &problem, const VectorXd &initial, int budget,
               double tol, bool warm, int shift) {
    if (!configured) throw std::logic_error("CGSession.configure() first");
    if (built && (problem.p.N != b.p.N || problem.K() != b.K()))
      throw std::invalid_argument("persistent graph dimensions changed");
    b.p = problem.p;
    b.x0 = problem.x0;
    b.uprev = problem.uprev;
    if (!built) {
      b.factors = problem.factors;
      build_graph();
    } else {
      // Same factor layout; only stage data (uprev) can change.
      for (size_t i = 0; i < b.factors.size(); i++) b.factors[i].uprev = problem.factors[i].uprev;
    }
    X[0]->setEstimate(b.x0);
    for (int k = 1; k <= b.p.N; k++) X[k]->setEstimate(initial.segment(NX * (k - 1), NX));
    for (int k = 0; k < b.p.N; k++) U[k]->setEstimate(initial.segment(NX * b.p.N + NU * k, NU));
    for (int k = 0; k < b.p.N; k++) S[k]->setEstimate(initial.segment((NX + NU) * b.p.N + k, 1));
    slalom::applyLegacySettings(opt, settings);
    opt.setEpsilonConstraint(tol);
    opt.setEpsilonConvergence(tol);
    // Keep the warm method selected at all times: switching to the Cold
    // method and back discards the point stored by the cold solve. A cold
    // solve only invalidates the stored point (cg2o then starts cold from the
    // given primal point and stores its final iterate for the next call).
    auto &start = opt.startPoint();
    start.select(static_cast<cg2o::ISPDStartMethod>(std::max(1, settings.warm_start_strategy)));
    start.options().shift.enabled = warm;
    start.options().shift.steps = warm ? std::max(0, std::min(b.p.N - 1, shift)) : 0;
    if (!warm) start.invalidate();
    opt.initializeOptimization();
    auto t = Clock::now();
    opt.optimize(budget);
    Result r;
    r.ms = ms_since(t);
    r.iterations = opt.lastIterationCount();
    r.converged = opt.lastConverged();
    r.kappa = opt.lastKappa();
    r.warm = warm;
    r.warm_used = static_cast<int>(start.lastReport().used);
    r.z.resize(b.n());
    for (int k = 1; k <= b.p.N; k++) r.z.segment(NX * (k - 1), NX) = X[k]->estimate();
    for (int k = 0; k < b.p.N; k++) r.z.segment(NX * b.p.N + NU * k, NU) = U[k]->estimate();
    for (int k = 0; k < b.p.N; k++) r.z[(NX + NU) * b.p.N + k] = S[k]->estimate()[0];
    return r;
  }
};

// ------------------------------------------------------------------ IPOPT TNLP
class TNLP : public Ipopt::TNLP {
public:
  const Problem &b;
  VectorXd initial;
  const VectorXd *warm_lambda;
  Result r;
  VectorXd final_lambda;
  bool exact = true;
  bool gauss_newton = false;   // Hessian = 2 sigma J^T J of the cost residuals only (cg2o's model)
  int m = 0, nnz_jac = 0, nnz_h = 0;
  std::vector<int> constraint, row_offset;   // constraint factor ids and their first row
  std::vector<int> h_row, h_col;             // Hessian structure, per factor (a,b) a>=b
  std::vector<std::vector<std::pair<int, int>>> pairs;

  TNLP(const Problem &b_, const VectorXd &z, const VectorXd *lambda, bool exact_, bool gn_ = false)
      : b(b_), initial(z), warm_lambda(lambda), exact(exact_), gauss_newton(gn_) {
    r.z = z;
    for (int i = 0; i < int(b.factors.size()); i++) {
      const Factor &f = b.factors[i];
      if (!f.cost()) {
        constraint.push_back(i);
        row_offset.push_back(m);
        const int rows = rows_of(f.kind);
        m += rows;
        for (int j : f.indices) if (j >= 0) nnz_jac += rows;
      }
      if (gauss_newton && !f.cost()) pairs.push_back({});
      else if (gauss_newton) {
        std::vector<std::pair<int, int>> all;
        for (int a = 0; a < int(f.indices.size()); a++)
          for (int c = 0; c <= a; c++) all.emplace_back(a, c);
        pairs.push_back(all);
      } else pairs.push_back(f.hessian_pairs());
      for (auto [a, c] : pairs.back()) {
        int ia = f.indices[a], ic = f.indices[c];
        if (ia < 0 || ic < 0) continue;
        h_row.push_back(std::max(ia, ic));
        h_col.push_back(std::min(ia, ic));
      }
    }
    nnz_h = int(h_row.size());
  }
  bool get_nlp_info(Ipopt::Index &n, Ipopt::Index &mm, Ipopt::Index &nj,
                    Ipopt::Index &nh, IndexStyleEnum &s) override {
    n = b.n(); mm = m; nj = nnz_jac; nh = exact ? nnz_h : 0; s = C_STYLE;
    return true;
  }
  bool get_bounds_info(Ipopt::Index n, double *l, double *u, Ipopt::Index,
                       double *gl, double *gu) override {
    for (int i = 0; i < n; i++) { l[i] = -1e19; u[i] = 1e19; }
    for (size_t c = 0; c < constraint.size(); c++) {
      const Factor &f = b.factors[constraint[c]];
      for (int i = 0; i < rows_of(f.kind); i++) {
        gl[row_offset[c] + i] = f.kind == Dynamics ? 0. : -1e19;
        gu[row_offset[c] + i] = 0.;
      }
    }
    return true;
  }
  bool get_starting_point(Ipopt::Index n, bool, double *x, bool iz, double *zl,
                          double *zu, Ipopt::Index mm, bool il, double *lambda) override {
    Eigen::Map<VectorXd>(x, n) = initial;
    if (iz) { Eigen::Map<VectorXd>(zl, n).setZero(); Eigen::Map<VectorXd>(zu, n).setZero(); }
    if (il) {
      if (!warm_lambda || warm_lambda->size() != mm) return false;
      Eigen::Map<VectorXd>(lambda, mm) = *warm_lambda;
    }
    return true;
  }
  bool eval_f(Ipopt::Index n, const double *x, bool, double &obj) override {
    auto t = Clock::now();
    Eigen::Map<const VectorXd> z(x, n);
    obj = 0.;
    for (auto &f : b.factors)
      if (f.cost()) obj += value(f, b.local(f, z), b.p).squaredNorm();
    r.callback_ms += ms_since(t);
    return std::isfinite(obj);
  }
  bool eval_grad_f(Ipopt::Index n, const double *x, bool, double *grad) override {
    auto t = Clock::now();
    Eigen::Map<const VectorXd> z(x, n);
    Eigen::Map<VectorXd> g(grad, n);
    g.setZero();
    for (auto &f : b.factors)
      if (f.cost()) {
        auto [v, J] = derivative(f, b.local(f, z), b.p);
        VectorXd d = 2. * J.transpose() * v;
        for (int j = 0; j < d.size(); j++) if (f.indices[j] >= 0) g[f.indices[j]] += d[j];
      }
    r.callback_ms += ms_since(t);
    return g.allFinite();
  }
  bool eval_g(Ipopt::Index n, const double *x, bool, Ipopt::Index, double *out) override {
    auto t = Clock::now();
    Eigen::Map<const VectorXd> z(x, n);
    for (size_t c = 0; c < constraint.size(); c++) {
      const Factor &f = b.factors[constraint[c]];
      auto v = value(f, b.local(f, z), b.p);
      Eigen::Map<VectorXd>(out + row_offset[c], v.size()) = v;
    }
    r.callback_ms += ms_since(t);
    return true;
  }
  bool eval_jac_g(Ipopt::Index n, const double *x, bool, Ipopt::Index, Ipopt::Index,
                  Ipopt::Index *rows, Ipopt::Index *cols, double *vals) override {
    auto t = Clock::now();
    int k = 0;
    if (!vals) {
      for (size_t c = 0; c < constraint.size(); c++) {
        const Factor &f = b.factors[constraint[c]];
        for (int i = 0; i < rows_of(f.kind); i++)
          for (int j : f.indices)
            if (j >= 0) { rows[k] = row_offset[c] + i; cols[k] = j; k++; }
      }
      return true;
    }
    Eigen::Map<const VectorXd> z(x, n);
    for (size_t c = 0; c < constraint.size(); c++) {
      const Factor &f = b.factors[constraint[c]];
      auto [v, J] = derivative(f, b.local(f, z), b.p);
      for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < J.cols(); j++)
          if (f.indices[j] >= 0) vals[k++] = J(i, j);
    }
    r.callback_ms += ms_since(t);
    return true;
  }
  bool eval_h(Ipopt::Index n, const double *x, bool, double sigma, Ipopt::Index,
              const double *lambda, bool, Ipopt::Index, Ipopt::Index *rows,
              Ipopt::Index *cols, double *vals) override {
    if (!exact) return false;
    if (!vals) {
      for (int k = 0; k < nnz_h; k++) { rows[k] = h_row[k]; cols[k] = h_col[k]; }
      return true;
    }
    auto t = Clock::now();
    Eigen::Map<const VectorXd> z(x, n);
    int k = 0;
    size_t c = 0;
    for (int i = 0; i < int(b.factors.size()); i++) {
      const Factor &f = b.factors[i];
      const double *w = nullptr;
      if (!f.cost()) { w = lambda + row_offset[c]; c++; }
      if (pairs[i].empty()) continue;
      MatrixXd H;
      if (gauss_newton) {
        auto [r, J] = derivative(f, b.local(f, z), b.p);
        H = 2. * sigma * J.transpose() * J;
      } else H = hessian(f, b.local(f, z), b.p, w, sigma);
      for (auto [a, d] : pairs[i]) {
        if (f.indices[a] < 0 || f.indices[d] < 0) continue;
        vals[k++] = H(a, d);   // duplicates across factors are summed by IPOPT
      }
    }
    const double h_ms = ms_since(t);
    r.callback_ms += h_ms;
    r.hessian_ms += h_ms;
    return true;
  }
  bool intermediate_callback(Ipopt::AlgorithmMode mode, Ipopt::Index, Ipopt::Number,
                             Ipopt::Number, Ipopt::Number, Ipopt::Number, Ipopt::Number,
                             Ipopt::Number regularization, Ipopt::Number, Ipopt::Number,
                             Ipopt::Index, const Ipopt::IpoptData *,
                             Ipopt::IpoptCalculatedQuantities *) override {
    if (mode == Ipopt::RestorationPhaseMode) r.restoration_iterations++;
    if (regularization > 0.) r.regularized_iterations++;
    return true;
  }
  void finalize_solution(Ipopt::SolverReturn s, Ipopt::Index n, const double *x,
                         const double *, const double *, Ipopt::Index, const double *,
                         const double *lambda, double, const Ipopt::IpoptData *data,
                         Ipopt::IpoptCalculatedQuantities *) override {
    r.z = Eigen::Map<const VectorXd>(x, n);
    r.iterations = data ? data->iter_count() : 0;
    r.status = int(s);
    if (lambda) final_lambda = Eigen::Map<const VectorXd>(lambda, m);
  }
};

using Option = std::variant<std::string, int, double>;
class IPOPTSession {
  VectorXd lambda;
  bool available = false;
  std::map<std::string, Option> options;
  // (kind, stage, obstacle, component) of every constraint row.
  static std::vector<std::tuple<int, int, int, int>> rows(const Problem &b) {
    std::vector<std::tuple<int, int, int, int>> out;
    for (const auto &f : b.factors)
      if (!f.cost())
        for (int i = 0; i < rows_of(f.kind); i++)
          out.emplace_back(int(f.kind), f.stage, f.obstacle, i);
    return out;
  }
  VectorXd shifted(const Problem &b, int shift) const {
    auto meta = rows(b);
    VectorXd out = VectorXd::Zero(meta.size());
    std::map<std::tuple<int, int, int, int>, int> index;
    for (int i = 0; i < int(meta.size()); ++i) index[meta[i]] = i;
    for (int i = 0; i < int(meta.size()); ++i) {
      auto [kind, stage, obstacle, component] = meta[i];
      int last = (kind == StateIneq || kind == ObstacleIneq) ? b.p.N : b.p.N - 1;
      auto it = index.find({kind, std::min(stage + shift, last), obstacle, component});
      if (it != index.end() && it->second < lambda.size()) out[i] = lambda[it->second];
    }
    return out;
  }
public:
  void reset() { available = false; lambda.resize(0); }
  void set_options(const std::map<std::string, Option> &o) { options = o; }
  // warm: the given primal point is the backend's own shifted solution;
  // dual=true additionally starts from the shifted multipliers.
  Result solve(const Problem &b, const VectorXd &z, int budget, double tol,
               bool warm, int shift, bool dual) {
    VectorXd shifted_lambda;
    const bool use = warm && dual && available;
    if (use) shifted_lambda = shifted(b, std::max(0, shift));
    // hessian_approximation: exact | gauss-newton (our option, passed to IPOPT as
    // a user-supplied Hessian) | limited-memory (IPOPT's quasi-Newton)
    bool exact = true, gn = false;
    auto h = options.find("hessian_approximation");
    if (h != options.end()) {
      const auto &mode = std::get<std::string>(h->second);
      gn = mode == "gauss-newton";
      exact = mode == "exact" || gn;
    }
    Ipopt::SmartPtr<TNLP> nlp = new TNLP(b, z, use ? &shifted_lambda : nullptr, exact, gn);
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();
    auto o = app->Options();
    o->SetIntegerValue("print_level", 0);
    o->SetStringValue("sb", "yes");
    o->SetIntegerValue("max_iter", budget);
    o->SetNumericValue("tol", tol);
    o->SetNumericValue("constr_viol_tol", tol);
    o->SetIntegerValue("acceptable_iter", 0);
    o->SetStringValue("linear_solver", "mumps");
    o->SetStringValue("mu_strategy", "adaptive");
    o->SetStringValue("hessian_approximation", "exact");
    for (auto &[key, value] : options) {
      if (key == "hessian_approximation") {
        o->SetStringValue(key, exact ? "exact" : std::get<std::string>(value));
        continue;
      }
      if (auto s = std::get_if<std::string>(&value)) o->SetStringValue(key, *s);
      else if (auto i = std::get_if<int>(&value)) o->SetIntegerValue(key, *i);
      else o->SetNumericValue(key, std::get<double>(value));
    }
    o->SetStringValue("warm_start_init_point", use ? "yes" : "no");
    if (app->Initialize() != Ipopt::Solve_Succeeded)
      throw std::runtime_error("IPOPT initialization failed");
    auto t = Clock::now();
    auto status = app->OptimizeTNLP(nlp);
    Result r = nlp->r;
    r.ms = ms_since(t);
    r.status = int(status);
    r.converged = status == Ipopt::Solve_Succeeded;
    r.warm = warm;
    r.warm_duals = use;
    if (nlp->final_lambda.size() == nlp->m) { lambda = nlp->final_lambda; available = true; }
    else reset();
    return r;
  }
};

// ------------------------------------------------------------------ evaluation
// Common evaluation of a point: objective, max |h|, max g+, and true
// geometric clearance of the five body spheres (no tracking reserve).
py::dict evaluate(const Problem &b, const VectorXd &z) {
  double cost = 0, eq = 0, ineq = 0, obstacle_ineq = 0;
  int obstacle_rows_violated = 0;
  for (auto &f : b.factors) {
    auto v = value(f, b.local(f, z), b.p);
    if (!v.allFinite()) { cost = eq = ineq = std::numeric_limits<double>::infinity(); break; }
    if (f.cost()) cost += v.squaredNorm();
    else if (f.kind == Dynamics) eq = std::max(eq, v.cwiseAbs().maxCoeff());
    else ineq = std::max(ineq, v.maxCoeff());
  }
  auto X = b.states(z);
  double clearance = 1e20, reserve_penetration = 0.;
  for (int k = 1; k <= b.p.N; k++) {
    auto R = rotation<double>(X(6, k), X(7, k), X(8, k));
    for (int i = 0; i < NB; i++) {
      Eigen::Vector3d w = X.col(k).head<3>() + R * b.p.body.row(i).transpose();
      for (int j = 0; j < b.K(); j++) {
        double d = std::sqrt(obstacle_distance2<double>(w, b.p, j));
        double c = d - b.p.obstacles(j, 4) - b.p.body_radius[i];
        clearance = std::min(clearance, c);
        double pen = b.p.margin - c;
        if (pen > 0.) { obstacle_rows_violated++; reserve_penetration = std::max(reserve_penetration, pen); }
      }
    }
  }
  py::dict d;
  d["final_cost"] = cost;
  d["final_max_eq_residual"] = eq;
  d["final_max_ineq_violation"] = ineq;
  d["min_body_clearance_m"] = clearance;
  d["max_reserve_penetration_m"] = reserve_penetration;
  d["obstacle_rows_inside_reserve"] = obstacle_rows_violated;
  d["max_slack"] = b.slacks(z).maxCoeff();
  return d;
}

py::dict result_dict(const Problem &b, const Result &r) {
  py::dict d = evaluate(b, r.z);
  d["z"] = r.z;
  d["iterations"] = r.iterations;
  d["solve_time_ms"] = r.ms;
  d["callback_ms"] = r.callback_ms;
  d["hessian_ms"] = r.hessian_ms;
  d["converged"] = r.converged;
  d["status"] = r.status;
  d["warm"] = r.warm;
  d["warm_duals"] = r.warm_duals;
  d["warm_start_used"] = r.warm_used;
  d["final_kappa"] = r.kappa;
  d["restoration_iterations"] = r.restoration_iterations;
  d["regularized_iterations"] = r.regularized_iterations;
  return d;
}

// Finite-difference audit of the shared derivatives and of the IPOPT callbacks.
py::dict audit(const Problem &b, const VectorXd &z, double h) {
  double jac = 0., hes = 0.;
  for (auto &f : b.factors) {
    VectorXd a = b.local(f, z);
    auto [v, J] = derivative(f, a, b.p);
    for (int j = 0; j < a.size(); j++) {
      VectorXd p = a, m = a;
      p[j] += h; m[j] -= h;
      VectorXd fd = (value(f, p, b.p) - value(f, m, b.p)) / (2 * h);
      jac = std::max(jac, (fd - J.col(j)).cwiseAbs().maxCoeff());
    }
    // Hessian check with unit weights against differences of the gradient.
    VectorXd w = VectorXd::LinSpaced(v.size(), .5, 1.5);
    MatrixXd H = hessian(f, a, b.p, w.data(), 1.);
    auto grad = [&](const VectorXd &q) {
      auto [vv, JJ] = derivative(f, q, b.p);
      return f.cost() ? VectorXd(2. * JJ.transpose() * vv) : VectorXd(JJ.transpose() * w);
    };
    double scale = 1. + H.cwiseAbs().maxCoeff();
    for (int j = 0; j < a.size(); j++) {   // every local variable, not only the curved set
      VectorXd p = a, m = a;
      p[j] += h; m[j] -= h;
      VectorXd fd = (grad(p) - grad(m)) / (2 * h);
      hes = std::max(hes, (fd - H.col(j)).cwiseAbs().maxCoeff() / scale);
    }
  }
  py::dict d;
  d["max_jacobian_fd_error"] = jac;
  d["max_relative_hessian_fd_error"] = hes;
  return d;
}
} // namespace challenging

PYBIND11_MODULE(_challenging, m) {
  using namespace challenging;
  py::class_<Params>(m, "Params")
      .def(py::init<>())
#define FIELD(a) .def_readwrite(#a, &Params::a)
      FIELD(N) FIELD(dt) FIELD(mass) FIELD(gravity) FIELD(drag) FIELD(tau_rate) FIELD(tmin) FIELD(tmax)
      FIELD(rate_xy) FIELD(rate_z) FIELD(vmax) FIELD(cos_tilt) FIELD(vs_max) FIELD(s_goal)
      FIELD(vs_cold) FIELD(e_amp) FIELD(e_freq) FIELD(e_phase) FIELD(alt0) FIELD(alt_amp)
      FIELD(alt_freq) FIELD(alt_phase) FIELD(body) FIELD(body_radius) FIELD(margin)
      FIELD(lower) FIELD(upper) FIELD(w_lag) FIELD(w_lag_terminal) FIELD(w_contour)
      FIELD(w_contour_terminal) FIELD(w_heading) FIELD(w_attitude) FIELD(w_thrust)
      FIELD(w_rate) FIELD(w_progress) FIELD(v_bar) FIELD(w_goal) FIELD(goal_delta) FIELD(w_slew) FIELD(w_slack) FIELD(obstacles)
#undef FIELD
      .def("hover", &Params::hover);
  py::class_<Problem>(m, "Problem")
      .def(py::init<>())
      .def_readwrite("p", &Problem::p)
      .def_readwrite("x0", &Problem::x0)
      .def_readwrite("uprev", &Problem::uprev)
      .def("build", &Problem::build)
      .def("n", &Problem::n)
      .def("states", &Problem::states)
      .def("inputs", &Problem::inputs)
      .def("slacks", &Problem::slacks)
      .def("pack", &Problem::pack)
      .def("evaluate", &evaluate)
      .def("audit", &audit, py::arg("z"), py::arg("h") = 1e-6)
      .def("cold_start", [](const Problem &b) { return cold_start(b); });
  m.attr("NX") = NX;
  m.attr("NU") = NU;
  m.attr("NB") = NB;
  m.attr("IW") = IW;
  m.attr("IS") = IS;
  m.def("rhs", [](const VectorXd &x, const VectorXd &u, const Params &p) { return rhs(x, u, p); });
  m.def("step", [](const VectorXd &x, const VectorXd &u, const Params &p, double h) {
    return step(x, u, p, h);
  });
  m.def("path", [](double s, const Params &p) { return Eigen::Vector3d(path<double>(s, p)); });
  m.def("path_derivative", [](double s, const Params &p) {
    return Eigen::Vector3d(path_derivative<double>(s, p));
  });
  m.def("rotation", [](double a, double b, double c) { return Eigen::Matrix3d(rotation<double>(a, b, c)); });
  m.def("project", &project);
  py::class_<CGSession>(m, "CGSession")
      .def(py::init<>())
      .def("configure", &CGSession::configure)
      .def("reset", &CGSession::reset)
      .def("solve", [](CGSession &s, const Problem &b, const VectorXd &z, int budget,
                       double tol, bool warm, int shift) {
        if (z.size() != b.n() || !z.allFinite()) throw std::invalid_argument("invalid initial point");
        Result r;
        { py::gil_scoped_release release; r = s.solve(b, z, budget, tol, warm, shift); }
        return result_dict(b, r);
      }, py::arg("problem"), py::arg("initial"), py::arg("budget"), py::arg("tol"),
         py::arg("warm") = false, py::arg("shift") = 1);
  py::class_<IPOPTSession>(m, "IPOPTSession")
      .def(py::init<>())
      .def("reset", &IPOPTSession::reset)
      .def("set_options", [](IPOPTSession &s, const py::dict &d) {
        std::map<std::string, Option> o;
        for (auto item : d) {
          auto key = py::cast<std::string>(item.first);
          auto v = item.second;
          if (py::isinstance<py::str>(v)) o[key] = py::cast<std::string>(v);
          else if (py::isinstance<py::bool_>(v)) o[key] = std::string(py::cast<bool>(v) ? "yes" : "no");
          else if (py::isinstance<py::int_>(v)) o[key] = py::cast<int>(v);
          else o[key] = py::cast<double>(v);
        }
        s.set_options(o);
      })
      .def("solve", [](IPOPTSession &s, const Problem &b, const VectorXd &z, int budget,
                       double tol, bool warm, int shift, bool dual) {
        if (z.size() != b.n() || !z.allFinite()) throw std::invalid_argument("invalid initial point");
        Result r;
        { py::gil_scoped_release release; r = s.solve(b, z, budget, tol, warm, shift, dual); }
        return result_dict(b, r);
      }, py::arg("problem"), py::arg("initial"), py::arg("budget"), py::arg("tol"),
         py::arg("warm") = false, py::arg("shift") = 1, py::arg("dual") = true);
}
