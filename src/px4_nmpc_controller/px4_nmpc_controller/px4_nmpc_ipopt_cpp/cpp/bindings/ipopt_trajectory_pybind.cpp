#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "native_ipopt_solver.h"

namespace py = pybind11;
using px4nmpc::ipoptcpp::NativeIpoptTrajectorySolver;
using px4nmpc::ipoptcpp::ProblemSettings;

namespace {
template <typename T>
T getOr(const py::dict& d, const char* key, const T& fallback) {
  if (!d.contains(key)) return fallback;
  return py::cast<T>(d[py::str(key)]);
}
}

PYBIND11_MODULE(cpp_px4_nmpc_ipopt_cpp, m) {
  m.doc() = "Direct C++ IPOPT backend for the PX4 NMPC trajectory OCP (no CasADi)";

  py::class_<NativeIpoptTrajectorySolver>(m, "NativeIpoptTrajectorySolver")
      .def(py::init([](int N, double dt, int K, double g, double vmax,
                       double tilt_max, Eigen::Vector3d jerk_max,
                       double acc_up_max, double acc_down_max,
                       Eigen::Vector3d q_pos, Eigen::Vector3d q_vel,
                       Eigen::Vector3d q_pos_terminal,
                       Eigen::Vector3d q_vel_terminal, double r_jerk,
                       double w_acc, double w_slack_obstacle_l1,
                       double w_slack_obstacle_l2, double w_slack_arena_l1,
                       double w_slack_arena_l2, double w_slack_dynamic_l1,
                       double w_slack_dynamic_l2, double w_terminal_penalty,
                       Eigen::Vector3d r_delta_input_first, int max_iter,
                       double tol, double max_cpu_time, py::dict solver_config,
                       bool yaw, double yaw_rate_max, double yaw_acc_max,
                       double w_yaw_lateral, double w_yaw_rate, double w_yaw_acc) {
        ProblemSettings s;
        s.N=N; s.dt=dt; s.K=K; s.g=g; s.vmax=vmax; s.tilt_max=tilt_max;
        s.jerk_max=jerk_max; s.acc_up_max=acc_up_max; s.acc_down_max=acc_down_max;
        s.q_pos=q_pos; s.q_vel=q_vel; s.q_pos_terminal=q_pos_terminal; s.q_vel_terminal=q_vel_terminal;
        s.r_jerk=r_jerk; s.w_acc=w_acc;
        s.w_slack_obstacle_l1=w_slack_obstacle_l1; s.w_slack_obstacle_l2=w_slack_obstacle_l2;
        s.w_slack_arena_l1=w_slack_arena_l1; s.w_slack_arena_l2=w_slack_arena_l2;
        s.w_slack_dynamic_l1=w_slack_dynamic_l1; s.w_slack_dynamic_l2=w_slack_dynamic_l2;
        s.w_terminal_penalty=w_terminal_penalty; s.r_delta_input_first=r_delta_input_first;
        s.max_iter=max_iter; s.tol=tol; s.max_cpu_time=max_cpu_time;
        s.yaw=yaw; s.yaw_rate_max=yaw_rate_max; s.yaw_acc_max=yaw_acc_max;
        s.w_yaw_lateral=w_yaw_lateral; s.w_yaw_rate=w_yaw_rate; s.w_yaw_acc=w_yaw_acc;
        s.acceptable_tol=getOr<double>(solver_config,"acceptable_tol",s.acceptable_tol);
        s.acceptable_iter=getOr<int>(solver_config,"acceptable_iter",s.acceptable_iter);
        s.mu_strategy=getOr<std::string>(solver_config,"mu_strategy",s.mu_strategy);
        s.linear_solver=getOr<std::string>(solver_config,"linear_solver",s.linear_solver);
        s.hessian_approximation=getOr<std::string>(solver_config,"hessian_approximation",s.hessian_approximation);
        s.print_level=getOr<int>(solver_config,"print_level",s.print_level);
        s.warm_start=getOr<std::string>(solver_config,"warm_start",s.warm_start);
        s.terminal_rollout=getOr<bool>(solver_config,"terminal_rollout",s.terminal_rollout);
        // any further IPOPT option, typed by its Python value (bool -> yes/no)
        if (solver_config.contains("options") && !solver_config["options"].is_none()) {
          for (auto item : py::cast<py::dict>(solver_config["options"])) {
            const std::string key = py::cast<std::string>(item.first);
            py::handle v = item.second;
            if (py::isinstance<py::bool_>(v)) s.string_options[key] = py::cast<bool>(v) ? "yes" : "no";
            else if (py::isinstance<py::int_>(v)) s.integer_options[key] = py::cast<int>(v);
            else if (py::isinstance<py::float_>(v)) s.numeric_options[key] = py::cast<double>(v);
            else s.string_options[key] = py::cast<std::string>(py::str(v));
          }
        }
        return std::make_unique<NativeIpoptTrajectorySolver>(s);
      }),
      py::arg("N"), py::arg("dt"), py::arg("K"), py::arg("g"), py::arg("vmax"),
      py::arg("tilt_max"), py::arg("jerk_max"), py::arg("acc_up_max"), py::arg("acc_down_max"),
      py::arg("q_pos"), py::arg("q_vel"), py::arg("q_pos_terminal"), py::arg("q_vel_terminal"),
      py::arg("r_jerk"), py::arg("w_acc"), py::arg("w_slack_obstacle_l1"),
      py::arg("w_slack_obstacle_l2"), py::arg("w_slack_arena_l1"), py::arg("w_slack_arena_l2"),
      py::arg("w_slack_dynamic_l1"), py::arg("w_slack_dynamic_l2"), py::arg("w_terminal_penalty"),
      py::arg("r_delta_input_first"), py::arg("max_iter"), py::arg("tol"),
      py::arg("max_cpu_time"), py::arg("solver_config"),
      py::arg("yaw")=false, py::arg("yaw_rate_max")=0.8, py::arg("yaw_acc_max")=2.0,
      py::arg("w_yaw_lateral")=1.0, py::arg("w_yaw_rate")=0.05, py::arg("w_yaw_acc")=0.05)
      .def("solve", [](NativeIpoptTrajectorySolver& self,
                        const Eigen::VectorXd& x0,
                        const Eigen::MatrixXd& pref,
                        const Eigen::MatrixXd& vref,
                        const Eigen::MatrixXd& obstacles,
                        const Eigen::Vector3d& arena_lb,
                        const Eigen::Vector3d& arena_ub,
                        const Eigen::Vector3d& uprev,
                        const Eigen::VectorXd& hdg_ref,
                        const Eigen::VectorXd& hdg_w,
                        int warm_shift) {
        // pure C++: release the GIL so that other Python threads (ROS setpoint streaming) keep running
        auto r=[&]{ py::gil_scoped_release release;
                    return self.solve(x0,pref,vref,obstacles,arena_lb,arena_ub,uprev,hdg_ref,hdg_w,warm_shift); }();
        py::dict info; info["warm_primal"]=r.warm_primal; info["warm_duals"]=r.warm_duals;
        return py::make_tuple(r.X,r.U,r.S,r.iterations,r.cost,r.solve_ms,r.status,r.success,info);
      }, py::arg("x0"),py::arg("pref"),py::arg("vref"),py::arg("obstacles"),
         py::arg("arena_lb"),py::arg("arena_ub"),py::arg("uprev"),
         py::arg("hdg_ref")=Eigen::VectorXd(),py::arg("hdg_w")=Eigen::VectorXd(),py::arg("warm_shift")=-1)
      .def("reset_warm_start", &NativeIpoptTrajectorySolver::resetWarmStart)
      .def_property_readonly("n_vars", &NativeIpoptTrajectorySolver::nVars)
      .def_property_readonly("n_cons", &NativeIpoptTrajectorySolver::nCons);
}
