# infeasible_slalom (base flight configuration)

The PX4 launcher of the quadrotor benchmark (`px4_nmpc_tools.sitl_video`, called by `benchmark/flight/run_px4.sh`)
builds every flight on this bundle: arena, take-off, RViz layout, vehicle radius and safety margin (`nmpc.yaml`,
`experiment.yaml`, `scenario_10.json`). The course, the MPC and the solver settings of the benchmark come from
[`../challenging_course/`](../challenging_course) and override what is set here.

The bundle is named after the earlier ten-obstacle slalom study, which is not part of this paper's results.
`study.yaml` and `study_paper.yaml` hold that study's solver variants; the launcher reads them only with
`--study-settings`, which the benchmark does not use.
