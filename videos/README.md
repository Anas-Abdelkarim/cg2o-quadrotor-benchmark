# Videos of the quadrotor benchmark

Two PX4 software-in-the-loop flights with cg2o, recorded with RViz (left) and Gazebo (right) side by side. Both use
the selected settings of Table IX (`data/tuning/*/selected_parameters.json`), the take-off point and heading of
flight 1 of `src/px4_nmpc_config/config/experiments/challenging_course/starts.json` (the start of `warm_01` and
`cold_01`), and the course of Sec. VII.

| File | Start | Length | Outcome |
| --- | --- | --- | --- |
| `quadrotor_warm.mp4` | warm start: each problem from the shifted previous solution | 48 s | goal in 14.8 s, smallest airframe clearance 0.22 m |
| `quadrotor_cold.mp4` | cold start at every problem: the obstacle-infeasible guide-line rollout | 33 s | goal in 14.8 s, smallest airframe clearance 0.26 m |

The title bar names cg2o's configured warm-start strategy; in the cold flight it is overridden by "cold every
solve". RViz shows the guide line, the obstacles with their safety margins, the planned trajectory over the 4 s
horizon and the solver status; Gazebo shows the simulated x500.

Recorded with `benchmark/flight/record_video.sh <run> warm|cold 0.126,-0.716 8.3`; a new recording goes to
`<run>/video_<mode>/slalom_sitl.mp4`.
