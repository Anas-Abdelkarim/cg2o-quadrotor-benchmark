# 1. The benchmark problem

Source of truth: [`course.yaml`](../src/px4_nmpc_config/config/experiments/challenging_course/course.yaml)
(course, vehicle, MPC) and [`benchmark/model/model.h`](../benchmark/model/model.h) (dynamics, cost,
constraints). Both solvers are built from this single model.

## Vehicle and loop

* Gazebo x500 quadrotor (2.06 kg) in PX4 v1.17 SITL.
* The MPC runs at 10 Hz and publishes collective thrust and body-rate setpoints
  (`VehicleRatesSetpoint`); PX4 runs only its rate controller.
* The predicted vehicle includes PX4's rate loop as a first-order lag with time constants
  0.095 s (roll), 0.070 s (pitch) and 0.390 s (yaw), identified from SITL flights.

## Optimal control problem

* State x = [p, v, φ, θ, ψ, ω, s] (13): position and velocity (NED), Euler angles, body rates, progress
  s along the guide line.
* Input u = [T, ω_sp, v_s] (5): collective thrust, body-rate setpoints, progress speed.
* Discretization: RK4, Δt = 0.1 s, N = 40 (4 s horizon).
* Guide line c(s) = [s, 0, −h(s)], h(s) = 2.5 − sin(πs/14): a straight line in North, an altitude wave
  between 1.5 m and 3.5 m, from s = −14 m to the goal at s = 14 m.
* Cost (all least-squares residuals): lag and contour error of the position to c(s), terminal versions of
  both, heading along the path tangent, small attitude, thrust and rate terms, a progress reward,
  input slew, and an L2 penalty on obstacle slacks.

## Obstacles and airframe

| Type | Count | Passing options | Constraint row |
| --- | --- | --- | --- |
| Sphere | 4 | sideways, over, under | 3-D distance |
| Vertical post | 4 (two form a slot) | sideways only | horizontal distance |
| Horizontal bar across the corridor | 2 | over or under only | distance in the North–Down plane |

The airframe is five spheres: the centre (0.12 m) and four rotors (0.16 m) at their positions rotated by
the predicted attitude. Each pair of body sphere and obstacle gives the row

    1 − d² / ρ² − σ ≤ 0,   ρ = r_obstacle + r_body + 0.40 m,

with a slack σ ≥ 0 and a 0.40 m reserve for tracking errors. With 10 obstacles, 5 body spheres and 40
stages there are 2000 obstacle rows.

Other constraints: exact tilt cone cos φ cos θ ≥ cos 40°, thrust between 0.25 and 0.9·maximum,
body-rate setpoints within ±1.5 rad/s, speed limit, arena walls, floor and ceiling.

Problem size: 760 variables, 520 equality rows (dynamics and initial state), 2800 inequality rows.

## Starts

* **Cold:** a rollout along the guide line that ignores the obstacles; identical for both solvers. In the
  recorded flights it violates on average 75 obstacle rows, in 91% of the problems.
* **Warm:** each solver's own previous solution, shifted by the elapsed steps.

## Design choices

The settings below were fixed in PX4/Gazebo flights before the tuning and the campaign; the reasons are also noted
next to each value in `course.yaml`.

| Choice | Reason |
| --- | --- |
| Rate loop as a first-order lag (0.095, 0.070, 0.390 s) | fitted to rate setpoint vs. measured body rate in a PX4/Gazebo log (R² 0.96 / 0.95 / 0.89); the yaw loop is four to five times slower than roll and pitch, which a model with instantaneous rates ignores |
| Obstacle reserve 0.40 m | with 0.15 m a propeller touched a sphere, with 0.30 m the airframe came within 0.02 m of a post |
| Body-rate limit 1.5 rad/s | with 2.5 rad/s the setpoint steps were larger than the x500 rate loop could follow |
| Heading weight 10 | with 1.0 the yaw swung by ±60° in PX4 |
| Slew weights [0.1, 3, 3, 1, 1] | smaller weights gave thrust and rate steps (e.g. 27.8 → 6.5 N within 0.2 s) that PX4 tracked with a large lag |
| Euler angles, not quaternions | the tilt stays within 40°, far from the singularity at 90° pitch; a quaternion would add a norm constraint without benefit |
| Thrust as an input, not a state | thrust smoothness is already enforced by the slew term; a thrust state with a thrust-rate input would add a jerk-like layer |
| Fixed obstacles, all of them in every problem | no nearest-obstacle selection and no moving obstacles, so every recorded problem can be rebuilt exactly from its hash |
