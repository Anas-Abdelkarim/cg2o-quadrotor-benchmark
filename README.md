# cg2o in a real-time quadrotor MPC

Code and data of the quadrotor benchmark (Sec. VII) of the paper *cg2o: Constrained Factor Graph Optimization via
an Infeasible-Start Primal–Dual Interior-Point Method*. The solver cg2o is a separate library:
<https://anonymous.4open.science/r/cg2o/>.

A nonlinear MPC sends collective thrust and body rates to PX4 (software-in-the-loop with Gazebo) while the
quadrotor flies through an obstacle course. The problem is described in Sec. VII of the paper.

## Results (Table XI of the paper)

20 PX4 SITL flights, all reaching the goal; the recorded MPC problems re-solved offline by cg2o and IPOPT.

| Start | Solver | Iterations | Mean time | Within 100 ms | Feasible |
| --- | --- | --- | --- | --- | --- |
| Cold (1407 problems) | IPOPT | 51.1 | 176.7 ms | 14.5% | 88.8% |
| | **cg2o** | **17.5** | **46.7 ms** | **98.2%** | **100%** |
| Warm (1425 problems) | IPOPT | 35.1 | 144.8 ms | 39.4% | 95.5% |
| | **cg2o** | **7.7** | **21.7 ms** | **99.2%** | **100%** |

Videos of a warm- and a cold-started flight: [`videos/`](videos/README.md).

## Reproduce

Tables IX to XII and Figs. 10 and 11 from the shipped measurements in [`data/`](data/README.md):

```bash
python3 -m venv --system-site-packages venv
venv/bin/pip install pip==26.2.2 setuptools==79.0.1
venv/bin/pip install -r requirements.txt
venv/bin/python experiments/run_experiment.py tables
```

Re-solving the problems, new tuning studies and new PX4 flights: [QUICKSTART.md](QUICKSTART.md). Installation:
[INSTALL.md](INSTALL.md). Third-party software: [THIRD_PARTY.md](THIRD_PARTY.md).
