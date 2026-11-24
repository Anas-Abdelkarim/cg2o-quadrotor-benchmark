# Installation

[QUICKSTART.md](QUICKSTART.md) is the short path. This page gives the details, the options and the versions.

## Profiles

| Profile | For | Needs |
| --- | --- | --- |
| **A. Solvers** | replay of the recorded problems, tuning, tables and figures | Ubuntu 22.04, the packages below |
| **B. Full system** | A + PX4 SITL flights with Gazebo, ROS 2 nodes, RViz, video | A + ROS 2 Humble; PX4 is built by the scripts |

```text
 1  system packages          sudo apt install ...                                  (A, B)
 2  Python packages          venv/bin/pip install -r requirements.txt              (A, B), after upgrading pip
 3  third-party software     ./scripts/build_externals.sh --solver-only            IPOPT, HSL, g2o, cg2o (A, B)
                             ./scripts/build_externals.sh --only=px4,xrce          PX4 SITL, XRCE-DDS agent (B)
 4  this repository          ./scripts/build_benchmark.sh                          backends, benchmark extension, ROS 2 workspace
 5  verify                   source scripts/setup_env.sh && make test
```

`make externals build test` runs steps 3 to 5 for profile B. Every script can be run again: what is already built
is reused; `--clean` (or `FORCE_EXTERNALS=1`) rebuilds. A failing component stops the script with a non-zero exit
code.

## Platform

| Component | Version |
| --- | --- |
| OS | Ubuntu 22.04 LTS, x86-64 |
| Compilers | GCC 12 (C, C++), gfortran 11 (Ubuntu packages); the paper's build |
| Python | 3.10 (system Python), packages pinned in `requirements*.txt` |
| ROS 2 | Humble (profile B) |
| BLAS/LAPACK | Intel oneAPI MKL 2024.0 for the paper (optional; system BLAS/LAPACK otherwise) |

Paths: a native Linux path is fastest. Paths that contain `(` or `)` fail in ROS/CMake-generated commands.

## Versions

Every external is pinned to the exact revision used for the paper. Tags and branches are only used to fetch;
the scripts then check the commit and stop if it differs.

| Component | Version | Commit / check | Script |
| --- | --- | --- | --- |
| cg2o | the paper's version, from the link in the paper | SHA-256 of the source tree `e8b08ad3…` (= commit `b14da0c9`) | `build_externals/cg2o.sh` |
| g2o | 2024-12-14 | `eec325a1da1273e87bc97887d49e70570f28570c` | `build_externals/g2o.sh` |
| IPOPT | 3.14.21 (stable/3.14) | `cc3307136498793f38d51f7a02f70652d724083a` | `build_externals/ipopt.sh` |
| ThirdParty-Mumps | 3.0.14 | `9b96a7e2c3b46457a57d73123c5d89a32e0e85fc` | `build_externals/ipopt.sh` |
| ThirdParty-ASL | 2.0.2 | `6f3e1bb5593b8d900d6232aafd58bcd7a20c2b7a` | `build_externals/ipopt.sh` |
| ThirdParty-HSL | 2.2.6 | `ec509dcf102525216985d4cf8b61996028bc155d` | `build_externals/ipopt.sh` |
| SPRAL | v2024.05.08 | `1a71375f9b3bf97245d80d169aff7e4a7db8ede9` | `build_externals/ipopt.sh` |
| coinbrew | 2026-10-18 | `89cdf805dbbf3f3e296a3cec53a607b824f9be1c` | `build_externals/ipopt.sh` |
| HSL (coinhsl) | 2024.05.15, licensed | SHA-256 of the source tree `e30a7afe…` (a note if it differs) | `build_externals/hsl.sh` |
| PX4-Autopilot | v1.17.0 | `d6f12ad1c4f70ad3230afd7d86e971421e02fef4` | `build_externals/px4.sh` |
| Micro XRCE-DDS Agent | v2.4.2 | `57d086216d01ec43121845d385894a25987f8a2c` | `build_externals/xrce.sh` |
| meson (SPRAL, HSL) | 1.12.1 | pip | `build_externals/{ipopt,hsl}.sh` |

The variables at the top of `scripts/build_externals/_common.sh` and `ipopt.sh` (`G2O_REF`, `IPOPT_COMMIT`,
`CG2O_TREE_SHA256`, ...) override a pin.

## 1. System packages

Profile A:

```bash
sudo apt update
sudo apt install -y git curl unzip cmake ninja-build build-essential gfortran pkg-config \
  python3-pip python3-venv python3-dev \
  libeigen3-dev libspdlog-dev libfmt-dev libsuitesparse-dev libmetis-dev libomp-dev \
  liblapack-dev libblas-dev libhwloc-dev
```

Profile B additionally needs ROS 2 Humble. Set up its apt repository following
<https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html>, then:

```bash
sudo apt install -y ros-humble-desktop ros-humble-fastcdr ros-humble-fastrtps \
  python3-colcon-common-extensions python3-rosdep ffmpeg \
  libfuse2 libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor0 \
  gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl
```

PX4's own installer (run by `build_externals/px4.sh`) installs Gazebo and the PX4 toolchain.

## 2. Python packages

```bash
python3 -m venv --system-site-packages venv
venv/bin/pip install pip==26.2.2 setuptools==79.0.1
venv/bin/pip install -r requirements.txt
```

The virtual environment is created with the system packages so that the ROS 2 Python modules stay visible
(profile B). Packages installed with `pip install --user` are visible too; if they differ from the pinned versions,
run with `PYTHONNOUSERSITE=1` (the PX4 launcher does this already). `requirements-offline.txt` is the subset needed
for the tables and figures alone.

## 3. Third-party software

```bash
HSL_SOURCE=/path/to/coinhsl-2024.05.15.tar.gz MARCH_NATIVE=ON ./scripts/build_externals.sh --solver-only
./scripts/build_externals.sh --only=px4,xrce                              # profile B
```

Sources go to `externals/`, builds to `build/externals/`, installs to `install/externals/`; nothing is installed
system-wide.

* **IPOPT** is built with coinbrew from the pinned commits, with MUMPS and SPRAL, and with PARDISO if MKL is found.
  HSL is *not* compiled into IPOPT: IPOPT loads MA27/MA57 at run time from the HSL library below (option `hsllib`),
  as for the paper. Compiling HSL into IPOPT (`IPOPT_HSL_SOURCE=...`) changes IPOPT's iterations.
* **HSL** cannot be downloaded by a script (licence, free for academic use:
  <https://licences.stfc.ac.uk/product/coin-hsl>). `HSL_SOURCE` takes the archive or the unpacked folder.
  `build_externals/hsl.sh` builds `install/externals/hsl/lib/x86_64-linux-gnu/libcoinhsl.so`, which IPOPT loads
  at run time (option `hsllib`). Without `HSL_SOURCE` it is skipped with a note; the IPOPT runs of the benchmark
  then need `run_experiment.py replay --no-hsl` (IPOPT with MUMPS; IPOPT's numbers differ from the paper).
* **cg2o** is downloaded from the link given in the paper (`CG2O_URL`) and checked against `CG2O_TREE_SHA256`, a
  SHA-256 over all files of the source tree except the top-level README. A different download stops the build.
  To build a local checkout instead (development), set `CG2O_DIR=/path/to/cg2o`; the script then reports whether
  it matches the paper's version. To clone a Git repository at a commit, set `CG2O_GIT_URL` (and `CG2O_REF`).
* **g2o** gets a one-line compatibility patch (`SparseOptimizer::update()` made virtual) and is built with
  `G2O_NO_IMPLICIT_OWNERSHIP_OF_OBJECTS=ON`, CHOLMOD and CSparse.
* **`MARCH_NATIVE=ON`** compiles g2o and cg2o with `-march=native`, as for the paper. The benchmark extension is
  always compiled with `-march=native`, and Eigen's memory alignment must be the same in all of them; the scripts
  refuse to mix the two. `make externals` sets it. The binaries then run only on CPUs with the same instruction
  sets.

Single components: `./scripts/build_externals/{ipopt,hsl,g2o,cg2o,px4,xrce}.sh [--clean]`.

## 4. This repository

```bash
./scripts/build_benchmark.sh            # --no-ros: skip the ROS 2 workspace
```

1. `scripts/build_backends.sh --clean`: the native solver backends of the ROS 2 controller (cg2o and IPOPT);
2. `benchmark/model` → `build/challenging_mpc/_challenging*.so`: the MPC problem of the benchmark with both
   solvers, used by the flights, the replay and the tuning;
3. `scripts/build_workspace.sh`: the seven ROS 2 packages under `src/` (profile B, if ROS 2 Humble is installed).

After changing C++ code, run the script again. To work on cg2o itself:
`CG2O_DIR=/absolute/path/to/cg2o ./scripts/rebuild_cg2o.sh` rebuilds cg2o and the extensions incrementally.

## 5. Verify

```bash
source scripts/setup_env.sh
make test                                   # six suites, one process each; no ROS needed
python3 scripts/smoke_test_backends.py      # one real solve with each backend
```

`make test` runs each suite in its own process because the extensions link different IPOPT builds.

## Running

* Experiments: `python3 experiments/run_experiment.py tables|replay|tuning|full [--quick]` (see QUICKSTART.md).
  It checks the Python packages, the benchmark extension, HSL and (for `full`) PX4 before starting.
* Timed steps (replay, tuning) run on `--cpus` (default `0-3`), one at a time; the MPC node of the PX4 flights
  runs on `--controller-cpus` (default `2,3`). The paper used CPUs 0-3 isolated with the kernel option
  `isolcpus=0-3`. On other CPUs the ratios between the solvers reproduce, the absolute times do not.
* The `make` targets of [docs/3_reproduce.md](docs/3_reproduce.md) run the same steps one by one.

## Troubleshooting

| Problem | Remedy |
| --- | --- |
| `The downloaded cg2o differs from the version used for the paper` | the archive behind the link changed; build the paper's version from a checkout with `CG2O_DIR=...`, or accept the new one with the `CG2O_TREE_SHA256` printed |
| `... is at <commit>, but the paper used <commit>` | an existing checkout in `externals/` is at another revision: remove that folder and run again |
| `HSL library missing` from `run_experiment.py` | build HSL: `HSL_SOURCE=... ./scripts/build_externals/hsl.sh` |
| `MARCH_NATIVE=... but local g2o uses ...` | g2o was built with the other setting: `MARCH_NATIVE=ON ./scripts/build_externals.sh --solver-only --clean` |
| segmentation fault in the first solve | g2o, cg2o and the extensions were compiled with different `-march` settings; rebuild as above |
| a test suite aborts or exits with 127 when run together with another | run `make test`, which runs every suite in its own process |
| `build backend is missing the 'build_editable' hook` | the venv's pip is too old: `venv/bin/pip install pip==26.2.2 setuptools==79.0.1` |
| ROS/colcon errors about `empy` | `empy==3.3.4` is required by ROS 2 Humble (pinned in requirements.txt) |
| `/tmp` full during the build | set `TMPDIR` to a folder with space; `source scripts/setup_env.sh` uses `build/tmp` |

`bash scripts/check_build_env.sh` checks the path, Python, `empy` and snap-related variables.
