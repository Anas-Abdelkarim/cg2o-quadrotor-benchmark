#!/usr/bin/env bash
# Build everything of this repository on top of the externals (scripts/build_externals.sh):
#   1. the native solver backends of the ROS 2 controller (scripts/build_backends.sh)
#   2. the benchmark extension _challenging (benchmark/model -> build/challenging_mpc): the MPC problem with
#      both solvers, used by the replay, the tuning and the flights
#   3. the ROS 2 workspace (scripts/build_workspace.sh), only if ROS 2 Humble is installed (or --no-ros)
#
#   ./scripts/build_benchmark.sh [--no-ros]
# Any failing step stops the script with a non-zero exit code.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
WITH_ROS=auto
for arg in "$@"; do
  case "$arg" in
    --no-ros) WITH_ROS=OFF ;;
    -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
die() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "$PYTHON" ]] || die "Python not found at $PYTHON; create the virtual environment first (QUICKSTART.md)."
# The build scripts call `python3` (pybind11, patchelf, smoke tests, pip): use the virtual environment's, as if
# it were activated (`source venv/bin/activate` sets the same two variables).
export PATH="$(dirname "$PYTHON"):$PATH"
if [[ -f "$(dirname "$(dirname "$PYTHON")")/pyvenv.cfg" ]]; then export VIRTUAL_ENV="$(dirname "$(dirname "$PYTHON")")"; fi
[[ "$(command -v python3)" == "$(dirname "$PYTHON")/python3" ]] || die "cannot put $(dirname "$PYTHON") first on PATH"
for lib in g2o cg2o ipopt; do
  [[ -d "$ROOT/install/externals/$lib" ]] || die "install/externals/$lib missing; run ./scripts/build_externals.sh --solver-only first."
done

echo "== 1/3 solver backends"
./scripts/build_backends.sh --clean

echo "== 2/3 benchmark extension (build/challenging_mpc)"
# configured from scratch every time: a cached configuration would keep the paths of an earlier IPOPT/cg2o build
rm -rf build/challenging_mpc
cmake -S benchmark/model -B build/challenging_mpc -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PYTHON"
cmake --build build/challenging_mpc -j "${JOBS:-$(nproc)}"
ls build/challenging_mpc/_challenging*.so >/dev/null 2>&1 || die "benchmark extension was not produced"

echo "== 3/3 ROS 2 workspace"
if [[ "$WITH_ROS" == OFF ]]; then
  echo "skipped (--no-ros): the replay, the tuning and the tables do not need it; PX4 flights do."
elif [[ -f "/opt/ros/${ROS_DISTRO_NAME:-humble}/setup.bash" ]]; then
  ./scripts/build_workspace.sh
else
  echo "skipped: ROS 2 ${ROS_DISTRO_NAME:-humble} not installed (needed only for the PX4 flights)."
fi
echo "build complete"
