#!/usr/bin/env bash
# Build only this project's seven ROS 2 packages.
#
#   ./scripts/build_workspace.sh
#   ./scripts/build_workspace.sh --clean
#   ./scripts/build_workspace.sh --rosdep
#   ./scripts/build_workspace.sh --clean --rosdep
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/$ROS_DISTRO_NAME/setup.bash}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CLEAN=0
RUN_ROSDEP=0

ROS_PACKAGES=(
  px4_msgs
  px4_nmpc_config
  px4_nmpc_core
  px4_nmpc_controller
  px4_nmpc_tools
  px4_nmpc_compare
  px4_nmpc_bringup
)

usage() {
  cat <<EOF
Usage: ./scripts/build_workspace.sh [--clean] [--rosdep]

  --clean   remove previous colcon build/install/log; keep native solver builds
  --rosdep  install ROS package dependencies before building

Environment: ROS_DISTRO_NAME (default: humble), ROS_SETUP, JOBS
EOF
}

die()  { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }
info() { echo "INFO: $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --rosdep) RUN_ROSDEP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$ROS_SETUP" ]] || die "ROS 2 $ROS_DISTRO_NAME not found at $ROS_SETUP. See INSTALL.md."

if [[ "$ROOT" == *"("* || "$ROOT" == *")"* ]]; then
  die "project path contains '(' or ')': $ROOT (move/rename the project to a path without parentheses)"
fi
if [[ "$ROOT" == /mnt/* ]]; then
  warn "project is on a WSL-mounted Windows filesystem: $ROOT"
  warn "this is supported, but native Linux storage (for example ~/px4_nmpc) is usually faster."
fi

# Prevent accidental workspace-wide colcon discovery from entering third-party repositories.
mkdir -p "$ROOT/externals"
touch "$ROOT/externals/COLCON_IGNORE"

# shellcheck disable=SC1090
source "$ROS_SETUP"
command -v colcon >/dev/null 2>&1 || die "colcon not found: sudo apt install python3-colcon-common-extensions"

PY="$(command -v python3)"
[[ -n "$PY" ]] || die "python3 not found"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  cfg="$VIRTUAL_ENV/pyvenv.cfg"
  [[ -f "$cfg" ]] || die "invalid active virtualenv: $VIRTUAL_ENV"
  grep -Eqi '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "$cfg" || \
    die "the active virtualenv hides ROS Python packages. Recreate it with: python3 -m venv --system-site-packages <dir>"
fi

info "Project root : $ROOT"
info "Python       : $PY"
info "ROS distro   : $ROS_DISTRO_NAME"
info "Jobs         : $JOBS"

PIP_ARGS=()
[[ -z "${VIRTUAL_ENV:-}" ]] && PIP_ARGS+=(--user)
"$PY" -m pip install "${PIP_ARGS[@]}" -q -r "$ROOT/requirements.txt"

em_version="$("$PY" -c 'import em; print(getattr(em, "__version__", "unknown"))' 2>/dev/null || true)"
[[ "$em_version" == 3.* ]] || die "ROS 2 Humble requires Empy 3.x, but Python sees '${em_version:-missing}'. requirements.txt pins empy==3.3.4; rerun the pip install and check your environment."
info "Empy        : $em_version"

cd "$ROOT"

if [[ $RUN_ROSDEP -eq 1 ]]; then
  command -v rosdep >/dev/null 2>&1 || die "rosdep not found: sudo apt install python3-rosdep"
  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init
  fi
  rosdep update
  rosdep install --from-paths "$ROOT/src" --ignore-src --rosdistro "$ROS_DISTRO_NAME" -y
fi

if [[ $CLEAN -eq 1 ]]; then
  "$ROOT/scripts/clean_colcon_artifacts.sh"
fi

COLCON_BIN="$(command -v colcon)"
info "colcon       : $COLCON_BIN"
info "Building only: ${ROS_PACKAGES[*]}"

# Invoke the colcon entry point with the same Python interpreter used above.
# This matters when a --system-site-packages virtualenv is active.
"$PY" "$COLCON_BIN" build \
  --base-paths "$ROOT/src" \
  --packages-select "${ROS_PACKAGES[@]}" \
  --symlink-install \
  --parallel-workers "$JOBS" \
  --event-handlers console_cohesion+ \
  --cmake-args "-DPython3_EXECUTABLE=$PY"

[[ -f "$ROOT/install/setup.bash" ]] || die "build finished without install/setup.bash"

cat <<EOF

Workspace build complete.

In every new terminal:
  source $ROS_SETUP
EOF
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "  source $VIRTUAL_ENV/bin/activate"
fi
cat <<EOF
  source $ROOT/install/setup.bash

Checks:
  ros2 pkg prefix px4_msgs
  ros2 pkg list | grep px4_nmpc
  python3 $ROOT/scripts/smoke_test_backends.py

Note: externals/COLCON_IGNORE prevents a plain 'colcon build' from compiling
PX4/g2o/Micro-XRCE-DDS-Agent as ROS workspace packages.
EOF
