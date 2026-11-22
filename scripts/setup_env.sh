#!/usr/bin/env bash
# Activate project paths in current shell (like ROS setup.bash).
#   source scripts/setup_env.sh
#   source scripts/setup_env.sh --no-ros
# Safe to re-source, and never searches for g2o/cg2o under /usr/local.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'This file must be sourced: source scripts/setup_env.sh' >&2
  exit 2
fi
_nmpc_setup_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_nmpc_setup_ros=1
for _nmpc_arg in "$@"; do
  case "$_nmpc_arg" in
    --no-ros) _nmpc_setup_ros=0 ;;
    --help|-h) echo 'Usage: source scripts/setup_env.sh [--no-ros]'; unset _nmpc_setup_root _nmpc_setup_ros _nmpc_arg; return 0 ;;
    *) echo "Unknown setup argument: $_nmpc_arg" >&2; unset _nmpc_setup_root _nmpc_setup_ros _nmpc_arg; return 2 ;;
  esac
done
export NMPC_ROOT="$_nmpc_setup_root"
export PX4_NMPC_EXTERNALS_DIR="${PX4_NMPC_EXTERNALS_DIR:-$NMPC_ROOT/externals}"
export EXTERNAL_INSTALL_ROOT="${EXTERNAL_INSTALL_ROOT:-$NMPC_ROOT/install/externals}"
export EXTERNAL_BUILD_ROOT="${EXTERNAL_BUILD_ROOT:-$NMPC_ROOT/build/externals}"
export PX4_NMPC_G2O_PREFIX="${PX4_NMPC_G2O_PREFIX:-$EXTERNAL_INSTALL_ROOT/g2o}"
export PX4_NMPC_CG2O_PREFIX="${PX4_NMPC_CG2O_PREFIX:-$EXTERNAL_INSTALL_ROOT/cg2o}"
# temporary files (compilers, colcon, pip, pytest) on the project drive, not on a small root filesystem;
# set PX4_NMPC_KEEP_TMPDIR=1 to keep your own TMPDIR
if [[ "${PX4_NMPC_KEEP_TMPDIR:-0}" != 1 ]]; then
  mkdir -p "$NMPC_ROOT/build/tmp" && export TMPDIR="$NMPC_ROOT/build/tmp"
fi
export PX4_NMPC_IPOPT_PREFIX="${PX4_NMPC_IPOPT_PREFIX:-$EXTERNAL_INSTALL_ROOT/ipopt}"
# IPOPT linear solvers of the local build: SPRAL needs OpenMP cancellation; MKL (PARDISO) single-threaded so every
# solver is timed on one core (the comparison/tuning tools set this for their own processes as well)
export OMP_CANCELLATION="${OMP_CANCELLATION:-TRUE}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-SEQUENTIAL}"
# Intel MKL (optional, used for the paper): when the system BLAS is switched to MKL (update-alternatives
# libblas.so.3 -> libmkl_rt), MKL's own libraries must be on the loader path.
_nmpc_mkl="${MKLROOT:-/opt/intel/oneapi/mkl/latest}/lib/intel64"
if [[ -d "$_nmpc_mkl" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in *":$_nmpc_mkl:"*) ;; *) export LD_LIBRARY_PATH="$_nmpc_mkl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;; esac
fi
unset _nmpc_mkl
for _nmpc_dep in "$PX4_NMPC_G2O_PREFIX" "$PX4_NMPC_CG2O_PREFIX"; do
  case "$_nmpc_dep" in
    /usr|/usr/*) echo "Refusing system g2o/cg2o prefix: $_nmpc_dep" >&2; unset _nmpc_dep; return 1 ;;
  esac
done
unset _nmpc_dep
export PX4_NMPC_CG2O_BUILD_DIR="${PX4_NMPC_CG2O_BUILD_DIR:-$NMPC_ROOT/build/cg2o_backend}"
export PX4_NMPC_IPOPT_CPP_BUILD_DIR="${PX4_NMPC_IPOPT_CPP_BUILD_DIR:-$NMPC_ROOT/build/ipopt_cpp_backend}"
_nmpc_prepend_path() {
  local name="$1" path="$2" previous
  [[ -d "$path" ]] || return 0
  previous="${!name:-}"
  case ":$previous:" in *":$path:"*) return 0 ;; esac
  printf -v "$name" '%s' "$path${previous:+:$previous}"
  export "$name"
}
if [[ "$_nmpc_setup_ros" == 1 ]]; then
  _nmpc_ros_setup="${ROS_SETUP:-/opt/ros/${ROS_DISTRO_NAME:-humble}/setup.bash}"
  if [[ -f "$_nmpc_ros_setup" ]]; then
    # shellcheck disable=SC1090
    source "$_nmpc_ros_setup"
  fi
  if [[ -f "$NMPC_ROOT/install/setup.bash" ]]; then
    # shellcheck disable=SC1090
    source "$NMPC_ROOT/install/setup.bash"
  fi
fi
_nmpc_prepend_path PYTHONPATH "$NMPC_ROOT/src/px4_nmpc_core"
_nmpc_prepend_path PYTHONPATH "$NMPC_ROOT/src/px4_nmpc_controller"
_nmpc_prepend_path PYTHONPATH "$NMPC_ROOT/src/px4_nmpc_compare"
_nmpc_prepend_path PYTHONPATH "$NMPC_ROOT/src/px4_nmpc_tools"
_nmpc_prepend_path PYTHONPATH "$NMPC_ROOT/src/px4_nmpc_bringup"
_nmpc_prepend_path PYTHONPATH "$PX4_NMPC_CG2O_BUILD_DIR"
_nmpc_prepend_path PYTHONPATH "$PX4_NMPC_IPOPT_CPP_BUILD_DIR"
_nmpc_prepend_path PATH "$EXTERNAL_INSTALL_ROOT/Micro-XRCE-DDS-Agent/bin"
_nmpc_prepend_path CMAKE_PREFIX_PATH "$PX4_NMPC_G2O_PREFIX"
_nmpc_prepend_path CMAKE_PREFIX_PATH "$PX4_NMPC_CG2O_PREFIX"
_nmpc_prepend_path LD_LIBRARY_PATH "$PX4_NMPC_G2O_PREFIX/lib"
_nmpc_prepend_path LD_LIBRARY_PATH "$PX4_NMPC_CG2O_PREFIX/lib"
_nmpc_prepend_path PKG_CONFIG_PATH "$PX4_NMPC_G2O_PREFIX/lib/pkgconfig"
_nmpc_prepend_path PKG_CONFIG_PATH "$PX4_NMPC_CG2O_PREFIX/lib/pkgconfig"
_nmpc_prepend_path PKG_CONFIG_PATH "$PX4_NMPC_IPOPT_PREFIX/lib/pkgconfig"
# Explicit package locations for scripts invoked after source.
if [[ -f "$PX4_NMPC_G2O_PREFIX/lib/cmake/g2o/g2oConfig.cmake" ]]; then
  export g2o_DIR="$PX4_NMPC_G2O_PREFIX/lib/cmake/g2o"
else
  unset g2o_DIR
fi
if [[ -f "$PX4_NMPC_CG2O_PREFIX/lib/cmake/cg2o/cg2oConfig.cmake" ]]; then
  export cg2o_DIR="$PX4_NMPC_CG2O_PREFIX/lib/cmake/cg2o"
else
  unset cg2o_DIR
fi
echo "PX4 NMPC environment activated: $NMPC_ROOT"
echo "  g2o : $PX4_NMPC_G2O_PREFIX $([[ -n "${g2o_DIR:-}" ]] && echo '[installed]' || echo '[not installed]')"
echo "  cg2o: $PX4_NMPC_CG2O_PREFIX $([[ -n "${cg2o_DIR:-}" ]] && echo '[installed]' || echo '[not installed]')"
echo "  ipopt: $PX4_NMPC_IPOPT_PREFIX $([[ -f "$PX4_NMPC_IPOPT_PREFIX/lib/pkgconfig/ipopt.pc" ]] && echo '[installed]' || echo '[not installed: system IPOPT is used]')"
unset -f _nmpc_prepend_path
unset _nmpc_setup_root _nmpc_setup_ros _nmpc_ros_setup _nmpc_arg
