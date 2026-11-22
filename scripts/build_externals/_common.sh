#!/usr/bin/env bash
# Internal build utility functions. Source only from scripts/build_externals/*.sh.
# Environment variables follow the original v3.4.0 installer. Never searches /usr/local
# for local g2o/cg2o installations.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNALS="${PX4_NMPC_EXTERNALS_DIR:-$ROOT/externals}"
BUILD_ROOT="${EXTERNAL_BUILD_ROOT:-$ROOT/build/externals}"
# Installs live outside build/, so cleaning the build tree does not remove them.
INSTALL_ROOT="${EXTERNAL_INSTALL_ROOT:-$ROOT/install/externals}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
# FORCE_EXTERNALS=1 rebuilds every enabled component from scratch.
# FORCE_ONLY rebuilds just the listed components (comma or space separated), e.g.
#   FORCE_ONLY=cg2o ./scripts/build_externals.sh
# Valid names: g2o, cg2o, px4, xrce.  Forcing g2o also rebuilds cg2o (it links against g2o).
FORCE="${FORCE_EXTERNALS:-0}"
FORCE_ONLY="${FORCE_ONLY:-}"
BUILD_TYPE="${BUILD_TYPE:-Release}"


FORCE_G2O=$FORCE; FORCE_CG2O=$FORCE; FORCE_PX4=$FORCE; FORCE_XRCE=$FORCE
for _c in ${FORCE_ONLY//,/ }; do
  case "$_c" in
    g2o)  FORCE_G2O=1 ;;
    cg2o) FORCE_CG2O=1 ;;
    px4)  FORCE_PX4=1 ;;
    xrce) FORCE_XRCE=1 ;;
    *) echo "ERROR: unknown FORCE_ONLY component '$_c' (valid: g2o, cg2o, px4, xrce)" >&2; exit 1 ;;
  esac
done
unset _c

# Source directories.  A caller may override any of them from the shell, e.g.
#   CG2O_DIR=/path/to/cg2o ./scripts/build_externals.sh
# Without CG2O_DIR, cg2o is downloaded into externals/cg2o (see CG2O_URL below).
CG2O_USER_DIR="${CG2O_DIR:-}"
CG2O_DIR="${CG2O_DIR:-$EXTERNALS/cg2o}"

G2O_DIR="${G2O_DIR:-$EXTERNALS/g2o}"
PX4_DIR="${PX4_DIR:-$EXTERNALS/PX4-Autopilot}"
QGC_DIR="${QGC_DIR:-$EXTERNALS/QGroundControl}"
XRCE_DIR="${XRCE_DIR:-$EXTERNALS/Micro-XRCE-DDS-Agent}"

# Normalize paths so reuse markers remain stable even when the script is launched
# from a different current working directory.
CG2O_DIR="$(realpath -m "$CG2O_DIR")"
G2O_DIR="$(realpath -m "$G2O_DIR")"
PX4_DIR="$(realpath -m "$PX4_DIR")"
QGC_DIR="$(realpath -m "$QGC_DIR")"
XRCE_DIR="$(realpath -m "$XRCE_DIR")"
BUILD_ROOT="$(realpath -m "$BUILD_ROOT")"
INSTALL_ROOT="$(realpath -m "$INSTALL_ROOT")"
# Fail closed. A typo or override must never install an NMPC g2o/cg2o build
# into the system's /usr tree (unrelated, unoptimized copies may live there).
case "$INSTALL_ROOT" in
  /usr|/usr/*) echo "ERROR: local dependency INSTALL_ROOT must not be under /usr: $INSTALL_ROOT" >&2; exit 1 ;;
esac

# Keep source, build and install trees separate.
G2O_BUILD_DIR="$BUILD_ROOT/g2o"
G2O_INSTALL_DIR="$INSTALL_ROOT/g2o"

CG2O_BUILD_DIR="$BUILD_ROOT/cg2o"
CG2O_INSTALL_DIR="$INSTALL_ROOT/cg2o"

PX4_BUILD_DIR="$BUILD_ROOT/PX4-Autopilot/px4_sitl_default"
PX4_INSTALL_DIR="$INSTALL_ROOT/PX4-Autopilot"

XRCE_BUILD_DIR="$BUILD_ROOT/Micro-XRCE-DDS-Agent"
XRCE_INSTALL_DIR="$INSTALL_ROOT/Micro-XRCE-DDS-Agent"

QGC_INSTALL_DIR="$INSTALL_ROOT/QGroundControl"

# Components.  g2o + cg2o are needed for the cg2o backend and the comparisons.
# PX4 and QGroundControl are only needed for PX4 + Gazebo flights.
#   WITH_PX4=OFF WITH_QGC=OFF ./scripts/build_externals.sh     # solver-only install
WITH_G2O_CG2O="${WITH_G2O_CG2O:-ON}"
WITH_PX4="${WITH_PX4:-ON}"
WITH_QGC="${WITH_QGC:-ON}"
# The Micro XRCE-DDS Agent is the bridge between PX4 and ROS 2 (/fmu/... topics).
# It is not packaged for Ubuntu 22.04, so it is built here.  Follows WITH_PX4 by default.
WITH_XRCE_AGENT="${WITH_XRCE_AGENT:-$WITH_PX4}"
XRCE_REF="${XRCE_REF:-v2.4.2}"

# ROS 2 middleware used by the XRCE Agent.  On Humble we deliberately build
# Agent v2.4.2 against ROS's installed Fast-DDS/Fast-CDR instead of the Agent
# superbuild.  The v2.4.2 superbuild references an old Fast-DDS 2.12.x branch
# which is no longer reliably available upstream.
ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-${ROS_DISTRO:-humble}}"
ROS_PREFIX="${ROS_PREFIX:-/opt/ros/${ROS_DISTRO_NAME}}"
FASTCDR_CMAKE_DIR="${FASTCDR_CMAKE_DIR:-${ROS_PREFIX}/lib/cmake/fastcdr}"
if [[ -z "${FASTRTPS_CMAKE_DIR:-}" ]]; then
  if [[ -f "${ROS_PREFIX}/share/fastrtps/cmake/fastrtps-config.cmake" ]]; then
    FASTRTPS_CMAKE_DIR="${ROS_PREFIX}/share/fastrtps/cmake"
  else
    FASTRTPS_CMAKE_DIR="${ROS_PREFIX}/lib/cmake/fastrtps"
  fi
fi

PX4_REF="${PX4_REF:-v1.17.0}"
PX4_SETUP="${PX4_SETUP:-ON}"
PX4_BUILD="${PX4_BUILD:-ON}"

# ---------------------------------------------------------------------------
# Performance settings
# ---------------------------------------------------------------------------
# -march=native lets Eigen use AVX/AVX2/FMA (or AVX-512) instead of plain SSE2.  This is
# the largest single speed-up for the dense block operations in g2o and cg2o.
# It also changes Eigen's alignment (32/64 bytes), so g2o, cg2o and the px4_nmpc cg2o
# extension MUST all be compiled with the same setting, otherwise the first solve crashes
# with a segmentation fault inside free().  build_cg2o_backend.sh follows whatever g2o was
# built with, so rebuild the extension after changing this.  The binaries then only run on
# CPUs with the same instruction sets as this machine; MARCH_NATIVE=OFF gives a portable build.
if [[ -z "${MARCH_NATIVE+x}" && "${NMPC_FOLLOW_G2O_MARCH:-0}" == "1" && -f "$BUILD_ROOT/g2o/CMakeCache.txt" ]]; then
  if grep -q '^BUILD_WITH_MARCH_NATIVE:BOOL=ON' "$BUILD_ROOT/g2o/CMakeCache.txt"; then MARCH_NATIVE=ON; else MARCH_NATIVE=OFF; fi
fi
MARCH_NATIVE="${MARCH_NATIVE:-OFF}"

# g2o's OpenMP support is marked experimental upstream and is slower for small NMPC graphs.
# Its parallel Hessian assembly is also not deterministic.  Leave OFF unless benchmarked.
G2O_OPENMP="${G2O_OPENMP:-OFF}"
# MKL PARDISO for cg2o (needs Intel oneAPI MKL in /opt/intel/oneapi/mkl/latest).
ENABLE_PARDISO="${ENABLE_PARDISO:-OFF}"

# Deliberately NOT enabled:
#   * -ffast-math / G2O_FAST_MATH: breaks isfinite()/NaN checks that the interior-point
#     solvers and Eigen's allFinite() rely on, and can change iteration counts.
#   * LTO: cg2o is a static library; LTO objects carry compiler-specific bitcode
#     (cg2o forces clang) into the link of the px4_nmpc extension, and across g2o's
#     shared libraries it cannot inline the virtual hot paths anyway.

ARCH_FLAGS=""
if [[ "$MARCH_NATIVE" == "ON" ]]; then
  ARCH_FLAGS="-march=native"
fi

# Ninja builds faster than Make and is what PX4's own Makefile uses.
# A CMAKE_GENERATOR set in the environment wins.
detect_generator_args() {
  GEN_ARGS=()
  if [[ -n "${CMAKE_GENERATOR:-}" ]]; then
    GEN_ARGS=(-G "$CMAKE_GENERATOR")
  elif command -v ninja >/dev/null 2>&1; then
    GEN_ARGS=(-G Ninja)
  fi
}
detect_generator_args

# Versions used for the paper.  Every external is pinned to an exact commit; a tag or branch
# name alone is not enough, because tags can be moved and branches advance.
# g2o master (since Dec 2025) finds CHOLMOD/CSparse only through SuiteSparse's own CMake
# config files (SuiteSparse >= 7).  Ubuntu 22.04 ships SuiteSparse 5.10 without them, so a
# current g2o is configured WITHOUT g2o::solver_cholmod and cg2o then fails to configure.
# This is the revision cg2o's own Dockerfile pins; it uses g2o's bundled FindCholmod.
# Use G2O_REF=master only with SuiteSparse >= 7 (Ubuntu 24.04 or newer).
G2O_REF="${G2O_REF:-eec325a1da1273e87bc97887d49e70570f28570c}"
PX4_COMMIT="${PX4_COMMIT:-d6f12ad1c4f70ad3230afd7d86e971421e02fef4}"     # tag v1.17.0
XRCE_COMMIT="${XRCE_COMMIT:-57d086216d01ec43121845d385894a25987f8a2c}"   # tag v2.4.2

# cg2o: the source linked in the paper.  The archive is checked against a SHA-256 over all
# its files except the top-level README.md (sorted paths, `sha256sum` of each file); a
# download whose code differs from the paper's version stops the build.
#   CG2O_DIR=/path/to/checkout   build a local checkout instead (development; not checked)
#   CG2O_GIT_URL=<url>           clone a Git repository at CG2O_REF instead of the archive
CG2O_URL="${CG2O_URL:-https://anonymous.4open.science/api/repo/cg2o/zip}"
CG2O_TREE_SHA256="${CG2O_TREE_SHA256:-e8b08ad384d953fd273062bcbf21ab86163d694198d3e445b247a7b009a03ff3}"
CG2O_GIT_URL="${CG2O_GIT_URL:-}"
CG2O_REF="${CG2O_REF:-b14da0c969fa856f1daf8c3518b0e6101df9b4f0}"

QGC_VERSION="${QGC_VERSION:-v5.0.8}"
QGC_URL="${QGC_URL:-https://github.com/mavlink/qgroundcontrol/releases/download/${QGC_VERSION}/QGroundControl-x86_64.AppImage}"
QGC_APPIMAGE="$QGC_INSTALL_DIR/QGroundControl-${QGC_VERSION}-x86_64.AppImage"

mkdir -p "$EXTERNALS" "$BUILD_ROOT" "$INSTALL_ROOT"

# Never let a workspace-level `colcon build` recurse into third-party sources or installs.
# g2o, cg2o, PX4 and the XRCE agent are built by this script, not by colcon.
touch "$EXTERNALS/COLCON_IGNORE" "$INSTALL_ROOT/COLCON_IGNORE"

banner() {
  printf '\n============================================================\n'
  printf '%s\n' "$1"
  printf '============================================================\n'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

have_cmake_package() {
  local root="$1"
  local config_name="$2"
  [[ -d "$root" ]] && find "$root" -type f -name "$config_name" -print -quit 2>/dev/null | grep -q .
}

# Every component records a hash of its complete configure line (compiler, generator,
# flags, install prefix, ...) in its build and install markers.  Any change, e.g. a new
# install root or MARCH_NATIVE, triggers a clean rebuild instead of silently reusing a
# build that was made with other settings.
config_hash() {
  printf '%s\n' "${CC:-}" "${CXX:-}" "$@" | sha256sum | cut -c1-16
}

write_marker() {    # write_marker <file> <source_dir> <hash>
  mkdir -p "$(dirname "$1")"
  printf 'source=%s\nbuild_type=%s\nmarch_native=%s\nconfig=%s\n' \
    "$2" "$BUILD_TYPE" "$MARCH_NATIVE" "$3" > "$1"
}

marker_matches() {  # marker_matches <file> <source_dir> <hash>
  [[ -f "$1" ]] || return 1
  grep -Fqx "source=$2" "$1" || return 1
  grep -Fqx "config=$3" "$1" || return 1
}

build_marker_matches() {    # <build_dir> <source_dir> <hash>
  [[ -f "$1/CMakeCache.txt" ]] || return 1
  marker_matches "$1/.px4_nmpc_build_complete" "$2" "$3"
}

mark_build_complete() {     # <build_dir> <source_dir> <hash>
  write_marker "$1/.px4_nmpc_build_complete" "$2" "$3"
}

install_marker_matches() {  # <install_dir> <source_dir> <hash>
  marker_matches "$1/.px4_nmpc_install_complete" "$2" "$3"
}

mark_install_complete() {   # <install_dir> <source_dir> <hash>
  write_marker "$1/.px4_nmpc_install_complete" "$2" "$3"
}

# Start from a clean build directory when the configure line changed (CMake cannot switch
# generators in place and keeps stale cache entries).  An interrupted build with the same
# settings is resumed incrementally.
prepare_build_dir() {       # <build_dir> <hash>
  local stamp="$1/.px4_nmpc_config"
  if [[ -d "$1" && "$(cat "$stamp" 2>/dev/null)" != "$2" ]]; then
    echo "Build settings changed; starting from a clean build directory: $1"
    rm -rf "$1"
  fi
  mkdir -p "$1"
  printf '%s\n' "$2" > "$stamp"
  rm -f "$1/.px4_nmpc_build_complete"
}

# Replace an install tree completely, so no files from an older configuration survive.
cmake_reinstall() {         # <build_dir> <install_dir>
  rm -rf "$2"
  cmake --install "$1"
}

# cg2o links g2o::solver_cholmod and g2o::solver_csparse.  g2o silently skips them when
# SuiteSparse is not found, so check what was really installed.
g2o_has_solvers() {
  local pat
  for pat in solver_cholmod solver_csparse; do
    find "$G2O_INSTALL_DIR" -name "libg2o_${pat}*" -print -quit 2>/dev/null | grep -q . || return 1
  done
}


patch_g2o_for_cg2o() {
  local header="$G2O_DIR/g2o/core/sparse_optimizer.h"

  if [[ ! -f "$header" ]]; then
    echo "ERROR: expected g2o header not found: $header" >&2
    exit 1
  fi

  if grep -q '^  virtual void update(const double\* update);' "$header"; then
    echo "g2o SparseOptimizer::update() is already virtual."
    return 0
  fi

  if grep -q '^  void update(const double\* update);' "$header"; then
    echo "Applying cg2o compatibility patch to g2o SparseOptimizer::update()..."
    sed -i \
      's/^  void update(const double\* update);/  virtual void update(const double* update);/' \
      "$header"
    echo "g2o compatibility patch applied."
    return 0
  fi

  echo "WARNING: SparseOptimizer::update() declaration did not match the expected form." >&2
  echo "Current g2o may already contain an upstream API change." >&2
  echo "Please inspect: $header" >&2
}

# cg2o's UMFPACK/SVD linear solvers write the complete Hessian to ./umfpack_linear_debug/ (about 0.4 MB per file, in the
# current directory of the running process) every time a factorization fails. A flight with repeated failures would fill
# the disk. The patch makes the dump opt-in (export CG2O_DEBUG_LINEAR_DUMP=1). Guarded, safe to rerun.
patch_cg2o_quiet_linear_debug() {
  local f
  for f in "$CG2O_DIR/cg2o/solvers/umfpack/linear_solver_eigen_umfpack_lu.h" \
           "$CG2O_DIR/cg2o/solvers/eigen/linear_solver_eigen_svd.h"; do
    [[ -f "$f" ]] || continue
    if grep -q 'CG2O_DEBUG_LINEAR_DUMP' "$f"; then continue; fi
    # Newer cg2o gates the dump behind the CMake option CG2O_DEBUG_LINEAR_SOLVER (OFF here):
    # nothing to patch, and the research checkout must not be modified.
    if grep -q '#if CG2O_DEBUG_LINEAR_SOLVER' "$f"; then continue; fi
    if grep -q '^    std::string folder_name = "[a-z]*_linear_debug";' "$f"; then
      sed -i 's|^\(    std::string folder_name = "[a-z]*_linear_debug";\)|    if (!std::getenv("CG2O_DEBUG_LINEAR_DUMP")) return;  // px4_nmpc patch: dumps only on request\n\1|' "$f"
      echo "cg2o patch applied (no debug dumps unless CG2O_DEBUG_LINEAR_DUMP is set): ${f#$CG2O_DIR/}"
    else
      echo "WARNING: $f has an unexpected layout; the linear-solver debug-dump patch was not applied." >&2
    fi
  done
}

clone_and_pin() {   # clone_and_pin <url> <dir> <ref>
  local url="$1" dir="$2" ref="$3"

  # Existing source does not need to be a Git checkout.  This is important for
  # local/custom trees such as the unconstrained cg2o source.
  if [[ -f "$dir/CMakeLists.txt" ]]; then
    echo "Using existing source: $dir"
  elif [[ -e "$dir" ]]; then
    echo "ERROR: $dir exists but does not look like a CMake source tree." >&2
    return 1
  else
    git clone "$url" "$dir" || return 1
  fi

  if [[ -n "$ref" ]]; then
    if [[ -d "$dir/.git" ]]; then
      git -C "$dir" fetch --quiet origin || true
      git -C "$dir" checkout --quiet "$ref" || return 1
    else
      echo "WARNING: $dir is not a Git checkout; cannot checkout ref '$ref'." >&2
    fi
  fi
}

# Stop unless a Git checkout is at the expected commit.
require_commit() {  # require_commit <dir> <commit> <name>
  local head
  head="$(git -C "$1" rev-parse HEAD 2>/dev/null)" || die "$3: $1 is not a Git checkout; cannot verify commit $2."
  [[ "$head" == "$2" ]] || die "$3 in $1 is at $head, but the paper used $2. Remove $1 to fetch the pinned version."
}

# SHA-256 over the files of a source tree (sorted paths; .git and the top-level README.md excluded).
source_tree_sha256() {  # source_tree_sha256 <dir>
  (cd "$1" && find . -type f ! -path './.git/*' ! -path './README.md' -print0 | LC_ALL=C sort -z \
     | xargs -0 sha256sum | sha256sum | cut -c1-64)
}

# Fetch the paper's cg2o into $CG2O_DIR: the archive at CG2O_URL (checked against CG2O_TREE_SHA256),
# or a Git clone of CG2O_GIT_URL at CG2O_REF.
fetch_cg2o() {
  [[ ! -e "$CG2O_DIR" ]] || die "$CG2O_DIR exists but is not a cg2o source tree; remove it and run again."
  if [[ -n "$CG2O_GIT_URL" ]]; then
    git clone "$CG2O_GIT_URL" "$CG2O_DIR" || die "Cannot clone $CG2O_GIT_URL."
    git -C "$CG2O_DIR" checkout --quiet "$CG2O_REF" || die "cg2o commit $CG2O_REF not found in $CG2O_GIT_URL."
    return 0
  fi
  command -v unzip >/dev/null || die "unzip is required to unpack cg2o (sudo apt install unzip)."
  local tmp sum
  tmp="$(mktemp -d)"
  echo "Downloading cg2o: $CG2O_URL"
  curl -fsSL --retry 3 -o "$tmp/cg2o.zip" "$CG2O_URL" || { rm -rf "$tmp"; die "Cannot download cg2o from $CG2O_URL."; }
  mkdir -p "$tmp/src" && unzip -q "$tmp/cg2o.zip" -d "$tmp/src" || { rm -rf "$tmp"; die "Cannot unpack the cg2o archive."; }
  # the archive holds the source tree either at its top level or in one folder
  local top="$tmp/src"
  if [[ ! -f "$top/CMakeLists.txt" ]]; then top="$(dirname "$(find "$tmp/src" -maxdepth 2 -name CMakeLists.txt | head -n1)")"; fi
  [[ -f "$top/CMakeLists.txt" ]] || { rm -rf "$tmp"; die "The cg2o archive contains no CMakeLists.txt."; }
  sum="$(source_tree_sha256 "$top")"
  if [[ "$sum" != "$CG2O_TREE_SHA256" ]]; then
    rm -rf "$tmp"
    die "The downloaded cg2o differs from the version used for the paper (tree SHA-256 $sum, expected $CG2O_TREE_SHA256). Set CG2O_DIR to a checkout of the paper's version, or CG2O_TREE_SHA256=$sum to accept this one."
  fi
  mkdir -p "$(dirname "$CG2O_DIR")" && mv "$top" "$CG2O_DIR"
  rm -rf "$tmp"
  echo "cg2o source verified (tree SHA-256 $sum): $CG2O_DIR"
}

check_local_g2o() {
  have_cmake_package "$G2O_INSTALL_DIR" g2oConfig.cmake || die "Local g2o missing at $G2O_INSTALL_DIR; run ./scripts/build_externals/g2o.sh"
  [[ -f "$G2O_BUILD_DIR/CMakeCache.txt" ]] || die "g2o CMakeCache missing at $G2O_BUILD_DIR; cannot verify ABI/optimization flags. Run ./scripts/build_externals/g2o.sh --rebuild"
  grep -q '^G2O_NO_IMPLICIT_OWNERSHIP_OF_OBJECTS:BOOL=ON' "$G2O_BUILD_DIR/CMakeCache.txt" || die "Local g2o lacks G2O_NO_IMPLICIT_OWNERSHIP_OF_OBJECTS=ON; rebuild with scripts/build_externals/g2o.sh --clean"
}
