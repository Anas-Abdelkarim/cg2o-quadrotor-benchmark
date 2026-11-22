#!/usr/bin/env bash
# Build/install g2o only, to the project-local prefix. Idempotent by default.
set -uo pipefail
NMPC_FOLLOW_G2O_MARCH=1
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CLEAN=0; REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1; REBUILD=1 ;;
    --rebuild) REBUILD=1 ;;
    --help|-h) echo 'Usage: ./scripts/build_externals/g2o.sh [--rebuild | --clean]'; exit 0 ;;
    *) die "Unrecognized g2o option: $arg" ;;
  esac
done
if [[ "$FORCE_G2O" == 1 ]]; then CLEAN=1; REBUILD=1; fi
# The original installer skips reuse on --rebuild and rebuilds incrementally.
if [[ "$REBUILD" == 1 && "$CLEAN" == 0 ]]; then CLEAN=1; fi
banner 'Build external: g2o (local only)'
# Release adds -O3 -DNDEBUG (g2o appends another -O3 for GCC).
# -fno-semantic-interposition lets GCC inline and call g2o's own exported functions directly
# inside each shared library instead of going through the PLT (clang already does this).
# Its OpenMP flags are added by g2o itself when G2O_USE_OPENMP=ON.
G2O_CMAKE_ARGS=(
  "${GEN_ARGS[@]}"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_INSTALL_PREFIX="$G2O_INSTALL_DIR"
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  "-DCMAKE_C_FLAGS=$ARCH_FLAGS"
  "-DCMAKE_CXX_FLAGS=$ARCH_FLAGS -fno-semantic-interposition"
  -DBUILD_WITH_MARCH_NATIVE="$MARCH_NATIVE"
  -DG2O_NO_IMPLICIT_OWNERSHIP_OF_OBJECTS=ON
  -DG2O_USE_OPENMP="$G2O_OPENMP"
  -DG2O_FAST_MATH=OFF
  -DG2O_BUILD_APPS=OFF
  -DG2O_BUILD_EXAMPLES=OFF
  -DG2O_USE_OPENGL=OFF
  -DG2O_USE_CHOLMOD=ON
  -DG2O_USE_CSPARSE=ON
  -DBUILD_UNITTESTS=OFF
)
G2O_HASH="$(config_hash "ref=$G2O_REF" "${G2O_CMAKE_ARGS[@]}")"

if [[ "$CLEAN" != "1" ]] \
     && have_cmake_package "$G2O_INSTALL_DIR" "g2oConfig.cmake" && g2o_has_solvers \
     && install_marker_matches "$G2O_INSTALL_DIR" "$G2O_DIR" "$G2O_HASH"; then
  echo "g2o already installed with the current settings at: $G2O_INSTALL_DIR"
  echo "Skipping g2o build."

elif [[ "$CLEAN" != "1" ]] \
     && build_marker_matches "$G2O_BUILD_DIR" "$G2O_DIR" "$G2O_HASH"; then
  echo "g2o is already built at: $G2O_BUILD_DIR"
  echo "Install is missing/stale; installing without rebuilding."
  cmake_reinstall "$G2O_BUILD_DIR" "$G2O_INSTALL_DIR" || die "g2o install failed."
  mark_install_complete "$G2O_INSTALL_DIR" "$G2O_DIR" "$G2O_HASH"

else
  if [[ "$CLEAN" == "1" ]]; then
    echo "Forced rebuild: cleaning g2o build/install."
    rm -rf -- "$G2O_BUILD_DIR" "$G2O_INSTALL_DIR"
  fi

  clone_and_pin https://github.com/RainerKuemmerle/g2o.git "$G2O_DIR" "$G2O_REF" \
    || die "could not get the g2o source."
  # a full commit hash is verified (G2O_REF=master etc. is left to the user)
  if [[ "$G2O_REF" =~ ^[0-9a-f]{40}$ ]]; then require_commit "$G2O_DIR" "$G2O_REF" g2o; fi

  # cg2o compatibility:
  # Some cg2o constrained optimizers override SparseOptimizer::update().
  # On affected g2o revisions this method is non-virtual, so make it virtual.
  patch_g2o_for_cg2o

  prepare_build_dir "$G2O_BUILD_DIR" "$G2O_HASH"

  cmake -S "$G2O_DIR" -B "$G2O_BUILD_DIR" "${G2O_CMAKE_ARGS[@]}" \
    || die "g2o configuration failed."
  cmake --build "$G2O_BUILD_DIR" --parallel "$JOBS" \
    || die "g2o compilation failed."
  mark_build_complete "$G2O_BUILD_DIR" "$G2O_DIR" "$G2O_HASH"
  cmake_reinstall "$G2O_BUILD_DIR" "$G2O_INSTALL_DIR" || die "g2o install failed."
  g2o_has_solvers || die "g2o was built without the CHOLMOD/CSparse solvers (cg2o needs them).
  Check the 'Enable support for Cholmod/CSparse' lines in the g2o configure output above.
  Install the dev packages:  sudo apt install libsuitesparse-dev
  On Ubuntu 22.04 use the default G2O_REF=eec325a1; newer g2o needs SuiteSparse >= 7."
  mark_install_complete "$G2O_INSTALL_DIR" "$G2O_DIR" "$G2O_HASH"
  fi


echo "g2o ready: $G2O_INSTALL_DIR"
