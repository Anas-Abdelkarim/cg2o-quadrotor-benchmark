#!/usr/bin/env bash
# Build and install only cg2o, against this workspace's locally installed g2o.
# Without CG2O_DIR the paper's cg2o is downloaded and verified (see CG2O_URL in _common.sh).
# Use: [CG2O_DIR=/path/to/cg2o] ./scripts/build_externals/cg2o.sh [--clean]
# The standalone cg2o repository has install/export rules; a custom 'example'
# project needs equivalent CMake install rules to be consumed by find_package(cg2o).
set -uo pipefail
NMPC_FOLLOW_G2O_MARCH=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

G2O_CMAKE_DIR="$G2O_INSTALL_DIR/lib/cmake/g2o"

[[ -f "$G2O_CMAKE_DIR/g2oConfig.cmake" ]] || \
  die "g2oConfig.cmake not found: $G2O_CMAKE_DIR"

CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --clone) ;;   # accepted for compatibility: the source is fetched when missing
    -h|--help)
      echo 'Usage: [CG2O_DIR=/path/to/cg2o] ./scripts/build_externals/cg2o.sh [--clean]'
      exit 0 ;;
    *) die "Unrecognized cg2o option: $arg" ;;
  esac
done
[[ "${FORCE_CG2O:-0}" == 1 ]] && CLEAN=1

# The local prefix (NOT /usr/local, NOT the externals/g2o source directory).
G2O_INSTALL_DIR="${G2O_INSTALL_DIR:-${EXTERNAL_INSTALL_ROOT:-$ROOT/install/externals}/g2o}"
G2O_INSTALL_DIR="$(realpath -m "$G2O_INSTALL_DIR")"
G2O_CMAKE_DIR="$G2O_INSTALL_DIR/lib/cmake/g2o"
if [[ ! -f "$G2O_CMAKE_DIR/g2oConfig.cmake" ]]; then
  # Permit lib64 layouts but do not fall back to a system g2o installation.
  G2O_CONFIG_FILE="$(find "$G2O_INSTALL_DIR" -type f -name g2oConfig.cmake -print -quit 2>/dev/null || true)"
  [[ -n "$G2O_CONFIG_FILE" ]] || die "Local g2oConfig.cmake not found under $G2O_INSTALL_DIR. Build/install g2o first."
  G2O_CMAKE_DIR="$(dirname "$G2O_CONFIG_FILE")"
fi
check_local_g2o

G2O_HASH="$(sed -n 's/^config=//p' "$G2O_INSTALL_DIR/.px4_nmpc_install_complete" 2>/dev/null | head -n1 || true)"
if [[ -z "$G2O_HASH" ]]; then
  [[ -f "$G2O_BUILD_DIR/CMakeCache.txt" ]] || die "Missing local g2o marker and CMakeCache.txt: $G2O_BUILD_DIR"
  echo 'Note: g2o has no NMPC install stamp; using its local CMakeCache hash.'
  G2O_HASH="manual-$(sha256sum "$G2O_BUILD_DIR/CMakeCache.txt" | cut -c1-16)"
fi

[[ -f "$G2O_BUILD_DIR/CMakeCache.txt" ]] || die "Cannot verify g2o native-architecture mode: missing $G2O_BUILD_DIR/CMakeCache.txt"
G2O_NATIVE="$(sed -n 's/^BUILD_WITH_MARCH_NATIVE:BOOL=//p' "$G2O_BUILD_DIR/CMakeCache.txt" | head -n1)"
[[ -n "$G2O_NATIVE" ]] || die "Local g2o CMakeCache lacks BUILD_WITH_MARCH_NATIVE"
[[ "$G2O_NATIVE" == "$MARCH_NATIVE" ]] || die "MARCH_NATIVE=$MARCH_NATIVE but local g2o uses $G2O_NATIVE. Source scripts/setup_env.sh or match g2o."

if [[ ! -f "$CG2O_DIR/CMakeLists.txt" ]]; then
  # An explicit CG2O_DIR must exist; it is never replaced by a download.
  [[ -z "$CG2O_USER_DIR" ]] || die "cg2o CMake source missing: $CG2O_DIR (from CG2O_DIR). Unset CG2O_DIR to download the paper's version."
  fetch_cg2o
fi
CG2O_DIR="$(realpath -m "$CG2O_DIR")"
if [[ -n "$CG2O_USER_DIR" ]]; then
  _sum="$(source_tree_sha256 "$CG2O_DIR")"
  if [[ "$_sum" == "$CG2O_TREE_SHA256" ]]; then echo "cg2o checkout matches the paper's version: $CG2O_DIR"
  else echo "NOTE: cg2o checkout $CG2O_DIR differs from the paper's version (tree SHA-256 $_sum); results may differ."; fi
  unset _sum
fi

# The research 'unconstrained' source is project(example); upstream is project(cg2o).
# They use different CMake option names. Do not silently ignore build switches.
if grep -Eq '^[[:space:]]*project[[:space:]]*\([[:space:]]*example([[:space:]]|\))' "$CG2O_DIR/CMakeLists.txt"; then
  CG2O_LAYOUT=example
elif grep -Eq '^[[:space:]]*project[[:space:]]*\([[:space:]]*cg2o([[:space:]]|\))' "$CG2O_DIR/CMakeLists.txt"; then
  CG2O_LAYOUT=standalone
else
  die "Unknown cg2o source layout at $CG2O_DIR: expected project(cg2o ...) or project(example ...)."
fi

banner "Build external: cg2o ($CG2O_LAYOUT, incremental)"
printf '  g2o prefix:  %s\n  g2o config:  %s\n  cg2o source: %s\n  cg2o install:%s\n' \
  "$G2O_INSTALL_DIR" "$G2O_CMAKE_DIR/g2oConfig.cmake" "$CG2O_DIR" "$CG2O_INSTALL_DIR"
CG2O_CMAKE_ARGS=(
  "${GEN_ARGS[@]}"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_INSTALL_PREFIX="$CG2O_INSTALL_DIR"
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_PREFIX_PATH="$G2O_INSTALL_DIR"
  # cg2o's FindOrBuildG2O.cmake prefers its own bundled install/externals/g2o; g2o_DIR wins over that search
  -Dg2o_DIR="$G2O_CMAKE_DIR"
  -DG2O_INCLUDE_DIRS="$G2O_CMAKE_DIR/include"
  -DG2O_LIBRARY_DIRS="$G2O_CMAKE_DIR/lib"
  "-DCMAKE_CXX_FLAGS=$ARCH_FLAGS"
  -DCG2O_DEBUG_LINEAR_SOLVER=OFF
  -DUSE_G2O_SOLVERS=ON
)
if [[ "$CG2O_LAYOUT" == standalone ]]; then
  CG2O_CMAKE_ARGS+=(
    -DCG2O_BUILD_EXAMPLES=OFF
    -DCG2O_BUILD_PARDISO="$ENABLE_PARDISO"
    -DCG2O_BUILD_EIGEN_SOLVERS=ON
    -DCG2O_MARCH_NATIVE="$MARCH_NATIVE"
  )
else
  # Names used by the custom unconstrained/CMakeLists.txt of the research checkout.
  CG2O_CMAKE_ARGS+=(
    -DG2O_BUILD_EXAMPLES=OFF
    -DG2O_BUILD_PARDISO="$ENABLE_PARDISO"
    -DG2O_BUILD_UMFPACK=ON
    -DG2O_INCLUDE_DIRS="$G2O_INSTALL_DIR/include"
    -DG2O_LIBRARY_DIRS="$G2O_INSTALL_DIR/lib"
  )
  if [[ "$MARCH_NATIVE" != ON ]] && grep -Fq 'set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")' "$CG2O_DIR/CMakeLists.txt"; then
    die "Custom cg2o CMakeLists.txt forces -march=native even when g2o has MARCH_NATIVE=OFF. Fix its CMAKE_CXX_FLAGS_RELEASE before building."
  fi
  echo 'NOTE: The custom example project needs CMake install/export rules for cg2oConfig.cmake.'
fi

CG2O_HASH="$(config_hash "source=$CG2O_DIR" "layout=$CG2O_LAYOUT" "g2o=$G2O_HASH" "${CG2O_CMAKE_ARGS[@]}")"
patch_cg2o_quiet_linear_debug
if [[ "$CLEAN" == 1 ]]; then
  echo 'Cleaning only cg2o build and install directories (g2o untouched).'
  rm -rf -- "$CG2O_BUILD_DIR" "$CG2O_INSTALL_DIR"
fi

if [[ -f "$CG2O_BUILD_DIR/CMakeCache.txt" ]] \
   && [[ "$(cat "$CG2O_BUILD_DIR/.px4_nmpc_config" 2>/dev/null || true)" == "$CG2O_HASH" ]]; then
  echo 'Reusing cg2o CMake configuration; compiling changed source incrementally.'
else
  prepare_build_dir "$CG2O_BUILD_DIR" "$CG2O_HASH"
  cmake -S "$CG2O_DIR" -B "$CG2O_BUILD_DIR" "${CG2O_CMAKE_ARGS[@]}" || die 'cg2o CMake configuration failed.'
fi

# Fail if CMake somehow selected a different, system g2o despite g2o_DIR above.
SELECTED_G2O_DIR="$(sed -n 's/^g2o_DIR:[^=]*=//p' "$CG2O_BUILD_DIR/CMakeCache.txt" | head -n1)"
[[ "$SELECTED_G2O_DIR" == "$G2O_CMAKE_DIR" ]] || die "CMake selected unexpected g2o_DIR: '$SELECTED_G2O_DIR' (wanted '$G2O_CMAKE_DIR')."

cmake --build "$CG2O_BUILD_DIR" --parallel "$JOBS" || die 'cg2o compilation failed.'
mark_build_complete "$CG2O_BUILD_DIR" "$CG2O_DIR" "$CG2O_HASH"
cmake --install "$CG2O_BUILD_DIR" || die 'cg2o install failed.'
if ! have_cmake_package "$CG2O_INSTALL_DIR" cg2oConfig.cmake; then
  die "cg2o compiled, but the source project installed no cg2oConfig.cmake to $CG2O_INSTALL_DIR. If CG2O_DIR points at unconstrained/, fix its CMake install/export rules (including cg2o targets), or unset CG2O_DIR to build the paper's cg2o. A shell argument alone cannot create a valid cg2o package."
fi
mark_install_complete "$CG2O_INSTALL_DIR" "$CG2O_DIR" "$CG2O_HASH"
echo "cg2o installed successfully: $CG2O_INSTALL_DIR"
