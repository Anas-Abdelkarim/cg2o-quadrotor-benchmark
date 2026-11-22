#!/usr/bin/env bash
# Build the NMPC cg2o extension; never download or rebuild external dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_ROOT="$(realpath -m "${EXTERNAL_INSTALL_ROOT:-$ROOT/install/externals}")"
BUILD_ROOT="$(realpath -m "${EXTERNAL_BUILD_ROOT:-$ROOT/build/externals}")"
BUILD_DIR="$(realpath -m "${PX4_NMPC_CG2O_BUILD_DIR:-$ROOT/build/cg2o_backend}")"
CPP="$ROOT/src/px4_nmpc_controller/px4_nmpc_controller/px4_nmpc_cg2o/cpp"
G2O_PREFIX="$(realpath -m "${PX4_NMPC_G2O_PREFIX:-$INSTALL_ROOT/g2o}")"
CG2O_PREFIX="$(realpath -m "${PX4_NMPC_CG2O_PREFIX:-$INSTALL_ROOT/cg2o}")"
# Prevent a deliberate or accidental override to system g2o/cg2o binaries.
for dep_path in "$G2O_PREFIX" "$CG2O_PREFIX"; do
  case "$dep_path" in /usr|/usr/*) echo "ERROR: Refusing system g2o/cg2o prefix: $dep_path" >&2; exit 1;; esac
done
G2O_CACHE="$BUILD_ROOT/g2o/CMakeCache.txt"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
OPTIMIZER="ISPD"
CLEAN=0; NO_TEST=0
fail() { echo "ERROR: $*" >&2; exit 1; }
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --no-test) NO_TEST=1 ;;
    -h|--help) printf '%s\n' 'Usage: ./scripts/build_backends/cg2o.sh [--clean] [--no-test]' '  Build only the native cg2o Python module; does not build the cg2o library.'; exit 0 ;;
    *) fail "Unknown option: $arg" ;;
  esac
done
[[ -z "${PX4_NMPC_CG2O_OPTIMIZER:-}" ]] || echo "NOTE: PX4_NMPC_CG2O_OPTIMIZER is ignored; the cg2o backend uses ISPD."
[[ -f "$CPP/CMakeLists.txt" ]] || fail "Missing backend source: $CPP"
[[ -f "$G2O_PREFIX/lib/cmake/g2o/g2oConfig.cmake" ]] || fail "Local g2o missing at $G2O_PREFIX. Run ./scripts/build_externals/g2o.sh"
[[ -f "$CG2O_PREFIX/lib/cmake/cg2o/cg2oConfig.cmake" ]] || fail "Local cg2o missing at $CG2O_PREFIX. Run ./scripts/build_externals/cg2o.sh"
[[ -f "$G2O_CACHE" ]] || fail "g2o CMakeCache missing at $G2O_CACHE; cannot ensure ABI-compatible Eigen flags."
grep -q '^G2O_NO_IMPLICIT_OWNERSHIP_OF_OBJECTS:BOOL=ON' "$G2O_CACHE" || fail 'Local g2o lacks required object ownership flag'
PYBIND11_DIR="$(python3 -m pybind11 --cmakedir 2>/dev/null || true)"
[[ -n "$PYBIND11_DIR" ]] || fail 'pybind11 missing for active python3. Install requirements.txt.'
G2O_NATIVE=OFF
if grep -q '^BUILD_WITH_MARCH_NATIVE:BOOL=ON' "$G2O_CACHE"; then G2O_NATIVE=ON; fi
# If a caller explicitly provides MARCH_NATIVE, it MUST match local g2o.
if [[ -n "${MARCH_NATIVE:-}" && "$MARCH_NATIVE" != "$G2O_NATIVE" ]]; then
  fail "MARCH_NATIVE=$MARCH_NATIVE differs from local g2o ($G2O_NATIVE); native Eigen ABI must match."
fi
FLAGS="${CXXFLAGS:-}"
if [[ "$G2O_NATIVE" == ON ]]; then FLAGS="$FLAGS -march=native"; fi
if [[ "$CLEAN" == 1 ]]; then echo "Cleaning only NMPC cg2o extension build: $BUILD_DIR"; rm -rf -- "$BUILD_DIR"; fi
cat <<MSG
NMPC cg2o backend: $OPTIMIZER
  g2o prefix   : $G2O_PREFIX
  cg2o prefix  : $CG2O_PREFIX
  Eigen native : $G2O_NATIVE
  backend build: $BUILD_DIR
MSG
# Force explicit prefix and config dirs. This prevents /usr/local from being used.
cmake -S "$CPP" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_BUILD_RPATH="$G2O_PREFIX/lib;$CG2O_PREFIX/lib" \
  -DPX4_NMPC_G2O_PREFIX="$G2O_PREFIX" \
  -DPX4_NMPC_CG2O_PREFIX="$CG2O_PREFIX" \
  -Dg2o_DIR="$G2O_PREFIX/lib/cmake/g2o" \
  -Dcg2o_DIR="$CG2O_PREFIX/lib/cmake/cg2o" \
  -DCMAKE_CXX_FLAGS="$FLAGS" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DCMAKE_PREFIX_PATH="$G2O_PREFIX;$CG2O_PREFIX;$PYBIND11_DIR"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
SO="$(find "$BUILD_DIR" -maxdepth 4 -name 'cpp_px4_nmpc_cg2o*.so' -type f -print -quit)"
[[ -n "$SO" ]] || fail "cg2o extension was not produced in $BUILD_DIR"
echo "Built native backend: $SO"
if [[ "$NO_TEST" == 0 ]]; then
  echo 'Running one-solve backend smoke test...'
  LD_LIBRARY_PATH="$G2O_PREFIX/lib:$CG2O_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    PYTHONPATH="$BUILD_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    PX4_NMPC_CG2O_BUILD_DIR="$BUILD_DIR" \
    python3 "$ROOT/scripts/smoke_test_backends.py" cg2o
fi
