#!/usr/bin/env bash
# Native C++ IPOPT extension, independent of cg2o/g2o.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CPP="$ROOT/src/px4_nmpc_controller/px4_nmpc_controller/px4_nmpc_ipopt_cpp/cpp"
BUILD_DIR="${PX4_NMPC_IPOPT_CPP_BUILD_DIR:-$ROOT/build/ipopt_cpp_backend}"
CLEAN=0; NO_TEST=0
for arg in "$@"; do
 case "$arg" in
   --clean) CLEAN=1 ;;
   --no-test) NO_TEST=1 ;;
   -h|--help) echo 'Usage: ./scripts/build_backends/ipopt_cpp.sh [--clean] [--no-test]'; exit 0 ;;
   *) echo "Unknown option: $arg" >&2; exit 2 ;;
 esac
done
# Prefer the project-local IPOPT built from GitHub (./scripts/build_externals/ipopt.sh, IPOPT 3.14 with
# MUMPS/SPRAL/PARDISO); fall back to the system package (Ubuntu 22.04: IPOPT 3.11, MUMPS only).
IPOPT_PREFIX="$(realpath -m "${PX4_NMPC_IPOPT_PREFIX:-${EXTERNAL_INSTALL_ROOT:-$ROOT/install/externals}/ipopt}")"
if [[ -f "$IPOPT_PREFIX/lib/pkgconfig/ipopt.pc" ]]; then
  export PKG_CONFIG_PATH="$IPOPT_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  IPOPT_RPATH="$IPOPT_PREFIX/lib"
else
  IPOPT_RPATH=""
fi
pkg-config --exists ipopt || { echo 'No IPOPT: run ./scripts/build_externals/ipopt.sh (or sudo apt install coinor-libipopt-dev).' >&2; exit 1; }
echo "IPOPT $(pkg-config --modversion ipopt) from $(pkg-config --variable=prefix ipopt)"
PYBIND11_DIR="$(python3 -m pybind11 --cmakedir 2>/dev/null || true)"
[[ -n "$PYBIND11_DIR" ]] || { echo 'pybind11 missing for active python3' >&2; exit 1; }
[[ "$CLEAN" == 0 ]] || rm -rf -- "$BUILD_DIR"
# a changed IPOPT (system <-> local) needs a fresh configure: pkg-config results are cached
PREV="$(cat "$BUILD_DIR/.px4_nmpc_ipopt" 2>/dev/null || true)"
CUR="$(pkg-config --variable=prefix ipopt) $(pkg-config --modversion ipopt)"
if [[ -n "$PREV" && "$PREV" != "$CUR" ]]; then echo "IPOPT changed ($PREV -> $CUR): clean configure"; rm -rf -- "$BUILD_DIR"; fi
mkdir -p "$BUILD_DIR"; echo "$CUR" > "$BUILD_DIR/.px4_nmpc_ipopt"
cmake -S "$CPP" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dpybind11_DIR="$PYBIND11_DIR" \
 -DCMAKE_PREFIX_PATH="$PYBIND11_DIR" -DPX4_NMPC_IPOPT_RPATH="$IPOPT_RPATH"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
SO="$(find "$BUILD_DIR" -maxdepth 4 -name 'cpp_px4_nmpc_ipopt_cpp*.so' -type f -print -quit)"
[[ -n "$SO" ]] || { echo 'Native IPOPT extension not produced' >&2; exit 1; }
echo "Built native IPOPT: $SO"
if [[ -n "$IPOPT_RPATH" ]]; then
  # private sonames: the local IPOPT and CasADi's bundled IPOPT share libipopt.so.3 / libcoinmumps.so.3
  python3 "$ROOT/scripts/build_backends/isolate_ipopt.py" --ipopt-lib "$IPOPT_RPATH" --ext "$SO" \
    --out "$BUILD_DIR/ipopt_private" || { echo 'IPOPT isolation failed (patchelf: python3 -m pip install patchelf)' >&2; exit 1; }
fi
if [[ "$NO_TEST" == 0 ]]; then
  PX4_NMPC_IPOPT_CPP_BUILD_DIR="$BUILD_DIR" python3 "$ROOT/scripts/smoke_test_backends.py" ipopt_cpp
fi
