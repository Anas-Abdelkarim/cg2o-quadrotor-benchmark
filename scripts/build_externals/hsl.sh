#!/usr/bin/env bash
# HSL linear solvers (MA27, MA57, ...) for IPOPT  ->  install/externals/hsl/lib/x86_64-linux-gnu/libcoinhsl.so
#
# IPOPT loads this library at run time (option hsllib); the benchmark uses MA57 (cold) and MA27 (warm).
# HSL is licensed separately and cannot be downloaded by a script: request "Coin-HSL" (free for academic
# use) at https://licences.stfc.ac.uk/product/coin-hsl and pass the source archive or folder:
#
#   HSL_SOURCE=/path/to/coinhsl-2024.05.15.tar.gz ./scripts/build_externals/hsl.sh [--clean]
#
# The paper used coinhsl 2024.05.15 built with meson (release), Intel MKL (libmkl_rt) as BLAS/LAPACK when
# present (otherwise the system BLAS/LAPACK) and METIS 5. Another HSL version builds as well, with a note.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

HSL_INSTALL_DIR="$INSTALL_ROOT/hsl"
HSL_BUILD_DIR="$BUILD_ROOT/hsl"
HSL_SOURCE="${HSL_SOURCE:-}"
WITH_MKL="${WITH_MKL:-ON}"
MESON_VERSION="${MESON_VERSION:-1.12.1}"
# SHA-256 over the files of coinhsl-2024.05.15 (README.md excluded), as in source_tree_sha256
HSL_TREE_SHA256="${HSL_TREE_SHA256:-e30a7afe6325d665fff43fe78a4d57f98658ddbb0da75a47f9919b7396d8ed67}"
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown hsl option: $arg" ;;
  esac
done

banner 'Build external: HSL (coinhsl) for IPOPT'
LIB="$HSL_INSTALL_DIR/lib/x86_64-linux-gnu/libcoinhsl.so"
if [[ "$CLEAN" == 0 && -f "$LIB" ]]; then
  echo "HSL already installed: $LIB (--clean rebuilds)"
  exit 0
fi
[[ -n "$HSL_SOURCE" ]] || die "HSL_SOURCE is not set. HSL needs a licence (free for academic use): https://licences.stfc.ac.uk/product/coin-hsl, then HSL_SOURCE=<coinhsl archive or folder> $0"
[[ -e "$HSL_SOURCE" ]] || die "HSL_SOURCE=$HSL_SOURCE does not exist"
command -v gfortran >/dev/null || die "gfortran is required (sudo apt install gfortran)"

rm -rf "$HSL_BUILD_DIR" "$HSL_INSTALL_DIR"
mkdir -p "$HSL_BUILD_DIR"
src="$HSL_BUILD_DIR/src"
if [[ -d "$HSL_SOURCE" ]]; then
  cp -r "$HSL_SOURCE" "$src"
else
  mkdir -p "$HSL_BUILD_DIR/unpack" && tar -xf "$HSL_SOURCE" -C "$HSL_BUILD_DIR/unpack" || die "Cannot unpack $HSL_SOURCE"
  top="$(dirname "$(find "$HSL_BUILD_DIR/unpack" -maxdepth 2 -name meson.build | head -n1)")"
  [[ -f "$top/meson.build" ]] || die "$HSL_SOURCE contains no meson.build (coinhsl 2023 or newer is needed)"
  mv "$top" "$src"
fi
sum="$(source_tree_sha256 "$src")"
if [[ "$sum" == "$HSL_TREE_SHA256" ]]; then echo "HSL source is coinhsl 2024.05.15, as used for the paper."
else echo "NOTE: this HSL source differs from the paper's coinhsl 2024.05.15 (tree SHA-256 $sum); IPOPT times may differ."; fi

# meson: the same version as for SPRAL (ipopt.sh), in a private virtual environment
venv="$BUILD_ROOT/meson-venv"
if [[ ! -x "$venv/bin/meson" ]]; then
  python3 -m venv "$venv" && "$venv/bin/pip" install -q "meson==$MESON_VERSION" ninja || die "Cannot install meson $MESON_VERSION"
fi
MESON="$venv/bin/meson"

MKL_ROOT="${MKLROOT:-/opt/intel/oneapi/mkl/latest}"
MKL_LIB="$MKL_ROOT/lib/intel64"; [[ -d "$MKL_LIB" ]] || MKL_LIB="$MKL_ROOT/lib"
BLAS_ARGS=()
if [[ "$WITH_MKL" == ON && -f "$MKL_LIB/libmkl_rt.so" ]]; then
  BLAS_ARGS=(-Dlibblas=mkl_rt -Dliblapack=mkl_rt "-Dlibblas_path=$MKL_LIB" "-Dliblapack_path=$MKL_LIB")
  echo "BLAS/LAPACK: Intel MKL (libmkl_rt) at $MKL_LIB, as for the paper"
else
  echo "BLAS/LAPACK: system (no Intel MKL found; the paper used MKL, so IPOPT times may differ slightly)"
fi

( cd "$src" && PATH="$venv/bin:$PATH" "$MESON" setup builddir --prefix="$HSL_INSTALL_DIR" --buildtype=release \
    -Dmodules=true -Dlibmetis_version=5 "${BLAS_ARGS[@]}" \
  && PATH="$venv/bin:$PATH" "$MESON" compile -C builddir -j "$JOBS" \
  && PATH="$venv/bin:$PATH" "$MESON" install -C builddir ) || die "HSL build failed (see above)"
[[ -f "$LIB" ]] || die "HSL build finished but $LIB is missing"
echo "HSL installed: $LIB"
