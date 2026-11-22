#!/usr/bin/env bash
# IPOPT for the native ipopt_cpp backend  ->  install/externals/ipopt  (pkg-config: ipopt)
#
#   ./scripts/build_externals/ipopt.sh                    default: --source
#   ./scripts/build_externals/ipopt.sh --source           IPOPT 3.14 from source (coinbrew) with every linear
#                                                         solver that can be built here:
#                                                           mumps       always (ThirdParty-Mumps)
#                                                           pardisomkl  if Intel MKL is found (MKLROOT or /opt/intel/oneapi/mkl/latest)
#                                                           spral       built from GitHub (needs meson; a venv is made for it)
#                                                           ma27 ... ma97  not compiled in: IPOPT loads them at run time from
#                                                                       build_externals/hsl.sh (option hsllib), as for the paper;
#                                                                       IPOPT_HSL_SOURCE=<coinhsl> compiles them in instead
#                                                                       (free academic licence: https://licences.stfc.ac.uk/product/coin-hsl)
#   Sources are fetched into externals/ipopt, built in externals/ipopt/build, installed in install/externals/ipopt.
#   ./scripts/build_externals/ipopt.sh --system           do nothing but check the system package
#                                                         (sudo apt install coinor-libipopt-dev, IPOPT 3.11.9, MUMPS only)
#   --clean                                               remove install/externals/ipopt and externals/ipopt/build first (sources kept)
#
# Environment: IPOPT_COMMIT and the ThirdParty commits below (the paper's versions), WITH_MKL=ON|OFF,
#              WITH_SPRAL=ON|OFF, IPOPT_HSL_SOURCE=<path>, MKLROOT=<path>
# MKL is used through libmkl_rt; run with MKL_THREADING_LAYER=SEQUENTIAL MKL_NUM_THREADS=1 so pardisomkl is timed on one core
# like the other solvers (the ipopt-linear-solvers preset sets both).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

IPOPT_INSTALL_DIR="$INSTALL_ROOT/ipopt"
IPOPT_BUILD_DIR="$(realpath -m "${IPOPT_DIR:-$EXTERNALS/ipopt}")"   # sources (coinbrew fetch, spral) and the coinbrew build/ tree live here
# Versions used for the paper (IPOPT 3.14.21 = stable/3.14 at the commit below). coinbrew fetches the
# branches named in Ipopt's .coin-or/Dependencies; every checkout is then moved to the pinned commit.
IPOPT_REF="${IPOPT_REF:-stable/3.14}"
IPOPT_COMMIT="${IPOPT_COMMIT:-cc3307136498793f38d51f7a02f70652d724083a}"
ASL_COMMIT="${ASL_COMMIT:-6f3e1bb5593b8d900d6232aafd58bcd7a20c2b7a}"      # ThirdParty-ASL releases/2.0.2-1
HSLTP_COMMIT="${HSLTP_COMMIT:-ec509dcf102525216985d4cf8b61996028bc155d}"  # ThirdParty-HSL releases/2.2.6
MUMPS_COMMIT="${MUMPS_COMMIT:-9b96a7e2c3b46457a57d73123c5d89a32e0e85fc}"  # ThirdParty-Mumps releases/3.0.14
SPRAL_REF="${SPRAL_REF:-v2024.05.08}"
SPRAL_COMMIT="${SPRAL_COMMIT:-1a71375f9b3bf97245d80d169aff7e4a7db8ede9}"
COINBREW_URL="${COINBREW_URL:-https://raw.githubusercontent.com/coin-or/coinbrew/89cdf805dbbf3f3e296a3cec53a607b824f9be1c/coinbrew}"
WITH_MKL="${WITH_MKL:-ON}"
WITH_SPRAL="${WITH_SPRAL:-ON}"
# HSL compiled into IPOPT changes its results (the paper loads HSL at run time); only on explicit request.
HSL_SOURCE="${IPOPT_HSL_SOURCE:-}"
MODE=source; CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --source) MODE=source ;;
    --system) MODE=system ;;
    --clean) CLEAN=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown ipopt option: $arg" ;;
  esac
done
[[ "$CLEAN" == 0 ]] || { echo "Removing $IPOPT_INSTALL_DIR and $IPOPT_BUILD_DIR/build (sources are kept)"; rm -rf -- "$IPOPT_INSTALL_DIR" "$IPOPT_BUILD_DIR/build" "$IPOPT_BUILD_DIR/spral/builddir"; }

build_spral() { # -> $IPOPT_INSTALL_DIR/lib/libspral.*
  banner "SPRAL ($SPRAL_REF) for IPOPT"
  local src="$IPOPT_BUILD_DIR/spral" venv="$IPOPT_BUILD_DIR/meson-venv"
  if ! command -v meson >/dev/null; then
    [[ -x "$venv/bin/meson" ]] || { python3 -m venv "$venv" && "$venv/bin/pip" install -q "meson==${MESON_VERSION:-1.12.1}"; } || return 1
    export PATH="$venv/bin:$PATH"
  fi
  [[ -d "$src/.git" ]] || git clone -q --depth 1 --branch "$SPRAL_REF" https://github.com/ralna/spral.git "$src" || return 1
  require_commit "$src" "$SPRAL_COMMIT" SPRAL
  rm -rf "$src/builddir"
  # no GPU; METIS 5 from the system (libmetis-dev); OpenMP is required by SSIDS
  ( cd "$src" && meson setup builddir --prefix="$IPOPT_INSTALL_DIR" --libdir=lib --buildtype=release \
      -Dgpu=false -Dexamples=false -Dtests=false -Dlibmetis_version=5 \
      -Dlibblas=$BLAS_NAME -Dliblapack=$LAPACK_NAME ${BLAS_DIR:+-Dlibblas_path=$BLAS_DIR -Dliblapack_path=$BLAS_DIR} \
    && meson compile -C builddir -j "$JOBS" && meson install -C builddir ) || return 1
  [[ -n "$(ls "$IPOPT_INSTALL_DIR"/lib/libspral.* 2>/dev/null)" ]]
}

case "$MODE" in
system)
  banner "IPOPT: system package"
  if pkg-config --exists ipopt 2>/dev/null; then
    echo "Found system IPOPT $(pkg-config --modversion ipopt) - nothing to install (linear solver: MUMPS only)."
  else
    echo "System IPOPT not found. On Ubuntu/Debian run:"
    echo "    sudo apt install coinor-libipopt-dev libmumps-seq-dev"
    echo "or build it with  ./scripts/build_externals/ipopt.sh --source"
    exit 1
  fi ;;
source)
  if [[ "$CLEAN" == 0 && "${FORCE_IPOPT:-0}" != 1 && -f "$IPOPT_INSTALL_DIR/lib/pkgconfig/ipopt.pc" ]]; then
    echo "IPOPT $(PKG_CONFIG_PATH="$IPOPT_INSTALL_DIR/lib/pkgconfig" pkg-config --modversion ipopt) already installed in $IPOPT_INSTALL_DIR (--clean or FORCE_IPOPT=1 rebuilds)"
    exit 0
  fi
  banner "IPOPT: build from source with coinbrew ($IPOPT_REF)"
  command -v git >/dev/null || die "git is required"
  command -v gfortran >/dev/null || die "gfortran is required (sudo apt install gfortran liblapack-dev libblas-dev pkg-config libmetis-dev)"
  mkdir -p "$IPOPT_BUILD_DIR" "$IPOPT_INSTALL_DIR/lib"

  # ---- BLAS / LAPACK: MKL (sequential) when present, it also provides PARDISO --------------------------
  MKL_ROOT="${MKLROOT:-/opt/intel/oneapi/mkl/latest}"
  MKL_LIB="$MKL_ROOT/lib/intel64"; [[ -d "$MKL_LIB" ]] || MKL_LIB="$MKL_ROOT/lib"
  if [[ "$WITH_MKL" == ON && -f "$MKL_LIB/libmkl_core.so" ]]; then
    # One dynamic library (libmkl_rt) and nothing else: g2o's CHOLMOD and the system libblas.so.3 already load MKL, and
    # linking MKL's static-layer libraries (gf / intel / sequential / intel_thread) a second time in the same process
    # made PARDISO fail and SPRAL hang. The threading layer is chosen at run time (MKL_THREADING_LAYER=SEQUENTIAL).
    LAPACK_LFLAGS="-L$MKL_LIB -Wl,-rpath,$MKL_LIB -lmkl_rt -lpthread -lm -ldl"
    BLAS_NAME=mkl_rt; LAPACK_NAME=mkl_rt; BLAS_DIR="$MKL_LIB"
    echo "BLAS/LAPACK: Intel MKL (libmkl_rt) at $MKL_LIB  -> linear_solver pardisomkl available"
  else
    LAPACK_LFLAGS="-llapack -lblas"
    BLAS_NAME=blas; LAPACK_NAME=lapack; BLAS_DIR=""
    echo "BLAS/LAPACK: system (-llapack -lblas); no MKL -> no pardisomkl"
  fi

  # ---- SPRAL -------------------------------------------------------------------------------------------
  SPRAL_ARGS=()
  if [[ "$WITH_SPRAL" == ON ]]; then
    if build_spral; then
      SPRAL_ARGS=(--with-spral-cflags="-I$IPOPT_INSTALL_DIR/include"
                  --with-spral-lflags="-L$IPOPT_INSTALL_DIR/lib -Wl,-rpath,$IPOPT_INSTALL_DIR/lib -lspral -lmetis -lhwloc -lgfortran -lgomp -lstdc++ $LAPACK_LFLAGS")
    else
      echo "WARNING: SPRAL build failed; IPOPT is built without it (set WITH_SPRAL=OFF to skip)." >&2
    fi
  fi

  # ---- coinbrew ----------------------------------------------------------------------------------------
  cd "$IPOPT_BUILD_DIR" || die "cannot enter $IPOPT_BUILD_DIR"
  if [[ ! -f coinbrew ]]; then
    curl -fsSL -o coinbrew "$COINBREW_URL" \
      || die "Cannot download coinbrew (GitHub unreachable?)"
    chmod +x coinbrew
  fi
  ./coinbrew fetch "Ipopt@$IPOPT_REF" --no-prompt --skip-update || die "coinbrew fetch failed"
  for _pin in "Ipopt $IPOPT_COMMIT" "ThirdParty/ASL $ASL_COMMIT" "ThirdParty/HSL $HSLTP_COMMIT" "ThirdParty/Mumps $MUMPS_COMMIT"; do
    set -- $_pin
    git -C "$1" checkout --quiet "$2" 2>/dev/null \
      || { git -C "$1" fetch --quiet origin "$2" && git -C "$1" checkout --quiet "$2"; } \
      || die "Cannot check out $1 at $2."
    require_commit "$IPOPT_BUILD_DIR/$1" "$2" "$1"
  done
  unset _pin; set --

  # ---- HSL (optional, licensed source) -----------------------------------------------------------------
  if [[ -n "$HSL_SOURCE" ]]; then
    hsl_dir="$IPOPT_BUILD_DIR/ThirdParty/HSL/coinhsl"
    rm -rf "$hsl_dir"
    if [[ -d "$HSL_SOURCE" ]]; then cp -r "$HSL_SOURCE" "$hsl_dir"
    elif [[ -f "$HSL_SOURCE" ]]; then
      mkdir -p "$hsl_dir.tmp" && tar -xf "$HSL_SOURCE" -C "$hsl_dir.tmp" \
        && mv "$(find "$hsl_dir.tmp" -mindepth 1 -maxdepth 1 -type d | head -n1)" "$hsl_dir" && rm -rf "$hsl_dir.tmp" \
        || die 'HSL source extraction failed.'
    else die "IPOPT_HSL_SOURCE=$HSL_SOURCE does not exist"; fi
    echo "HSL source placed in $hsl_dir -> ma27/ma57/ma77/ma86/ma97 (whatever the package contains)"
  else
    rm -rf "$IPOPT_BUILD_DIR/ThirdParty/HSL/coinhsl"
    echo "HSL is not compiled into IPOPT; IPOPT loads it at run time (option hsllib) from build_externals/hsl.sh, as for the paper."
  fi

  ./coinbrew build Ipopt --prefix "$IPOPT_INSTALL_DIR" --tests none --no-prompt --parallel-jobs "$JOBS" \
    --verbosity 2 --with-lapack-lflags="$LAPACK_LFLAGS" --with-metis-lflags="-lmetis" --with-metis-cflags="-I/usr/include" \
    "${SPRAL_ARGS[@]}" ${IPOPT_CONFIGURE_FLAGS:-} || die "coinbrew build failed (see output above)"
  [[ -f "$IPOPT_INSTALL_DIR/include/coin-or/IpIpoptApplication.hpp" ]] || die "IPOPT headers missing after build"

  # report which linear solvers made it into the library
  cfg="$IPOPT_INSTALL_DIR/include/coin-or/IpoptConfig.h"
  echo; echo "Installed IPOPT $(PKG_CONFIG_PATH="$IPOPT_INSTALL_DIR/lib/pkgconfig" pkg-config --modversion ipopt) into $IPOPT_INSTALL_DIR"
  echo "Linear solvers compiled in:"
  grep -E '^#define IPOPT_HAS_(MUMPS|PARDISO_MKL|PARDISO|SPRAL|HSL|WSMP)' "$cfg" 2>/dev/null | sed 's/#define IPOPT_HAS_/  /' \
    || echo "  (could not read $cfg)" ;;
esac
echo "Activate it with:  source scripts/setup_env.sh   (then ./scripts/build_backends/ipopt_cpp.sh --clean)"
