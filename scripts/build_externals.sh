#!/usr/bin/env bash
# Backward-compatible dispatcher. Component logic lives in build_externals/*.sh.
# Run --help for examples. Default (no args) retains the original full install.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  bash "${BASH_SOURCE[0]}" "$@"
  return $?
fi
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELECT=''; CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --solver-only) SELECT='ipopt,hsl,g2o,cg2o' ;;
    --cg2o-only) SELECT='cg2o' ;;
    --full) SELECT='ipopt,hsl,g2o,cg2o,px4,xrce,qgc' ;;
    --clean) CLEAN=1 ;;
    --only=*) SELECT="${arg#--only=}" ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./scripts/build_externals.sh [--solver-only | --full | --only=ipopt,hsl,g2o,cg2o,px4,xrce,qgc] [--clean]

--solver-only = ipopt (IPOPT 3.14 from GitHub via coinbrew: MUMPS, SPRAL, PARDISO if MKL) + hsl (needs
                HSL_SOURCE=<coinhsl archive>, licensed separately) + g2o + cg2o.

Default (no options): legacy complete installation, honoring WITH_* variables.
Build against a local cg2o checkout (development):
  MARCH_NATIVE=ON CG2O_DIR=/path/to/my/cg2o ./scripts/build_externals.sh --solver-only

For rapid cg2o iteration, use ./scripts/rebuild_cg2o.sh instead.
USAGE
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
if [[ -z "$SELECT" ]]; then
  SELECT=''
  [[ "${WITH_IPOPT:-ON}" == ON ]] && SELECT='ipopt'
  [[ "${WITH_G2O_CG2O:-ON}" == ON ]] && SELECT="${SELECT:+$SELECT,}g2o,cg2o"
  if [[ "${WITH_PX4:-ON}" == ON ]]; then SELECT="${SELECT:+$SELECT,}px4"; fi
  if [[ "${WITH_XRCE_AGENT:-${WITH_PX4:-ON}}" == ON ]]; then SELECT="${SELECT:+$SELECT,}xrce"; fi
  if [[ "${WITH_QGC:-ON}" == ON ]]; then SELECT="${SELECT:+$SELECT,}qgc"; fi
fi
# Preserve old rebuild environment: FORCE_ONLY=cg2o / FORCE_EXTERNALS=1.
[[ "${FORCE_EXTERNALS:-0}" == 0 ]] || CLEAN=1
IFS=',' read -r -a names <<< "${SELECT// /,}"
for item in "${names[@]}"; do
  [[ -n "$item" ]] || continue
  case "$item" in ipopt|hsl|g2o|cg2o|px4|xrce|qgc) ;; *) echo "Invalid component: $item" >&2; exit 2 ;; esac
  if [[ "$item" == hsl ]]; then
    if [[ -z "${HSL_SOURCE:-}" && ! -f "$ROOT/install/externals/hsl/lib/x86_64-linux-gnu/libcoinhsl.so" ]]; then
      echo "NOTE: HSL_SOURCE not set: HSL (MA27/MA57) is not built, and the IPOPT runs of the benchmark need it." >&2
      continue
    fi
    if [[ "$CLEAN" == 1 ]]; then bash "$ROOT/scripts/build_externals/hsl.sh" --clean || exit 1
    else bash "$ROOT/scripts/build_externals/hsl.sh" || exit 1; fi
    continue
  fi
  arg=()
  if [[ "$item" == ipopt ]]; then
    if [[ "$CLEAN" == 1 ]]; then arg=(--source --clean); else arg=(--source); fi
    bash "$ROOT/scripts/build_externals/ipopt.sh" "${arg[@]}" || exit 1
    continue
  fi
  if [[ "$item" != qgc ]] && { [[ "$CLEAN" == 1 ]] || [[ ",${FORCE_ONLY:-}," == *",$item,"* ]]; }; then arg=(--clean); fi
  bash "$ROOT/scripts/build_externals/$item.sh" "${arg[@]}" || exit 1
done
printf '\nExternal components processed: %s\n' "$SELECT"
echo "Activate project-local dependency paths with: source scripts/setup_env.sh"
