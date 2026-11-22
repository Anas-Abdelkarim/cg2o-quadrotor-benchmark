#!/usr/bin/env bash
# FAST developer loop: edit cg2o -> rebuild local cg2o library -> reinstall ->
# rebuild native NMPC extension -> smoke test. Never rebuilds g2o, PX4, XRCE or QGC.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLEAN=0; NO_TEST=0; MODE=both
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --no-test) NO_TEST=1 ;;
    --library-only) MODE=library ;;
    --backend-only) MODE=backend ;;
    -h|--help)
      cat <<'HELP'
Usage: ./scripts/rebuild_cg2o.sh [--clean] [--no-test] [--library-only | --backend-only]

Default: incremental local cg2o library build + install + NMPC pybind extension + smoke test.
--clean        clean only cg2o library/extension build trees, NOT g2o/PX4/XRCE.
--no-test      skip one-solve smoke test.
--library-only rebuild/reinstall cg2o, not the NMPC binding.
--backend-only rebuild binding only (requires previously installed local cg2o).

For custom research source:
  CG2O_DIR=/absolute/path/to/cg2o ./scripts/rebuild_cg2o.sh
HELP
      exit 0 ;;
    *) echo "Unknown option $arg" >&2; exit 2 ;;
  esac
done
OPTS=(); [[ "$CLEAN" == 0 ]] || OPTS+=(--clean)
if [[ "$MODE" != backend ]]; then
  echo '== 1/2: compile/install LOCAL cg2o library =='
  bash "$ROOT/scripts/build_externals/cg2o.sh" "${OPTS[@]}"
fi
if [[ "$MODE" != library ]]; then
  echo '== 2/2: compile LOCAL NMPC cg2o backend =='
  [[ "$NO_TEST" == 0 ]] || OPTS+=(--no-test)
  bash "$ROOT/scripts/build_backends/cg2o.sh" "${OPTS[@]}"
fi
printf '\nDone. Re-source paths after switching installations:\n  source scripts/setup_env.sh\n'
