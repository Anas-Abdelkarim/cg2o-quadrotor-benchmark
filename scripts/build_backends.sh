#!/usr/bin/env bash
# Build both native backends, or just one with --only=.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHICH='cg2o,ipopt_cpp'; ARGS=()
for arg in "$@"; do
 case "$arg" in
  --only=*) WHICH="${arg#--only=}" ;;
  --clean|--no-test) ARGS+=("$arg") ;;
  -h|--help) echo 'Usage: ./scripts/build_backends.sh [--only=cg2o|ipopt_cpp] [--clean] [--no-test]'; exit 0 ;;
  *) echo "Unknown option: $arg" >&2; exit 2 ;;
 esac
done
IFS=, read -r -a components <<< "$WHICH"
for item in "${components[@]}"; do
 case "$item" in cg2o|ipopt_cpp) ;; *) echo "Unknown backend: $item" >&2; exit 2 ;; esac
 bash "$ROOT/scripts/build_backends/$item.sh" "${ARGS[@]}"
done
