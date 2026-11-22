#!/usr/bin/env bash
# Legacy entrypoint; actual implementation in scripts/build_backends/ipopt_cpp.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/build_backends/ipopt_cpp.sh" "$@"
