#!/usr/bin/env bash
# Legacy entrypoint for build_backends.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/build_backends.sh" "$@"
