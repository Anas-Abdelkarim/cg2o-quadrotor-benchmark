#!/usr/bin/env bash
# Remove only colcon outputs while preserving native builds and installed externals.
# Intended for build_workspace.sh --clean. Works without ROS so it can be tested.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ $# -eq 0 ]] || { echo "Usage: $0 (no arguments)" >&2; exit 2; }
[[ -f "$ROOT/VERSION" && -d "$ROOT/src" ]] || {
  echo "ERROR: not a PX4 NMPC repository: $ROOT" >&2; exit 1;
}
echo "Cleaning colcon artifacts in $ROOT (keeping externals/native builds)"
rm -rf -- "$ROOT/log"
shopt -s nullglob
if [[ -d "$ROOT/install" ]]; then
  for item in "$ROOT"/install/* "$ROOT"/install/.[!.]* "$ROOT"/install/..?*; do
    [[ "$(basename "$item")" == "externals" ]] && continue
    rm -rf -- "$item"
  done
fi
if [[ -d "$ROOT/build" ]]; then
  for item in "$ROOT"/build/* "$ROOT"/build/.[!.]* "$ROOT"/build/..?*; do
    case "$(basename "$item")" in
      externals|cg2o_backend|ipopt_cpp_backend) echo "Keeping $item" ;;
      *) rm -rf -- "$item" ;;
    esac
  done
fi
shopt -u nullglob
