#!/usr/bin/env bash
# Legacy entry point. For ROS and native libs, source scripts/setup_env.sh instead.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Source the environment: source scripts/source_env.sh' >&2
  exit 2
fi
_source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1090
source "$_source_root/scripts/setup_env.sh" --no-ros
unset _source_root
