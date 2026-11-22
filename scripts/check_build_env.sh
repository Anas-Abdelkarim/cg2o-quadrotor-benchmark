#!/usr/bin/env bash
# Sanity checks before building / running the PX4 NMPC workspace.
set -u
bad=0
warn() { echo "WARN: $*"; }
fail() { echo "FAIL: $*"; bad=1; }

printf 'Current directory: %s\n' "$PWD"
printf 'ROS distro:        %s\n' "${ROS_DISTRO:-<not sourced>}"
printf 'Virtualenv:        %s\n' "${VIRTUAL_ENV:-<none>}"
printf 'python3:           %s\n' "$(command -v python3 2>/dev/null || echo '<missing>')"
printf 'colcon:            %s\n' "$(command -v colcon 2>/dev/null || echo '<missing>')"

[[ "$PWD" == *"("* || "$PWD" == *")"* ]] && fail "current path contains parentheses; ROS 2 Humble CMake targets break there."
[[ "$PWD" == /mnt/* ]] && warn "path is on a WSL-mounted Windows drive; a copy in ~/px4_nmpc is faster and more reliable."
if [[ -n "${VIRTUAL_ENV:-}" ]] && ! grep -qi '^include-system-site-packages *= *true' "$VIRTUAL_ENV/pyvenv.cfg" 2>/dev/null; then
  fail "the active virtualenv hides the ROS Python packages; recreate it with 'python3 -m venv --system-site-packages <dir>' or 'deactivate'."
fi
[[ -f /opt/ros/humble/setup.bash ]] || fail "/opt/ros/humble/setup.bash is missing (install ros-humble-desktop)."
command -v colcon >/dev/null 2>&1 || fail "colcon missing (sudo apt install python3-colcon-common-extensions)."

em_version=$(python3 -c 'import em; print(getattr(em, "__version__", "?"))' 2>/dev/null || echo missing)
case "$em_version" in
  3.*) ;;
  missing) warn "python empy not importable; rosidl needs empy 3.x (sudo apt install python3-empy)." ;;
  *) fail "empy $em_version found; ROS 2 Humble needs empy 3.x (install empy==3.3.4; use --user only outside a virtualenv)." ;;
esac
python3 -c 'import casadi' 2>/dev/null || warn "casadi not installed for $(command -v python3) (python3 -m pip install -r requirements.txt; add --user only outside a virtualenv)."

snap_vars=$(env | grep -E '^[A-Za-z_]+=' | grep '/snap/' | cut -d= -f1 | grep -v '^PATH$' | tr '\n' ' ')
if [[ -n "$snap_vars" ]]; then
  warn "snap-injected variables in this terminal: $snap_vars"
  echo "      GUIs (rviz2, gz sim) started from here can fail with 'symbol lookup error ... /snap/core20/...libpthread'."
  echo "      nmpc.launch.py cleans them for RViz; otherwise use a normal terminal or: unset GTK_PATH GIO_MODULE_DIR LOCPATH"
fi
[[ $bad -eq 0 ]] && echo "Build environment: OK" || echo "Build environment: problems found"
exit "$bad"
