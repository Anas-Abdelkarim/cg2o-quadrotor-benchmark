#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
for arg in "$@"; do case "$arg" in
  --clean) FORCE_PX4=1 ;;
  --help|-h) echo 'Usage: ./scripts/build_externals/px4.sh [--clean]'; exit 0 ;;
  *) die "Unknown PX4 option $arg" ;;
esac; done
banner 'Build external: PX4 SITL'
stage_px4_install() {
  rm -rf "$PX4_INSTALL_DIR"
  mkdir -p "$PX4_INSTALL_DIR/bin"
  cp -a "$PX4_BUILD_DIR/bin/." "$PX4_INSTALL_DIR/bin/" || return 1
  if [[ -d "$PX4_BUILD_DIR/etc" ]]; then
    cp -a "$PX4_BUILD_DIR/etc" "$PX4_INSTALL_DIR/" || return 1
  fi
}

if [[ "$WITH_PX4" != "ON" ]]; then
  echo "WITH_PX4=$WITH_PX4: skipping PX4-Autopilot."
else
  if [[ -f "$PX4_DIR/CMakeLists.txt" ]]; then
    echo "Using existing PX4 source: $PX4_DIR"
  elif [[ -e "$PX4_DIR" ]]; then
    echo "ERROR: $PX4_DIR exists but does not look like PX4 source." >&2
    exit 1
  else
    git clone --recursive --branch "$PX4_REF" \
      https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR" \
      || die "could not clone PX4-Autopilot $PX4_REF."
  fi
  # the tag must still point to the commit used for the paper (PX4_COMMIT= skips the check for other refs)
  [[ -z "$PX4_COMMIT" ]] || require_commit "$PX4_DIR" "$PX4_COMMIT" PX4-Autopilot

  PX4_SETUP_MARKER="$PX4_DIR/.px4_nmpc_ubuntu_setup_done"
  if [[ "$PX4_SETUP" == "ON" ]]; then
    if [[ -f "$PX4_SETUP_MARKER" ]]; then
      echo "PX4 Ubuntu setup already completed. Skipping setup."
    else
      bash "$PX4_DIR/Tools/setup/ubuntu.sh" || die 'PX4 Ubuntu setup failed.'
      touch "$PX4_SETUP_MARKER"
    fi
  fi

  # PX4's ubuntu.sh installs ninja, so look for it again.  PX4 uses its own compiler flags
  # for SITL; the flight stack is not the bottleneck, so no -march=native here.
  detect_generator_args
  PX4_CMAKE_ARGS=(
    "${GEN_ARGS[@]}"
    -DCONFIG=px4_sitl_default
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  )
  PX4_HASH="$(config_hash "${PX4_CMAKE_ARGS[@]}")"

  PX4_BIN="$PX4_BUILD_DIR/bin/px4"
  PX4_INSTALLED_BIN="$PX4_INSTALL_DIR/bin/px4"

  if [[ "$PX4_BUILD" == "ON" ]]; then
    if [[ "$FORCE_PX4" != "1" && -x "$PX4_INSTALLED_BIN" ]] \
       && install_marker_matches "$PX4_INSTALL_DIR" "$PX4_DIR" "$PX4_HASH"; then
      echo "PX4 SITL already installed: $PX4_INSTALLED_BIN"
      echo "Skipping PX4 build."

    elif [[ "$FORCE_PX4" != "1" && -x "$PX4_BIN" ]] \
         && build_marker_matches "$PX4_BUILD_DIR" "$PX4_DIR" "$PX4_HASH"; then
      echo "PX4 SITL already built: $PX4_BIN"
      echo "Install is missing; staging without rebuilding."
      stage_px4_install || die "PX4 install staging failed."
      mark_install_complete "$PX4_INSTALL_DIR" "$PX4_DIR" "$PX4_HASH"

    else
      if [[ "$FORCE_PX4" == "1" ]]; then
        echo "Forced rebuild: cleaning PX4 build/install."
        rm -rf "$PX4_BUILD_DIR" "$PX4_INSTALL_DIR"
      fi

      prepare_build_dir "$PX4_BUILD_DIR" "$PX4_HASH"

      # PX4's top-level Makefile is a wrapper around CMake.  Configuring directly
      # lets us keep its build tree outside the source directory.
      cmake -S "$PX4_DIR" -B "$PX4_BUILD_DIR" "${PX4_CMAKE_ARGS[@]}" \
        || die "PX4 configuration failed."
      cmake --build "$PX4_BUILD_DIR" --parallel "$JOBS" \
        || die "PX4 compilation failed."
      mark_build_complete "$PX4_BUILD_DIR" "$PX4_DIR" "$PX4_HASH"

      stage_px4_install || die "PX4 install staging failed."
      mark_install_complete "$PX4_INSTALL_DIR" "$PX4_DIR" "$PX4_HASH"
    fi
  fi
fi

