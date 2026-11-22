#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
for arg in "$@"; do case "$arg" in
  --clean) FORCE_XRCE=1 ;;
  --help|-h) echo 'Usage: ./scripts/build_externals/xrce.sh [--clean]'; exit 0 ;;
  *) die "Unknown XRCE option $arg" ;;
esac; done
banner 'Build external: Micro XRCE-DDS Agent'
XRCE_OK=0

# Logger and P2P profiles off: no per-message logging work in the bridge.
XRCE_CMAKE_ARGS=(
  "${GEN_ARGS[@]}"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_INSTALL_PREFIX="$XRCE_INSTALL_DIR"
  -DCMAKE_PREFIX_PATH="$ROS_PREFIX"
  '-DCMAKE_INSTALL_RPATH=$ORIGIN/../lib'
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  -DUAGENT_SUPERBUILD=OFF
  -DUAGENT_USE_SYSTEM_FASTDDS=ON
  -DUAGENT_USE_SYSTEM_FASTCDR=ON
  -DUAGENT_P2P_PROFILE=OFF
  -DUAGENT_LOGGER_PROFILE=OFF
  -Dfastcdr_DIR="$FASTCDR_CMAKE_DIR"
  -Dfastrtps_DIR="$FASTRTPS_CMAKE_DIR"
)
XRCE_HASH="$(config_hash "ref=$XRCE_REF" "${XRCE_CMAKE_ARGS[@]}")"

if [[ "$WITH_XRCE_AGENT" != "ON" ]]; then
  echo "WITH_XRCE_AGENT=$WITH_XRCE_AGENT: skipping Micro XRCE-DDS Agent."
  XRCE_OK=1

elif [[ "$FORCE_XRCE" != "1" \
     && -x "$XRCE_INSTALL_DIR/bin/MicroXRCEAgent" \
     && -f "$XRCE_INSTALL_DIR/.px4_nmpc_xrce_ref" \
     && "$(cat "$XRCE_INSTALL_DIR/.px4_nmpc_xrce_ref")" == "$XRCE_REF" ]] \
     && install_marker_matches "$XRCE_INSTALL_DIR" "$XRCE_DIR" "$XRCE_HASH"; then
  echo "Micro XRCE-DDS Agent $XRCE_REF already installed: $XRCE_INSTALL_DIR/bin/MicroXRCEAgent"
  echo "Skipping build."
  XRCE_OK=1

else
  echo "Using ROS 2 $ROS_DISTRO_NAME middleware libraries:"
  echo "  ROS prefix : $ROS_PREFIX"
  echo "  Fast-CDR   : $FASTCDR_CMAKE_DIR"
  echo "  Fast-DDS   : $FASTRTPS_CMAKE_DIR"

  if [[ ! -f "$FASTCDR_CMAKE_DIR/fastcdr-config.cmake" ]]; then
    echo "ERROR: Fast-CDR CMake package was not found: $FASTCDR_CMAKE_DIR" >&2
    echo "Install it with: sudo apt install ros-${ROS_DISTRO_NAME}-fastcdr" >&2

  elif [[ ! -f "$FASTRTPS_CMAKE_DIR/fastrtps-config.cmake" ]]; then
    echo "ERROR: Fast-DDS/Fast-RTPS CMake package was not found: $FASTRTPS_CMAKE_DIR" >&2
    echo "Install it with: sudo apt install ros-${ROS_DISTRO_NAME}-fastrtps" >&2

  elif ! clone_and_pin https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$XRCE_DIR" "$XRCE_REF"; then
    echo "ERROR: could not get the Micro XRCE-DDS Agent source ($XRCE_REF)." >&2

  else
    [[ -z "$XRCE_COMMIT" ]] || require_commit "$XRCE_DIR" "$XRCE_COMMIT" Micro-XRCE-DDS-Agent
    if [[ "$FORCE_XRCE" == "1" ]]; then
      echo "Forced rebuild: cleaning XRCE build/install."
      rm -rf "$XRCE_BUILD_DIR" "$XRCE_INSTALL_DIR"
    fi

    if [[ "$FORCE_XRCE" != "1" ]] \
       && build_marker_matches "$XRCE_BUILD_DIR" "$XRCE_DIR" "$XRCE_HASH"; then
      echo "Micro XRCE-DDS Agent is already built at: $XRCE_BUILD_DIR"
      echo "Install is missing/stale; installing without rebuilding."

      if cmake_reinstall "$XRCE_BUILD_DIR" "$XRCE_INSTALL_DIR"; then
        printf '%s\n' "$XRCE_REF" > "$XRCE_INSTALL_DIR/.px4_nmpc_xrce_ref"
        mark_install_complete "$XRCE_INSTALL_DIR" "$XRCE_DIR" "$XRCE_HASH"
        XRCE_OK=1
      else
        echo "ERROR: Micro XRCE-DDS Agent installation failed." >&2
      fi

    else
      prepare_build_dir "$XRCE_BUILD_DIR" "$XRCE_HASH"

      echo
      echo "Configuring Agent $XRCE_REF against ROS $ROS_DISTRO_NAME Fast-DDS/Fast-CDR..."

      if ! cmake -S "$XRCE_DIR" -B "$XRCE_BUILD_DIR" "${XRCE_CMAKE_ARGS[@]}"; then
        echo "ERROR: Micro XRCE-DDS Agent configuration failed." >&2
    
      elif ! cmake --build "$XRCE_BUILD_DIR" --parallel "$JOBS"; then
        echo "ERROR: Micro XRCE-DDS Agent compilation failed." >&2
    
      else
        mark_build_complete "$XRCE_BUILD_DIR" "$XRCE_DIR" "$XRCE_HASH"

        if ! cmake_reinstall "$XRCE_BUILD_DIR" "$XRCE_INSTALL_DIR"; then
          echo "ERROR: Micro XRCE-DDS Agent installation failed." >&2
              elif [[ ! -x "$XRCE_INSTALL_DIR/bin/MicroXRCEAgent" ]]; then
          echo "ERROR: Agent build finished but executable was not found:" >&2
          echo "  $XRCE_INSTALL_DIR/bin/MicroXRCEAgent" >&2
        else
          printf '%s\n' "$XRCE_REF" > "$XRCE_INSTALL_DIR/.px4_nmpc_xrce_ref"
          mark_install_complete "$XRCE_INSTALL_DIR" "$XRCE_DIR" "$XRCE_HASH"
          XRCE_OK=1
          echo
          echo "Micro XRCE-DDS Agent successfully installed:"
          echo "  $XRCE_INSTALL_DIR/bin/MicroXRCEAgent"
        fi
      fi
    fi
  fi
fi

[[ "$XRCE_OK" == 1 ]] || die 'XRCE Agent was not successfully installed (see above)'
