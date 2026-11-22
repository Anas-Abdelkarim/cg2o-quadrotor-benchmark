#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
for arg in "$@"; do case "$arg" in
  --help|-h) echo 'Usage: ./scripts/build_externals/qgc.sh'; exit 0 ;;
  *) die "Unknown QGC option $arg" ;;
esac; done
banner 'Download external: QGroundControl'
if [[ "$WITH_QGC" != "ON" ]]; then
  echo "WITH_QGC=$WITH_QGC: skipping QGroundControl."
else
  mkdir -p "$QGC_INSTALL_DIR"

  if [[ -x "$QGC_APPIMAGE" ]]; then
    echo "QGroundControl already present: $QGC_APPIMAGE"
    echo "Skipping download."
  else
    if command -v curl >/dev/null 2>&1; then
      curl -L --fail --progress-bar "$QGC_URL" -o "$QGC_APPIMAGE" || die 'QGroundControl download failed.'
    elif command -v wget >/dev/null 2>&1; then
      wget --show-progress -O "$QGC_APPIMAGE" "$QGC_URL" || die 'QGroundControl download failed.'
    else
      echo "ERROR: curl or wget is required." >&2
      exit 1
    fi
    chmod +x "$QGC_APPIMAGE"
  fi
fi

