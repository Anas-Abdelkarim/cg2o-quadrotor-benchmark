#!/usr/bin/env bash
# Video of one PX4 SITL flight with the tuned settings: RViz and Gazebo side by side (slalom_sitl.mp4) and the
# screen capture (screen.mp4). Needs a visible X11 display.
#   benchmark/flight/record_video.sh RUN warm|cold [dN,dE YAW_DEG]
# Tuned settings: the newest studies in RUN/tuning that selected a setting, else those of data/tuning (override with WARM_TUNED/COLD_TUNED).
# Writes RUN/video_<mode>; an existing video folder is kept.
set -u
[[ $# -ge 2 ]] || { echo "usage: $0 RUN warm|cold [dN,dE YAW_DEG]" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
run=$1; mode=$2; start=${3:-0,0}; yaw=${4:-0}
case "$(realpath -m "$run")" in "$root/data"|"$root/data/"*) echo "refusing to write into data/" >&2; exit 2;; esac
newest() { for d in $(ls -td "$run"/tuning/*_challenging_cg2o_"$1" data/tuning/*_challenging_cg2o_"$1" 2>/dev/null); do [[ -f $d/selected_parameters.json ]] && { echo "$d"; return; }; done; }
WARM_TUNED=${WARM_TUNED:-$(newest warm)/selected_parameters.json}
COLD_TUNED=${COLD_TUNED:-$(newest cold)/selected_parameters.json}
R=$run/video_$mode
if [[ -f $R/slalom_sitl.mp4 ]]; then echo "keep existing $R"; exit 0; fi
# an incomplete earlier attempt is moved aside, not deleted
[[ -e $R ]] && mv "$R" "$R.incomplete_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run"
if [[ $mode == cold ]]; then extra=(--cold-every-solve --challenging-cg2o "$COLD_TUNED")
else extra=(--challenging-cg2o "$WARM_TUNED" --challenging-cg2o-cold "$COLD_TUNED"); fi
benchmark/flight/run_px4.sh challenging "$R" --challenging-vs-max 2.5 "${extra[@]}" \
  --challenging-start="$start" --challenging-yaw-deg="$yaw" > "$R.log" 2>&1
echo "video exit $?: $R/slalom_sitl.mp4"
