#!/usr/bin/env bash
# Step 3 of a new run: the repeated PX4 SITL flights of cg2o with frozen tuned settings and randomized take-off
# points and headings (challenging_course/starts.json), each followed by its ground-truth evaluation.
# Warm flights: warm-tuned settings for warm solves, cold-tuned values for their cold solves (start of
# navigation). Cold flights: cold-tuned settings, cold start at every problem.
#   benchmark/flight/campaign.sh RUN warm|cold [FIRST LAST]
# Tuned settings: the newest studies in RUN/tuning that selected a setting (override with WARM_TUNED=... COLD_TUNED=...).
# Existing flight folders are kept and skipped; nothing in data/ is written.
set -u
[[ $# -ge 2 ]] || { echo "usage: $0 RUN warm|cold [FIRST LAST]" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
run=$1; mode=$2; first=${3:-1}; last=${4:-10}
case "$(realpath -m "$run")" in "$root/data"|"$root/data/"*) echo "refusing to write into data/" >&2; exit 2;; esac
newest() { for d in $(ls -td "$run"/tuning/*_challenging_cg2o_"$1" 2>/dev/null); do [[ -f $d/selected_parameters.json ]] && { echo "$d"; return; }; done; }
WARM_TUNED=${WARM_TUNED:-$(newest warm)/selected_parameters.json}
COLD_TUNED=${COLD_TUNED:-$(newest cold)/selected_parameters.json}
for f in "$COLD_TUNED" $([[ $mode == warm ]] && echo "$WARM_TUNED"); do
  [[ -f $f ]] || { echo "missing tuned settings $f (run the tuning first, or set WARM_TUNED/COLD_TUNED)" >&2; exit 2; }
done
STARTS=src/px4_nmpc_config/config/experiments/challenging_course/starts.json
out=$run/challenging_campaign
mkdir -p "$out"
if [[ $mode == cold ]]; then extra=(--cold-every-solve --challenging-cg2o "$COLD_TUNED")
else extra=(--challenging-cg2o "$WARM_TUNED" --challenging-cg2o-cold "$COLD_TUNED"); fi
for i in $(seq "$first" "$last"); do
  R=$out/${mode}_$(printf %02d "$i")
  if [[ -f $R/challenging_evaluation.json ]]; then echo "keep existing $R"; continue; fi
  [[ -e $R ]] && mv "$R" "$R.incomplete_$(date +%Y%m%d_%H%M%S)"   # incomplete attempt: moved aside, not deleted
  read -r dn de yaw < <(python3 -c "import json;s=json.load(open('$STARTS'))['starts'][$i-1];print(s['dN'],s['dE'],s['yaw_deg'])")
  benchmark/flight/run_px4.sh challenging "$R" --challenging-vs-max 2.5 "${extra[@]}" \
    --challenging-start="$dn,$de" --challenging-yaw-deg="$yaw" \
    --headless --no-gazebo-gui > "$R.log" 2>&1
  echo "$mode $i exit $?"
  ( set +u; set --; source scripts/setup_env.sh >/dev/null 2>&1; venv/bin/python benchmark/flight/evaluate_flight.py "$R" > "$R/evaluate.log" 2>&1 )
  python3 -c "import json;e=json.load(open('$R/challenging_evaluation.json'));print({k:e.get(k) for k in ('outcome','time_to_goal_s','min_true_body_clearance_m','max_true_tilt_deg')})" 2>&1
done
