#!/usr/bin/env bash
# Step 1 of a new run: the three PX4 SITL flights whose problems the tuning studies replay, flown with the
# untuned course settings, and their reference replays (both solvers, 250 iterations, IPOPT with MA57).
#   benchmark/flight/tuning_flights.sh RUN        (RUN = new run folder, e.g. results/run_01)
# Existing flights and replays in RUN are kept and skipped; nothing in data/ is written.
set -u
[[ $# -eq 1 ]] || { echo "usage: $0 RUN" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
run=$1
case "$(realpath -m "$run")" in "$root/data"|"$root/data/"*) echo "refusing to write into data/" >&2; exit 2;; esac
mkdir -p "$run/sitl" "$run/challenging_mpc"
REF=src/px4_nmpc_config/config/experiments/challenging_course/ipopt_reference.json
fly() {   # name, extra options
  local R=$run/sitl/$1; shift
  if [[ -f $R/summary.json ]]; then echo "keep existing $R"; return; fi
  [[ -e $R ]] && mv "$R" "$R.incomplete_$(date +%Y%m%d_%H%M%S)"   # incomplete attempt: moved aside, not deleted
  benchmark/flight/run_px4.sh challenging "$R" --challenging-vs-max 2.5 "$@" > "$R.log" 2>&1
  echo "$R exit $?"
}
fly challenging_lag_warm_v25_headless --headless --no-gazebo-gui
fly challenging_lag_warm_v25_video
fly challenging_lag_cold_v25_headless --cold-every-solve --headless --no-gazebo-gui
set +u; set --; source scripts/setup_env.sh >/dev/null 2>&1; set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for pair in warm_v25_headless:warm_v25 warm_v25_video:warm_video cold_v25_headless:cold_v25; do
  flight=$run/sitl/challenging_lag_${pair%%:*}; out=$run/challenging_mpc/replay_lag_${pair##*:}_ma57_b250
  if [[ -f $out/paired.json ]]; then echo "keep existing $out"; continue; fi
  [[ -e $out ]] && mv "$out" "$out.incomplete_$(date +%Y%m%d_%H%M%S)"
  taskset -c "${CPUS:-0-3}" venv/bin/python -m px4_nmpc_compare.challenging_replay "$flight" "$out" \
      --ipopt-params "$REF" --budget 250 > "$out.log" 2>&1
  echo "$out exit $?"
done
