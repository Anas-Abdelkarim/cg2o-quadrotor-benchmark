#!/usr/bin/env bash
# Step 4: paired open-loop replay of every campaign flight. cg2o uses the flight's own settings; IPOPT uses its
# tuned warm options for warm solves and its default options with MA57 for cold solves (no cold IPOPT setting
# passed the gates of the equal-budget study). Budget = flight budget (80). Sequential, CPUs 0-3.
#   benchmark/comparison/replay_campaign.sh SRC DST
# SRC: dataset with the flights (data or a run folder). DST: run folder for the replays. If DST differs from
# SRC, the flight records are copied to DST first, so SRC is never written. Existing replays are skipped.
# IPOPT warm options: the newest RUN/tuning/*_challenging_ipopt_warm of SRC (override with IPOPT_WARM=...).
set -u
[[ $# -eq 2 ]] || { echo "usage: $0 SRC DST" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
src=$1; dst=$2
case "$(realpath -m "$dst")" in "$root/data"|"$root/data/"*) echo "refusing to write into data/" >&2; exit 2;; esac
newest_ipopt() { for d in $(ls -td "$src"/tuning/*_challenging_ipopt_warm "$dst"/tuning/*_challenging_ipopt_warm 2>/dev/null); do [[ -f $d/selected_parameters.json ]] && { echo "$d"; return; }; done; }
IPOPT_WARM=${IPOPT_WARM:-$(newest_ipopt)/selected_parameters.json}
IPOPT_COLD=src/px4_nmpc_config/config/experiments/challenging_course/ipopt_cold_default.json
[[ -f $IPOPT_WARM ]] || { echo "missing $IPOPT_WARM (set IPOPT_WARM=...)" >&2; exit 2; }
set +u; set --; source scripts/setup_env.sh >/dev/null 2>&1; set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$dst/challenging_campaign"
for F in "$src"/challenging_campaign/{warm,cold}_[0-9][0-9]; do
  [[ -d $F ]] || continue
  n=$(basename "$F"); R=$dst/challenging_campaign/$n
  if [[ $(realpath -m "$R") != $(realpath -m "$F") && ! -e $R ]]; then
    mkdir -p "$R" && (cd "$F" && tar --exclude=./replay -cf - .) | (cd "$R" && tar -xf -)
  fi
  if [[ -f $R/replay/paired.json ]]; then echo "keep existing $R/replay"; continue; fi
  [[ -e $R/replay ]] && mv "$R/replay" "$R/replay.incomplete_$(date +%Y%m%d_%H%M%S)"
  case $n in
    warm_*) args=(--ipopt-params "$IPOPT_WARM" --ipopt-params-cold "$IPOPT_COLD") ;;
    cold_*) args=(--ipopt-params "$IPOPT_COLD") ;;
  esac
  # IPOPT_OPTIONS (JSON) overrides IPOPT options of every solve, e.g. '{"linear_solver": "mumps"}' without HSL
  [[ -n ${IPOPT_OPTIONS:-} ]] && args+=("--ipopt-options=$IPOPT_OPTIONS")
  taskset -c "${CPUS:-0-3}" venv/bin/python -m px4_nmpc_compare.challenging_replay "$R" "$R/replay" "${args[@]}" > "$R/replay.log" 2>&1
  echo "$n replay exit $?"
done
