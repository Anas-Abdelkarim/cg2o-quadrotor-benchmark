#!/usr/bin/env bash
# Use the original branch's PX4/RViz/Gazebo workflow with a selectable model.
set -e
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 linear|nonlinear|challenging OUTPUT [sitl_video options]" >&2
  exit 2
fi
warm_v2_model="$1"
warm_v2_output="$2"
shift 2
case "$warm_v2_model" in linear|nonlinear|challenging) ;; *) echo 'Choose linear, nonlinear or challenging' >&2; exit 2;; esac
warm_v2_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$warm_v2_root"
warm_v2_extra=("$@")
set --
source scripts/setup_env.sh >/dev/null
set -- "${warm_v2_extra[@]}"
warm_v2_python="$warm_v2_root/venv/bin/python"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
warm_v2_check="$(mktemp /tmp/warm-v2-loader.XXXXXX)"
trap 'rm -f "$warm_v2_check"' EXIT
# The old backend links optional MKL dependencies. On this workstation they
# require the OpenMP/core/interface libraries in the global loader scope.
if ! "$warm_v2_python" -c 'from px4_nmpc_controller.px4_nmpc_cg2o.solver import _load_extension; _load_extension()' >"$warm_v2_check" 2>&1; then
  warm_v2_iomp=/opt/intel/oneapi/compiler/latest/lib/libiomp5.so
  warm_v2_mkl=/opt/intel/oneapi/mkl/latest/lib/intel64
  if [[ -f "$warm_v2_iomp" && -f "$warm_v2_mkl/libmkl_core.so" ]]; then
    export LD_PRELOAD="$warm_v2_iomp:$warm_v2_mkl/libmkl_core.so:$warm_v2_mkl/libmkl_intel_lp64.so:$warm_v2_mkl/libmkl_intel_thread.so${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  "$warm_v2_python" -c 'from px4_nmpc_controller.px4_nmpc_cg2o.solver import _load_extension; _load_extension()' >"$warm_v2_check" 2>&1 || { cat "$warm_v2_check" >&2; exit 1; }
fi
warm_v2_interface=trajectory
warm_v2_model_args=()
if [[ "$warm_v2_model" == nonlinear ]]; then
  warm_v2_interface=body_rate
  warm_v2_model_args=(--nonlinear-profile matched --nonlinear-tracking-margin 0.17
    --nonlinear-backtracking-min 0.1 --speed 2.3)
  "$warm_v2_python" -c 'from px4_nmpc_controller.nonlinear_solver import extension; assert hasattr(extension(), "CGSession")'
fi
if [[ "$warm_v2_model" == challenging ]]; then
  # Course, horizon and NLP come from experiments/challenging_course/course.yaml.
  warm_v2_interface=body_rate
  "$warm_v2_python" -c 'from px4_nmpc_compare.challenging import extension; assert hasattr(extension(), "CGSession")'
fi
warm_v2_budget=250
if [[ "$warm_v2_model" == challenging ]]; then
  warm_v2_budget="$("$warm_v2_python" -c 'from px4_nmpc_compare.challenging import load_course; print(int(load_course()["mpc"]["flight_max_iterations"]))')"
fi
"$warm_v2_python" -m px4_nmpc_tools.sitl_video --prediction-model "$warm_v2_model" \
  --control-interface "$warm_v2_interface" \
  "${warm_v2_model_args[@]}" \
  --max-iterations "$warm_v2_budget" --timeout 80 --controller-cpus "${CONTROLLER_CPUS:-2,3}" \
  --output "$warm_v2_output" "$@"
