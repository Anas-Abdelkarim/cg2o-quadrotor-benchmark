# Thrust-and-rate quadrotor MPC benchmark. Step-by-step guide: QUICKSTART.md and docs/3_reproduce.md.
# The experiments of the paper also run through one entry point: python3 experiments/run_experiment.py <preset>.
#
#   RUN      folder of a new generation (default results/run_01). Use a new name for every new generation;
#            within a run, existing flights, replays and videos are kept and skipped.
#   DATASET  data that the tables and figures are made from: data (shipped, the paper) or a RUN folder.
# data/ is never written by any target. Timed work runs on the isolated CPUs given by CPUS (replays, tuning)
# and CONTROLLER_CPUS (MPC node in the PX4 flights), e.g. make tune CPUS=0-1.
.PHONY: help externals build test tuning-flights tune flights replay replay-shipped tables figures video all
.DEFAULT_GOAL := help
PYTHON  ?= venv/bin/python
CPUS    ?= 0-3
CONTROLLER_CPUS ?= 2,3
export CPUS CONTROLLER_CPUS
# g2o, cg2o and the extensions are compiled with -march=native, as for the paper (they must all match).
MARCH_NATIVE ?= ON
export MARCH_NATIVE
PIN     ?= taskset -c $(CPUS)
RUN     ?= results/run_01
DATASET ?= data
ANALYSIS = results/analysis/$(subst /,_,$(DATASET))
SPECS    = src/px4_nmpc_config/config/experiments/challenging_course/tuning

help:
	@printf '%s\n' \
	  'Generate everything (install, build, fly, tune, compare, video) into RUN=$(RUN):' \
	  '  make externals         IPOPT 3.14, HSL (HSL_SOURCE=...), g2o, cg2o, PX4 v1.17 SITL, XRCE-DDS agent' \
	  '  make build             solver extensions, benchmark extension, ROS 2 packages' \
	  '  make test              unit and integration tests' \
	  '  make tuning-flights    3 PX4 flights with untuned settings + reference replays' \
	  '  make tune              4 equal-budget tuning studies on those flights' \
	  '  make flights           10 warm + 10 cold PX4 flights with the tuned settings' \
	  '  make replay            paired cg2o/IPOPT replay of the 20 flights' \
	  '  make video             recorded PX4 flight (RViz + Gazebo)' \
	  '  make tables figures DATASET=$(RUN)' \
	  '  make all               every step above, in order' \
	  'Use the shipped data of the paper (data/, never modified):' \
	  '  make tables figures    tables and figures of the paper (DATASET=data, Python only)' \
	  '  make replay-shipped    re-solve the shipped flight problems into RUN (solvers only)'

externals:
	./scripts/build_externals.sh --solver-only
	./scripts/build_externals.sh --only=px4,xrce

build:
	PYTHON=$(CURDIR)/$(PYTHON) ./scripts/build_benchmark.sh

# One process per suite: the extensions link different IPOPT builds and must not share a process.
test:
	$(PYTHON) -m pytest -q tests
	$(PYTHON) -m pytest -q src/px4_nmpc_compare/test
	$(PYTHON) -m pytest -q src/px4_nmpc_core/test
	$(PYTHON) -m pytest -q tools/solver_tuning/tests
	$(PYTHON) -m pytest -q src/px4_nmpc_controller/test
	$(PYTHON) -m pytest -q src/px4_nmpc_tools/test

tuning-flights:
	benchmark/flight/tuning_flights.sh $(RUN)

TUNE_DATA ?= $(RUN)
tune:
	BENCH_DATA=$(TUNE_DATA) $(PIN) $(PYTHON) -m tuning run $(SPECS)/cg2o_warm.yaml  --trials 80  --out $(RUN)/tuning/challenging_cg2o_warm
	BENCH_DATA=$(TUNE_DATA) $(PIN) $(PYTHON) -m tuning run $(SPECS)/ipopt_warm.yaml --trials 80  --out $(RUN)/tuning/challenging_ipopt_warm
	BENCH_DATA=$(TUNE_DATA) $(PIN) $(PYTHON) -m tuning run $(SPECS)/cg2o_cold.yaml  --trials 150 --out $(RUN)/tuning/challenging_cg2o_cold
	BENCH_DATA=$(TUNE_DATA) $(PIN) $(PYTHON) -m tuning run $(SPECS)/ipopt_cold.yaml --trials 150 --out $(RUN)/tuning/challenging_ipopt_cold

flights:
	benchmark/flight/campaign.sh $(RUN) warm
	benchmark/flight/campaign.sh $(RUN) cold

replay:
	benchmark/comparison/replay_campaign.sh $(RUN) $(RUN)

replay-shipped:
	benchmark/comparison/replay_campaign.sh data $(RUN)

video:
	benchmark/flight/record_video.sh $(RUN) warm

tables:
	$(PYTHON) benchmark/analysis/campaign_summary.py $(DATASET)/challenging_campaign $(ANALYSIS)/campaign_summary.json
	$(PYTHON) benchmark/analysis/sensitivity.py $$(ls -td $(DATASET)/tuning/*_challenging_cg2o_cold | head -n1) $(ANALYSIS)/sensitivity.json

figures:
	$(PYTHON) benchmark/analysis/paper_figures.py $(ANALYSIS)/figures $(DATASET)/challenging_campaign

all: externals build test tuning-flights tune flights replay video
	$(MAKE) tables figures DATASET=$(RUN)
