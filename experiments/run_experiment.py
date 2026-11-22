#!/usr/bin/env python3
"""Run one experiment of the quadrotor benchmark (Sec. VII of the paper) into a new results folder.

    python3 experiments/run_experiment.py PRESET [--quick] [--cpus 0-3] [--controller-cpus 2,3]

Presets
    tables    Tables IX-XII and Figs. 10-11 from the measurements shipped in data/ (Python only, seconds).
    replay    Re-solve every recorded MPC problem of the 20 shipped flights with cg2o and IPOPT (paired
              comparison), then all tables and figures. Needs the solver build and HSL (about 50 min).
              --quick: one warm and one cold flight (about 5 min). --no-hsl: IPOPT with MUMPS instead of the
              HSL solvers MA27/MA57 of the paper (no HSL licence needed; IPOPT's numbers then differ).
    tuning    The four equal-budget tuning studies on the shipped tuning flights, then all tables and figures.
              Needs the solver build and HSL (many hours). --quick: 3 trials per study.
    full      Everything from scratch with PX4 SITL: tuning flights, tuning, 20 flights, paired replay, video,
              tables and figures. Needs the full installation (about a day). --quick is not available.

Every run writes into its own folder results/experiments/<YYYYmmdd_HHMMSS>_<preset>[_quick]/ with run.json
(preset, status, commit, times), the paper's tables (paper_table_*.txt/.json) and figures (paper_fig*_*.pdf).
experiments/paper_results.py rebuilds the tables and figures of any such folder.
Timed steps are pinned to the CPUs given by --cpus (the paper: isolated CPUs 0-3) and run one at a time.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
VENV_PY = ROOT / 'venv/bin/python'
EXTENSION_DIR = ROOT / 'build/challenging_mpc'
HSL_LIB = ROOT / 'install/externals/hsl/lib/x86_64-linux-gnu/libcoinhsl.so'
PX4_BIN = ROOT / 'install/externals/PX4-Autopilot/bin/px4'
SPECS = ROOT / 'src/px4_nmpc_config/config/experiments/challenging_course/tuning'
PRESETS = ('tables', 'replay', 'tuning', 'full')
STUDIES = (('cg2o_warm', 80), ('ipopt_warm', 80), ('cg2o_cold', 150), ('ipopt_cold', 150))


# --------------------------------------------------------------------------------------------- checks
def fail(msg):
    sys.exit(f'ERROR: {msg}')


def check_python(preset):
    need = ['numpy', 'scipy', 'matplotlib', 'yaml']
    if preset in ('tuning', 'full'):
        need += ['optuna', 'tuning']
    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    code = 'import importlib.util,sys; m=[n for n in sys.argv[1:] if importlib.util.find_spec(n) is None]; print(" ".join(m))'
    missing = subprocess.run([str(py), '-c', code, *need], capture_output=True, text=True).stdout.split()
    if missing:
        fail(f'Python packages missing in {py}: {", ".join(missing)}. Install them with '
             f'venv/bin/pip install -r requirements.txt (QUICKSTART.md, step 2).')


def env_shell(cmd):
    """Run cmd in bash after sourcing scripts/setup_env.sh (as the benchmark scripts do)."""
    return ['bash', '-c', f'set --; source scripts/setup_env.sh >/dev/null 2>&1; {cmd}']


def check_solvers(hsl=True):
    if not VENV_PY.exists():
        fail('venv/bin/python not found; create the virtual environment first (QUICKSTART.md, step 2).')
    if not list(EXTENSION_DIR.glob('_challenging*.so')):
        fail(f'benchmark extension not built in {EXTENSION_DIR}; run ./scripts/build_benchmark.sh (QUICKSTART.md, step 4).')
    r = subprocess.run(env_shell('venv/bin/python -c "from px4_nmpc_compare import challenging as C; '
                                 'e=C.extension(); assert hasattr(e, \'CGSession\') and hasattr(e, \'IPOPTSession\')"'),
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        fail(f'the benchmark extension does not load:\n{r.stderr.strip()[-2000:]}')
    if hsl and not HSL_LIB.exists():
        fail(f'HSL library missing ({HSL_LIB}); IPOPT uses MA57/MA27 from it. Build it with '
             f'HSL_SOURCE=<coinhsl archive> ./scripts/build_externals/hsl.sh (QUICKSTART.md, step 3), '
             f'or run the replay with --no-hsl (IPOPT with MUMPS).')


def check_px4():
    if not PX4_BIN.exists():
        fail(f'PX4 SITL not installed ({PX4_BIN}); run ./scripts/build_externals.sh --only=px4,xrce (INSTALL.md).')
    if not (ROOT / 'install/setup.bash').exists():
        fail('ROS 2 workspace not built; run ./scripts/build_workspace.sh (INSTALL.md).')


# --------------------------------------------------------------------------------------------- steps
def run(cmd, log, env, shell=False):
    print(f'$ {cmd if shell else " ".join(map(str, cmd))}   (log: {log})', flush=True)
    with open(log, 'a') as fh:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, shell=shell,
                           executable='/bin/bash' if shell else None)
    if r.returncode:
        raise RuntimeError(f'step failed with exit code {r.returncode}; see {log}')


def quick_source(folder):
    """A dataset with one warm and one cold shipped flight (links into data/) for replay --quick."""
    src = folder / 'quick_source'
    (src / 'challenging_campaign').mkdir(parents=True)
    for name in ('warm_01', 'cold_01'):
        (src / 'challenging_campaign' / name).symlink_to(DATA / 'challenging_campaign' / name)
    (src / 'tuning').symlink_to(DATA / 'tuning')
    return src


def replay(folder, quick, env):
    src = quick_source(folder) if quick else DATA
    run(['benchmark/comparison/replay_campaign.sh', str(src), str(folder)], folder / 'replay.log', env)
    flights = sorted((folder / 'challenging_campaign').glob('*_[0-9][0-9]'))
    missing = [f.name for f in flights if not (f / 'replay/paired.json').exists()]
    expected = 2 if quick else 20
    if len(flights) != expected or missing:
        raise RuntimeError(f'replay incomplete: {len(flights)} flights, without paired.json: {missing}; '
                           f'see <flight>/replay.log')
    if quick:
        shutil.rmtree(folder / 'quick_source')


def tuning(folder, quick, env):
    out = folder / 'tuning'
    out.mkdir(parents=True, exist_ok=True)
    tenv = dict(env, BENCH_DATA=str(DATA.relative_to(ROOT)))
    for name, trials in STUDIES:
        n = 3 if quick else trials
        # the tuning tool prefixes the folder name with its start time: tuning/<stamp>_challenging_<name>
        run(['taskset', '-c', env['CPUS'], str(VENV_PY), '-m', 'tuning', 'run', str(SPECS / f'{name}.yaml'),
             '--trials', str(n), '--out', str(out / f'challenging_{name}')], folder / 'tuning.log', tenv)


def full(folder, env):
    rel = folder.relative_to(ROOT)
    for target in ('tuning-flights', 'tune', 'flights', 'replay', 'video'):
        if target == 'video' and not (env.get('DISPLAY') or env.get('WAYLAND_DISPLAY')):
            print('no display: the video flight is skipped (run make video RUN=... on a desktop)', flush=True)
            continue
        run(['make', target, f'RUN={rel}'], folder / f'make_{target}.log', env)


# --------------------------------------------------------------------------------------------- main
def git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out + (' (modified)' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], epilog='See QUICKSTART.md.')
    ap.add_argument('preset', choices=PRESETS)
    ap.add_argument('--quick', action='store_true', help='short version (replay: 2 flights; tuning: 3 trials per study)')
    ap.add_argument('--no-hsl', action='store_true',
                    help='replay only: IPOPT with MUMPS instead of HSL MA27/MA57 (the paper used HSL)')
    ap.add_argument('--cpus', default=os.environ.get('CPUS', '0-3'), help='CPUs of the timed steps (default 0-3)')
    ap.add_argument('--controller-cpus', default=os.environ.get('CONTROLLER_CPUS', '2,3'),
                    help='CPUs of the MPC node in PX4 flights (default 2,3)')
    a = ap.parse_args(argv)
    if a.quick and a.preset in ('tables', 'full'):
        fail(f'--quick is not available for {a.preset}')
    if a.no_hsl and a.preset != 'replay':
        fail('--no-hsl is available for replay only (the tuning searches over the HSL solvers; tables reads data/)')

    check_python(a.preset)
    if a.preset in ('replay', 'tuning', 'full'):
        check_solvers(hsl=not a.no_hsl)
    if a.preset == 'full':
        check_px4()

    name = time.strftime('%Y%m%d_%H%M%S') + f'_{a.preset}' + ('_quick' if a.quick else '') + ('_mumps' if a.no_hsl else '')
    folder = ROOT / 'results/experiments' / name
    folder.mkdir(parents=True)
    # flights: the shipped ones for tables and tuning, the run's own otherwise; tuning studies: the run's own
    # for tuning and full, the shipped ones otherwise
    dataset = 'data' if a.preset in ('tables', 'tuning') else str(folder.relative_to(ROOT))
    tuning_studies = str(folder.relative_to(ROOT) / 'tuning') if a.preset in ('tuning', 'full') else 'data/tuning'
    info = dict(preset=a.preset, quick=a.quick, ipopt_hsl=not a.no_hsl, status='running', dataset=dataset, tuning=tuning_studies,
                commit=git_commit(),
                cpus=a.cpus, controller_cpus=a.controller_cpus, command=' '.join(sys.argv),
                started=time.strftime('%Y-%m-%d %H:%M:%S'))
    (folder / 'run.json').write_text(json.dumps(info, indent=1))
    print(f'experiment folder: {folder}', flush=True)
    env = dict(os.environ, CPUS=a.cpus, CONTROLLER_CPUS=a.controller_cpus, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if a.no_hsl:
        env['IPOPT_OPTIONS'] = json.dumps({'linear_solver': 'mumps'})
    t0 = time.time()
    try:
        if a.preset == 'replay':
            replay(folder, a.quick, env)
        elif a.preset == 'tuning':
            tuning(folder, a.quick, env)
        elif a.preset == 'full':
            full(folder, env)
        py = str(VENV_PY if VENV_PY.exists() else sys.executable)
        run([py, 'experiments/paper_results.py', str(folder)], folder / 'paper_results.log', env)
        info['status'] = 'complete'
    except (RuntimeError, KeyboardInterrupt) as exc:
        info['status'] = 'failed'
        info['error'] = str(exc) or type(exc).__name__
        print(f'ERROR: {info["error"]}', file=sys.stderr)
    info['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    info['duration_s'] = round(time.time() - t0, 1)
    (folder / 'run.json').write_text(json.dumps(info, indent=1))
    if info['status'] == 'complete':
        for f in ('paper_table_x.txt', 'paper_table_xi.txt'):
            print('\n' + (folder / f).read_text())
        print(f'\ndone: {folder}')
    return 0 if info['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
