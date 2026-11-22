#!/usr/bin/env python3
"""Check which solver backends work on this machine (one real NMPC solve each).

    python3 scripts/smoke_test_backends.py                 # all three backends
    python3 scripts/smoke_test_backends.py cg2o ipopt_cpp  # selected backends

Exit code 0 when every requested backend solved the test problem.  Each backend runs
in its own process, so a crash inside a native extension is reported instead of
killing this script.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
PYTHONPATH = os.pathsep.join(str(SRC / d) for d in
                             ('px4_nmpc_core', 'px4_nmpc_controller', 'px4_nmpc_compare', 'px4_nmpc_tools'))

CHILD = r'''
import sys
import numpy as np
from px4_nmpc_core.config import build_configs
from px4_nmpc_core.ocp import make_ocp_solver

backend = sys.argv[1]
scfg, _, _ = build_configs({'solver_backend': backend, 'control_interface': 'trajectory',
                            'horizon_steps': 25, 'solver_allow_fallback': False})
solver = make_ocp_solver(scfg)
N = solver.N
pref = np.zeros((3, N + 1)); vref = np.zeros((3, N + 1))
for k in range(N + 1):
    pref[:, k] = [0.1 * k, 0.0, -2.0]; vref[:, k] = [1.0, 0.0, 0.0]
obstacles = np.array([[3.8, 0.0, -2.0, 1.2]])
res = solver.solve_ocp(np.zeros(9), pref, vref, obstacles,
                       arena_lb=np.array([-7., -5., -5.]), arena_ub=np.array([7., 5., 0.]), shift=None)
ok = bool(res.usable) and res.X is not None and np.all(np.isfinite(res.X))
print(f'{backend}: usable={res.usable} status={res.status} iterations={res.iterations} '
      f'solve_ms={res.solve_ms:.1f} cost={res.cost:.4f}')
sys.exit(0 if ok else 1)
'''


def main(argv):
    backends = argv or ['ipopt_casadi', 'ipopt_cpp', 'cg2o']
    env = dict(os.environ, PYTHONPATH=PYTHONPATH + os.pathsep + os.environ.get('PYTHONPATH', ''))
    failed = []
    for backend in backends:
        proc = subprocess.run([sys.executable, '-c', CHILD, backend], env=env,
                              capture_output=True, text=True)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(backend + ':')]
        if proc.returncode == 0 and lines:
            print(f'  OK    {lines[-1]}')
            continue
        failed.append(backend)
        if proc.returncode < 0:
            why = f'crashed with signal {-proc.returncode} (SIGSEGV = 11)'
            if backend == 'cg2o':
                why += ('\n          Usually a compiler-flag mismatch between g2o/cg2o and the backend '
                        '(-march=native).\n          Fix: ./scripts/build_externals/g2o.sh --clean && ./scripts/rebuild_cg2o.sh --clean '
                        '')
        else:
            why = (proc.stderr.strip().splitlines() or proc.stdout.strip().splitlines() or ['no output'])[-1]
        print(f'  FAIL  {backend}: {why}')
    print('\nall requested backends work' if not failed else f'\nnot working: {", ".join(failed)}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
