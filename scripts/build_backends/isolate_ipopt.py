#!/usr/bin/env python3
"""Give the native ipopt_cpp extension a private copy of the local IPOPT (renamed sonames).

CasADi bundles its own IPOPT 3.14.x and MUMPS with the sonames libipopt.so.3 / libcoinmumps.so.3, the same
sonames as the IPOPT built from GitHub by scripts/build_externals/ipopt.sh. The dynamic loader keeps one library
per soname, so a process that uses both `ipopt_casadi` and `ipopt_cpp` (tests, comparisons) would run one of them
against the other's IPOPT and crash. This step copies libipopt / libcoinmumps into <build>/ipopt_private/ as
libpx4nmpc_ipopt.so.3 / libpx4nmpc_coinmumps.so.3 and points the extension at them. Python loads extensions with
RTLD_LOCAL, so the two IPOPTs then coexist. Idempotent; called by scripts/build_backends/ipopt_cpp.sh.

    python3 scripts/build_backends/isolate_ipopt.py --ipopt-lib install/externals/ipopt/lib \
        --ext build/ipopt_cpp_backend/cpp_px4_nmpc_ipopt_cpp*.so --out build/ipopt_cpp_backend/ipopt_private
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

RENAMES = {'libipopt.so.3': 'libpx4nmpc_ipopt.so.3', 'libcoinmumps.so.3': 'libpx4nmpc_coinmumps.so.3'}


def patchelf():
    for cand in (shutil.which('patchelf'), str(Path(sys.executable).parent / 'patchelf')):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit('patchelf not found: python3 -m pip install patchelf (or sudo apt install patchelf)')


def run(*args):
    return subprocess.run([patchelf(), *map(str, args)], check=True, capture_output=True, text=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ipopt-lib', required=True)
    ap.add_argument('--ext', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    lib, out, ext = Path(a.ipopt_lib).resolve(), Path(a.out).resolve(), Path(a.ext).resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_rpath = run('--print-rpath', lib / 'libipopt.so.3')
    for old, new in RENAMES.items():
        src = (lib / old).resolve()
        dst = out / new
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        run('--set-soname', new, dst)
    ipopt = out / RENAMES['libipopt.so.3']
    run('--replace-needed', 'libcoinmumps.so.3', RENAMES['libcoinmumps.so.3'], ipopt)
    run('--set-rpath', ':'.join(['$ORIGIN'] + [p for p in original_rpath.split(':') if p]), ipopt)
    mumps = out / RENAMES['libcoinmumps.so.3']
    run('--set-rpath', ':'.join(['$ORIGIN'] + [p for p in run('--print-rpath', mumps).split(':') if p]), mumps)
    needed = run('--print-needed', ext).split()
    if 'libipopt.so.3' in needed:
        run('--replace-needed', 'libipopt.so.3', RENAMES['libipopt.so.3'], ext)
    run('--force-rpath', '--set-rpath', f'{out}:{lib}', ext)
    print(f'ipopt_cpp uses a private IPOPT: {ipopt} (CasADi\'s bundled IPOPT can be loaded in the same process)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
