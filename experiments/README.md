# experiments/

| File | Role |
| --- | --- |
| `run_experiment.py` | one entry point for every experiment of Sec. VII: `tables`, `replay`, `tuning`, `full` (`--quick` for a short check). Checks the Python packages, the benchmark extension, HSL and PX4 first; writes a new folder `results/experiments/<stamp>_<preset>/` |
| `paper_results.py` | the paper's tables (IX to XII) and figures (10, 11) of such a folder; without arguments the newest complete folder that is not a `--quick` run; `--compare` checks it against the paper |
| `paper_reference.yaml` | the numbers printed in the paper |

Each folder holds `run.json` (preset, status, commit, CPUs, times), `paper_table_*.txt/.json`,
`paper_fig10_course.pdf`, `paper_fig11_solve_times.pdf`, `analysis/` and the logs of every step. See
[QUICKSTART.md](../QUICKSTART.md).
