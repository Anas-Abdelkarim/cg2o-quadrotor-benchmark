# 4. Where each result comes from

Numbering as in Sec. VII of the paper. `<folder>` is an experiment folder written by
`experiments/run_experiment.py` (`results/experiments/<stamp>_<preset>/`); `<dataset>` is `data` (the paper's
measurements) or a run folder; `<analysis>` is `<folder>/analysis/`.

| Paper item | Computed by | Output | From |
| --- | --- | --- | --- |
| Table VIII, parameters | configuration | — | `src/px4_nmpc_config/config/experiments/challenging_course/course.yaml`, `cg2o_overrides.yaml` |
| Table IX, selected solver settings | `paper_results.py` | `<folder>/paper_table_ix.txt` | `<dataset>/tuning/*_challenging_{cg2o,ipopt}_{warm,cold}/selected_parameters.json`; IPOPT cold: `challenging_course/ipopt_cold_default.json` (no setting passed the quality limits) |
| Table X, closed-loop flights | `paper_results.py` (`campaign_summary.py`) | `<folder>/paper_table_x.txt` | each flight's `challenging_evaluation.json` (ground truth: goal, time, tilt) and `challenging_trace.jsonl` (in-flight solves, navigation phase) |
| Table XI, paired comparison | `paper_results.py` (`campaign_summary.py`, section `paper`) | `<folder>/paper_table_xi.txt` | each flight's `replay/paired.json`, navigation phase |
| Table XII, sensitivity | `paper_results.py` (`sensitivity.py`) | `<folder>/paper_table_xii.txt` | `<dataset>/tuning/*_challenging_cg2o_cold/trials.csv`; "42% faster": `report.txt` (score of the selected setting) |
| Fig. 10, the 20 flights | `paper_figures.py` | `<folder>/paper_fig10_course.pdf` | each flight's `truth.npz` (Gazebo ground truth) |
| Fig. 11, solve times | `paper_figures.py` | `<folder>/paper_fig11_solve_times.pdf` | each flight's `replay/paired.json` |
| Videos | `record_video.sh <run> warm\|cold` | `videos/quadrotor_{warm,cold}.mp4`; new: `results/<run>/video_<mode>/slalom_sitl.mp4` | flights with the selected settings, take-off point 1 |

## Numbers in the text

| Quantity | Value | Source |
| --- | --- | --- |
| Flights reaching the goal | 20 / 20 | `paper_table_x.txt` |
| Feasible cg2o solutions in flight | 2832 / 2832 | `paper_table_x.txt` |
| In-flight solves within 100 ms, warm / cold | 98.3% / 92.8% | `paper_table_x.txt` (notes) |
| Cold starts that are obstacle-infeasible | 2582 / 2832 (91%), 75 violated rows on average, up to 0.93 m | `paper_table_xi.txt` (notes) |
| Mean solve time, cold: cg2o / IPOPT | 46.7 / 176.7 ms (3.8×) | `paper_table_xi.txt` |
| Mean solve time, warm: cg2o / IPOPT | 21.7 / 144.8 ms (6.7×) | `paper_table_xi.txt` |
| Median cost of cg2o relative to IPOPT, cold / warm | +0.5% / +0.4% | `paper_table_xi.txt` (notes) |
| Sensitivity: 150 trials, 52 with a nonlinear or damped auxiliary update (51 rejected) | | `paper_table_xii.txt` (notes) |
| Median accepted trial vs. untuned setting | 1.245× faster | `paper_table_xii.txt` (notes) |
| Selected cold-start setting vs. untuned setting | 42.5% faster | `paper_table_xii.txt` (notes) |
