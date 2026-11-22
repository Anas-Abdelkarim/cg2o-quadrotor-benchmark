"""Post-flight truth evaluation of a challenging-MPC PX4/Gazebo flight.

Runs the existing truth alignment (benchmark/flight/truth.py:
ULog ground truth, clock matching, x500 model-to-body offset), then measures
the true clearance of the five airframe spheres to all course obstacles
(spheres, posts, bars) with the true attitude. Truth is never fed back.

    python benchmark/flight/evaluate_flight.py RUN_DIR
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from truth import evaluate  # noqa: E402
from px4_nmpc_compare import challenging as C  # noqa: E402


def body_clearance(p, obstacles, pos, rpy):
    """Per-sample minimum clearance [m] of the five body spheres (no reserve)."""
    ext = C.extension()
    out = np.empty(pos.shape[1])
    worst = np.empty(pos.shape[1], int)
    for j in range(pos.shape[1]):
        R = ext.rotation(rpy[0, j], rpy[1, j], rpy[2, j])
        best, arg = 1e9, -1
        for i in range(5):
            w = pos[:, j] + R @ np.asarray(p.body)[i]
            for k, o in enumerate(obstacles):
                d = w - o[1:4]
                if int(o[0]) == 1:
                    d[2] = 0.
                elif int(o[0]) == 2:
                    d[1] = 0.
                c = float(np.linalg.norm(d) - o[4] - p.body_radius[i])
                if c < best:
                    best, arg = c, k
        out[j], worst[j] = best, arg
    return out, worst


def main(folder):
    folder = Path(folder)
    base = evaluate(folder)
    truth = np.load(folder / 'truth.npz')
    scene = json.loads((folder / 'runtime/scenario.json').read_text())
    obstacles = np.asarray(scene['typed_obstacles'], float)
    trace = [json.loads(l) for l in (folder / 'challenging_trace.jsonl').read_text().splitlines() if l]
    meta = trace[0]
    p = C.make_params(C.load_course(meta['course']), meta['overrides'])
    nav = truth['navigation']
    pos, rpy = truth['position'][:, nav], truth['rpy'][:, nav]
    clearance, worst = body_clearance(p, obstacles, pos, rpy)
    i = int(np.argmin(clearance))
    tilt = np.degrees(np.arccos(np.clip(np.cos(rpy[0]) * np.cos(rpy[1]), -1, 1)))
    problems = [r for r in trace if r.get('kind') == 'problem']
    result = dict(
        outcome=base['outcome'], time_to_goal_s=base['time_to_goal_wall_s'],
        min_true_body_clearance_m=float(clearance[i]), closest_obstacle=obstacles[worst[i]].tolist(),
        closest_position_hall_ned=pos[:, i].tolist(), tracking_reserve_m=float(p.margin),
        max_true_tilt_deg=float(tilt.max()), max_true_speed_mps=base['maximum_true_speed_mps'],
        filtered_jerk_rms_mps3=base['filtered_jerk_rms_mps3'], path_length_m=base['path_length_m'],
        vertical_span_m=base['vertical_span_m'], lateral_span_m=base['lateral_span_m'],
        solves=len(problems), usable=int(sum(r['usable'] for r in problems)),
        evaluation='Gazebo truth after the flight; five airframe spheres, true attitude, all course obstacles')
    (folder / 'challenging_evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main(sys.argv[1])
