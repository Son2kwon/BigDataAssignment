"""M1 피처 테스트용 합성 스텝 데이터 (test_features.py, test_features_leakage.py가 같이 쓴다)."""
import numpy as np
import polars as pl

from afp import features

STEP_SCHEMA = {"traj_id": pl.String, "step_idx": pl.Int64, "command": pl.String, "command_name": pl.String, "obs_len": pl.Int64,
               "is_error": pl.Boolean, "edit_failed": pl.Boolean, "open_file": pl.String}


def random_steps(seed, n_traj=40, max_steps=25):
    rng = np.random.default_rng(seed)
    rows = []
    n_steps = rng.integers(1, max_steps, n_traj)
    names = list(features.COMMAND_NAMES)
    for i in range(n_traj):
        for s in range(1, n_steps[i] + 1):
            c = names[rng.integers(0, 6)]
            rows.append({"traj_id": f"t{i:03d}", "step_idx": s, "command": f"{c} {rng.integers(0, 3)}", "command_name": c,
                         "obs_len": int(rng.integers(0, 500)), "is_error": bool(rng.random() < 0.3), "edit_failed": bool(rng.random() < 0.1),
                         "open_file": None if rng.random() < 0.3 else f"/r/pkg/f{rng.integers(0, 4)}.py"})
    steps = pl.DataFrame(rows, schema=STEP_SCHEMA)
    trajs = pl.DataFrame({"traj_id": [f"t{i:03d}" for i in range(n_traj)], "instance_id": [f"i{i % 7}" for i in range(n_traj)],
                          "issue_text": [f"bug in pkg/f{i % 4}.py" if i % 3 else "no file" for i in range(n_traj)], "n_steps": n_steps})
    return steps, trajs, rng


def features_of(steps, trajs, n_grid):
    m = features.mentions_table(trajs)
    return features.m1_features(features.step_values(steps, m, trajs.select("traj_id", "instance_id")), trajs, n_grid, m)
