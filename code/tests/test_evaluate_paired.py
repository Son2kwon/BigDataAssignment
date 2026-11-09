"""두 모델 짝지은 비교 (evaluate.pair_preds·paired_stat)와 N 요약 (evaluate.n_summary)."""
import numpy as np
import polars as pl
import pytest

from afp import evaluate, metrics


def population(seed=0, n_traj=400):
    rng = np.random.default_rng(seed)
    n_steps = rng.integers(1, 40, n_traj)
    trajs = pl.DataFrame({"traj_id": [f"t{i:04d}" for i in range(n_traj)], "instance_id": [f"i{i % 41}" for i in range(n_traj)],
                          "target": rng.random(n_traj) < 0.3, "n_steps": n_steps})
    costs = pl.DataFrame({"traj_id": [f"t{i:04d}" for i in range(n_traj) for _ in range(n_steps[i])],
                          "step_idx": [s for i in range(n_traj) for s in range(1, n_steps[i] + 1)],
                          "ctx": rng.integers(100, 1000, int(n_steps.sum()))})
    return trajs, costs, rng


def model_preds(trajs, rng, n, folds=3):
    alive = metrics.running(trajs, n)
    base = (~alive["target"]).cast(pl.Float64).to_numpy() * 0.3 + rng.random(alive.height) * 0.7
    p = pl.DataFrame({"traj_id": alive["traj_id"], "N": [n] * alive.height, "p_fail": base})
    for k in range(folds):
        p = p.with_columns(pl.Series(f"{evaluate.FOLD_SCORE_PREFIX}{k}", np.clip(base + rng.normal(0, 0.05, alive.height), 0, 1)))
    return p


def test_paired_stat_against_itself_is_zero_and_matches_single():
    trajs, costs, rng = population()
    n = 10
    a = model_preds(trajs, rng, n)
    points = {"single": (0.6, None), "combined": (0.7, 25)}
    stat = evaluate.paired_stat(n, points, points, costs)
    out = stat(trajs, evaluate.pair_preds(a, a))
    assert all(v == 0 for k, v in out.items() if ".d_" in k or k.startswith("d_"))
    d = evaluate.discrimination_row(trajs, a.select("traj_id", "N", "p_fail"), n)
    assert out["a.auroc"] == d["auroc"] and out["b.within_pair"] == d["within_pair"]
    pt = evaluate.test_point(trajs, a, n, 0.7, 25, costs, spread=False)
    assert out["combined.a.saved_steps"] == pytest.approx(pt["saved_steps"])


def test_paired_stat_difference_and_bootstrap():
    trajs, costs, rng = population(1)
    n = 8
    a, b = model_preds(trajs, rng, n), model_preds(trajs, rng, n)
    pa, pb = {"single": (0.6, None)}, {"single": (0.65, None)}
    stat = evaluate.paired_stat(n, pa, pb, costs)
    paired = evaluate.pair_preds(a, b)
    out = stat(trajs, paired)
    ra = evaluate.test_point(trajs, a, n, 0.6, None, costs, spread=False)
    rb = evaluate.test_point(trajs, b, n, 0.65, None, costs, spread=False)
    assert out["single.d_saved_steps"] == pytest.approx(ra["saved_steps"] - rb["saved_steps"])
    assert out["d_auroc"] == pytest.approx(out["a.auroc"] - out["b.auroc"])
    boot = metrics.bootstrap(trajs, paired, stat, n_boot=30)
    row = boot.filter(pl.col("stat") == "d_auroc").row(0, named=True)
    assert row["lo"] <= row["estimate"] <= row["hi"]
    with pytest.raises(ValueError):
        evaluate.pair_preds(a, b.head(b.height - 1))


def test_n_summary_has_tokens_and_k_star_first():
    rows = [{"model": "m", "policy": "single", "N": 10, "split": "test", "saved_steps": 0.2, "saved_tokens": 0.3},
            {"model": "b0_cap", "policy": "cap", "N": None, "split": "test", "saved_steps": 0.19, "saved_tokens": 0.34},
            {"model": "m", "policy": "single", "N": 10, "split": "train", "saved_steps": 0.2, "saved_tokens": 0.3}]
    ops = evaluate.frame(rows, evaluate.OP_COLUMNS, evaluate.OP_SCHEMA)
    s = evaluate.n_summary(ops)
    assert s.columns == evaluate.N_SUMMARY_COLUMNS and s.height == 2 and s["model"][0] == "b0_cap"
    assert {"saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi"} <= set(s.columns)
