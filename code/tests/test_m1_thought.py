"""5d 배선·선택·train-only 출력 계약. 가짜 학습기와 작은 입력만 사용한다."""
import importlib
import json

import numpy as np
import polars as pl
import pytest

from afp import evaluate, features, thought

runner = importlib.import_module("04g_m1_thought")


def test_oof_uses_each_folds_own_matrix_without_real_training(monkeypatch):
    D = {"X": np.zeros((10, 1)), "y": np.tile([False, True], 5),
         "inst": np.repeat([f"issue{i}" for i in range(5)], 2), "fold": np.repeat(np.arange(5), 2)}
    extras = {k: np.full((10, 1), k + 1.0) for k in range(5)}
    calls = []

    def fit(cfg, X, y, device):
        assert X.shape == (8, 2)
        assert device == "cpu"
        value = X[0, -1]
        assert np.all(X[:, -1] == value)
        calls.append(value)
        return value

    monkeypatch.setattr(runner.m1w, "fit_xgb", fit)
    monkeypatch.setattr(runner.m1, "predict_model", lambda model, X: np.full(len(X), model / 10))
    result = runner.oof_task("test", {}, D["y"], runner.fold_blocks(D, extras), "cpu")
    np.testing.assert_allclose(result["oof"], np.repeat(np.arange(1, 6) / 10, 2))
    assert calls == [1, 2, 3, 4, 5]
    assert result["seen_issue_scores"] == 0


def test_row_issue_overlap_and_incomplete_oof_fail(monkeypatch):
    D = {"X": np.zeros((10, 1)), "y": np.tile([False, True], 5),
         "inst": np.repeat(["same"] * 5, 2), "fold": np.repeat(np.arange(5), 2)}
    with pytest.raises(ValueError):
        runner.fold_blocks(D)
    monkeypatch.setattr(runner.m1w, "fit_xgb", lambda *a: None)
    monkeypatch.setattr(runner.m1, "predict_model", lambda m, x: np.zeros(len(x)))
    with pytest.raises(ValueError, match="coverage"):
        runner.oof_task("test", {}, np.array([False, True]), [(np.array([0]), np.array([1]), np.zeros((1, 1)), np.zeros((1, 1)))], "cpu")


def disc_rows(bundle_scores):
    return pl.DataFrame([{"bundle": v, "model": m, "N": n, "within_pair": score}
                         for v, score in bundle_scores.items() for m in runner.MODELS for n in evaluate.N_GRID])


def test_search_tie_prefers_larger_t_then_smaller_k_and_s3_prefers_fewer_features(monkeypatch):
    monkeypatch.setattr(runner, "S2", ("chg", ()))
    disc = disc_rows({f"lex_t{t}_k{k}": 0.6 for t, k in runner.SEARCH})
    assert runner.pick_vocab(disc, "m1_xgb")[0] == (100, 10)
    disc = disc_rows({"s2": 0.600, "length": 0.601, "lexical": 0.6015, "length_lexical": 0.602})
    assert runner.pick_bundle(disc, "m1_xgb", 10)[0] == "s2"
    with pytest.raises(ValueError, match="incomplete"):
        runner.mean_score(disc.filter(pl.col("N") != 20), "s2", "m1_xgb")


def test_column_contract_and_cuda_default_with_explicit_cpu_override(monkeypatch):
    monkeypatch.setattr(runner, "S2", ("chg", ()))
    assert len(runner.columns("s2", 3, False)) == 73
    assert len(runner.columns("length", 3, False)) == 77
    assert len(runner.columns("lexical", 3, False, 10)) == 83
    assert len(runner.columns("length_lexical", 5, True, 20)) == 133
    assert runner.parser().parse_args([]).device == "cuda"
    assert runner.parser().parse_args([]).gpu_jobs == 4
    assert runner.parser().parse_args(["--device", "cpu"]).device == "cpu"


def write_5c_summary(path, s2, criteria_ok=True, combo=("a2", "sl5")):
    path.write_text(json.dumps({"device": "cpu", "selection": {"S2": s2, "combo": list(combo)},
                                "criteria": {"C1": True, "C2": criteria_ok}}))


@pytest.mark.parametrize("s2", ("chg", "sl5"))
def test_s2_check_reads_5c_selection(tmp_path, monkeypatch, s2):
    monkeypatch.setattr(runner.m1t, "s1_check", lambda: {})
    monkeypatch.setattr(runner.m1t, "OUT_DIR", tmp_path)
    monkeypatch.setattr(runner, "S2", None)
    write_5c_summary(tmp_path / "summary.json", s2)
    upstream = runner.s2_check()
    assert upstream == {"phase5c_S2": s2, "phase5c_device": "cpu"}
    assert runner.S2 == (s2, ())
    assert runner.columns("s2", 10, False) == runner.m1t.bundle_cols(s2, 10, False)
    assert runner.columns("length", 10, True)[:len(runner.m1t.bundle_cols(s2, 10, True))] == runner.m1t.bundle_cols(s2, 10, True)


def test_s2_check_reads_combo_members(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.m1t, "s1_check", lambda: {})
    monkeypatch.setattr(runner.m1t, "OUT_DIR", tmp_path)
    monkeypatch.setattr(runner, "S2", None)
    write_5c_summary(tmp_path / "summary.json", "combo")
    assert runner.s2_check()["phase5c_combo"] == ["a2", "sl5"]
    assert runner.columns("s2", 10, False) == runner.m1t.bundle_cols("combo", 10, False, ("a2", "sl5"))


def test_s2_check_stops_on_incomplete_5c_criteria_or_unknown_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.m1t, "s1_check", lambda: {})
    monkeypatch.setattr(runner.m1t, "OUT_DIR", tmp_path)
    monkeypatch.setattr(runner, "S2", None)
    write_5c_summary(tmp_path / "summary.json", "sl5", criteria_ok=False)
    with pytest.raises(ValueError, match="incomplete criteria"):
        runner.s2_check()
    assert runner.S2 is None
    write_5c_summary(tmp_path / "summary.json", "nope")
    with pytest.raises(KeyError):
        runner.s2_check()
    assert runner.S2 is None
    with pytest.raises(RuntimeError, match="S2 not loaded"):
        runner.columns("s2", 10, False)


def fake_inputs():
    rows, steps = [], []
    for i in range(50):
        for success in (False, True):
            tid = f"t{i:03d}_{int(success)}"
            rows.append({"traj_id": tid, "instance_id": f"org__repo-{i:03d}", "fold": i // 10,
                         "target": success, "n_steps": 30})
            steps.extend({"traj_id": tid, "step_idx": t, "action_text": "works now" if success else "stuck again"}
                         for t in range(1, 21))
    train = pl.DataFrame(rows)
    texts = train.select("traj_id", "instance_id").with_columns(issue_text=pl.lit("stuck issue"))
    b1 = pl.DataFrame([{"traj_id": t, "N": n, "p_fail": 0.5} for t in train["traj_id"] for n in evaluate.N_GRID])
    return train, thought.TrainScope(train), pl.DataFrame(steps), texts, b1


def fake_base(train):
    result = {}
    for n in evaluate.N_GRID:
        f = train.select("traj_id").with_columns(N=pl.lit(n))
        result[n] = f.with_columns([pl.lit(0.0).alias(c) for c in runner.m1t.bundle_cols("chg", n, False)])
    return result, {"official_matches_04_m1_cache": None}


@pytest.mark.parametrize("skip_bootstrap", (True, False))
def test_smoke_end_to_end_with_fake_grid_has_no_test_or_gpu_or_official_outputs(tmp_path, monkeypatch, skip_bootstrap):
    inputs = fake_inputs()
    monkeypatch.setattr(runner, "s2_check", lambda: {"phase5c_S2": "chg", "phase5c_device": "cpu"})
    monkeypatch.setattr(runner, "S2", ("chg", ()))
    monkeypatch.setattr(runner, "load_train_inputs", lambda: inputs)
    monkeypatch.setattr(runner, "base_features", fake_base)
    monkeypatch.setattr(runner, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path / "official")
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path / "official_data")
    monkeypatch.setattr(runner.m1w, "check_device", lambda d: None if d == "cpu" else pytest.fail("GPU requested"))
    monkeypatch.setattr(runner.m1w, "fit_xgb", lambda *a: pytest.fail("real model fitting requested"))
    monkeypatch.setattr(evaluate, "load_eval_sets", lambda: pytest.fail("test loader requested"))
    monkeypatch.setattr(evaluate, "load_all_costs", lambda: pytest.fail("all-data costs requested"))
    monkeypatch.setattr(runner.metrics, "eval_costs", lambda train: pl.DataFrame([
        {"traj_id": t, "step_idx": i, "ctx": 100} for t in train["traj_id"] for i in range(1, 31)]))

    def fake_grid(D, extras, grid, device, jobs, key):
        runner.fold_blocks(D, extras)  # 실제 fold 연결만 검증
        p = D["y"].astype(float) * 0.6 + 0.2
        rows = [{"i": i, "oof": p, "oof_auroc": 1.0, "seen_issue_scores": 0} for i in range(len(grid))]
        return rows[0], rows

    monkeypatch.setattr(runner, "fit_grid", fake_grid)
    def fake_bootstrap(train, predictions, stat, n_boot):
        # 공용 통계 adapter의 키/비용/paired 배선 검증. 재표본 학습·실험은 하지 않는다.
        estimates = stat(train, predictions)
        return pl.DataFrame([{"stat": name, "estimate": value, "lo": value, "hi": value, "n_boot": n_boot}
                             for name, value in estimates.items()])

    monkeypatch.setattr(runner.metrics, "bootstrap", fake_bootstrap)
    args = ["--device", "cpu", "--smoke"]
    if skip_bootstrap:
        with pytest.raises(SystemExit, match="incomplete"):
            runner.main([*args, "--skip-bootstrap"])
    else:
        runner.main(args)
    assert not (tmp_path / "official").exists()
    assert not (tmp_path / "official_data").exists()
    summary = json.loads((tmp_path / "smoke" / "summary.json").read_text())
    assert summary["mode"] == "smoke"
    assert summary["criteria"]["C6_bootstrap_ci"] is (not skip_bootstrap)
    assert summary["selection"]["S3"] == "s2"
    ops = pl.read_csv(tmp_path / "smoke" / "operating_points_oof.csv")
    assert {"saved_steps", "saved_tokens", "d_saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi"} <= set(ops.columns)
    assert {"cap", "cap_proposal"} <= set(ops["policy"])
    assert (tmp_path / "smoke" / "oof" / "handoff.json").exists()
    if not skip_bootstrap:
        assert ops["d_saved_tokens_lo"].null_count() == 0
        assert ops.filter(pl.col("policy") == "cap")["d_saved_tokens_lo"][0] == 0
        assert ops.filter(pl.col("model") != "b0_cap")["do_saved_tokens_lo"].abs().max() == 0


def test_full_mode_candidate_shortage_stops_before_model_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "s2_check", lambda: {"phase5c_S2": "chg", "phase5c_device": "cpu"})
    monkeypatch.setattr(runner, "S2", ("chg", ()))
    monkeypatch.setattr(runner, "load_train_inputs", fake_inputs)
    monkeypatch.setattr(runner, "base_features", fake_base)
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path / "results")
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(runner.m1w, "check_device", lambda d: None)
    monkeypatch.setattr(runner, "fit_grid", lambda *a: pytest.fail("fitting started before candidate validation"))
    monkeypatch.setattr(runner.metrics, "eval_costs", lambda train: pl.DataFrame([
        {"traj_id": t, "step_idx": i, "ctx": 100} for t in train["traj_id"] for i in range(1, 31)]))
    with pytest.raises(ValueError, match="No model was trained"):
        runner.main(["--device", "cpu"])
    assert (tmp_path / "results" / "vocab_diagnostics.csv").exists()
    assert json.loads((tmp_path / "results" / "summary.json").read_text())["status"] == "incomplete"


def test_train_loader_requests_only_train(monkeypatch):
    # 로더 계약: parquet를 열지 않고 첫 eval_set 인자에서 중단한다.
    seen = []

    def stop(name):
        seen.append(name)
        raise RuntimeError("stop before disk IO")

    monkeypatch.setattr(runner.metrics, "eval_set", stop)
    with pytest.raises(RuntimeError, match="before disk"):
        runner.load_train_inputs()
    assert seen == ["train"]
