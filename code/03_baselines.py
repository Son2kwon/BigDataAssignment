"""Phase 4: 기준선 B0·B1, 참고용 오라클, 운영점 선택(3.5) 적용

실행: uv run --no-sync python code/03_baselines.py [--force] [--n-boot 1000] [--skip-bootstrap] [--jobs 16]

입력: data/processed/trajectories.parquet, steps.parquet (01_parse.py), splits.parquet (02_split.py)
평가 집합은 70B 궤적뿐이다 (train, test). 8B·405B 궤적은 제안서 수치 재현 검사(C4·C6)에만 읽는다.
- data/processed/preds/<model>.parquet  traj_id, N, p_fail (train OOF + test)
    b0_err, b0_editfail       앞 N스텝 is_error / edit_failed 개수 (학습 없음)
    b1_static, b1_n           이슈 본문 TF-IDF + LR. static = N=0 모델 하나, n = N마다 진행 중 궤적으로 재학습
    ref_oracle_loo            같은 집합·같은 이슈 다른 궤적 성공률 (참고 천장, 피처 사용 금지)
- data/processed/context_costs.parquet  스텝별 맥락 길이 캐시
- results/phase4/                       판별 표, 고정 상한 표, 운영점 표, 운영점 이전성 표, 비용 곡선, B1 그리드, 요약

예측 파일이 이미 있으면 다시 계산한 결과와 같은지 확인한다. 다르면 멈추고, --force면 덮어쓴다.
끝에 완료 기준(PLAN_4.md 4.7)을 검사하고 다르면 assert로 멈춘다.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from afp import baselines, evaluate, io, metrics, oof, preds, split

MODELS = ("b0_err", "b0_editfail", "b1_static", "b1_n", "ref_oracle_loo")
OP_MODELS = ("b0_err", "b0_editfail", "b1_static", "b1_n")  # 운영점 표에 넣는 모델 (오라클 제외)
B0_COLUMNS = {"b0_err": "is_error", "b0_editfail": "edit_failed"}
B1_GRID = tuple(
    baselines.B1Config(analyzer=a, C=c, weighting=w)
    for a in ("word", "char_wb") for c in (0.03, 0.1, 0.3, 1.0, 3.0) for w in ("traj", "issue")
)
B1_TIE = 0.002  # OOF AUROC 차이가 이보다 작으면 더 단순한 설정 (word, 작은 C, traj 가중)
K_PROPOSAL = 35  # 제안서의 고정 상한 (전체 데이터에서 고른 값, 참고 행)
FIXED_CAP_MAX_K = 80
CURVE_N = (0, 10)  # b1_scale 분위수 표의 N
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
FLOAT = 6

EXPECTED = {
    "split_trajectories": {"train": 62_311, "test": 12_481},
    "issues_after_strip": 3_591,
    "issues_with_hint": 13,
    "k_star": 35,  # train에서 손실 ≤5%인 최소 K (70B 무작위 분할 기준. 예전 분할에서는 34)
    # 제안서 재현 (분할과 무관한 전체 80,036궤적, 8B·405B 포함): 하네스 검사용
    "oracle_full_loo_auroc": 0.945,
    "oracle_full_loo_n_traj": 80_026,
    # Phase 3 결과 (전체 데이터 고정 상한): K → (손실, 절약 스텝). 반올림 자릿수까지 비교
    "fixed_cap_all": {30: (0.070, 0.294), 35: (0.049, 0.246), 40: (0.037, 0.208), 50: (0.022, 0.158)},
    "fixed_cap_all_k35_tokens": 0.422,
}


def parallel(fn, tasks, jobs):
    """작업마다 결과가 정해져 있어(고정 시드) 실행 순서와 무관하게 같은 값이 나온다."""
    if jobs <= 1:
        return [fn(*t) for t in tasks]
    os.environ["POLARS_MAX_THREADS"] = "1"  # 이제 뜨는 작업 프로세스에만 적용 (이 프로세스는 이미 스레드 풀이 있음)
    return Parallel(n_jobs=jobs, backend="loky", inner_max_num_threads=1)(delayed(fn)(*t) for t in tasks)


# ---- 예측 만들기 ----

def build_b0(sets, column):
    flags = baselines.load_step_flags(max(evaluate.N_GRID))
    allt = pl.concat([t for t in sets.values()])
    return baselines.count_scores(flags, allt, evaluate.N_GRID, column)


def trajectory_auroc(trajs, issue_scores, n):
    a = metrics.align(trajs, baselines.to_trajectories(trajs, issue_scores, n), n)
    return metrics.discrimination(a["target"], a["p_fail"])


def b1_grid_task(config, rows, train):
    fit, predict = baselines.b1_functions(config)
    res = oof.oof_predict(rows, fit, predict)
    d = trajectory_auroc(train, res.train.select("instance_id", "p_fail"), 0)
    return {**config.as_dict(), "ngram_range": str(config.ngram_range()), "oof_auroc": d["auroc"], "oof_pr_auc": d["pr_auc"],
            "seen_issue_scores": res.seen_issue_scores}


def choose_b1(grid: pl.DataFrame) -> int:
    best = grid["oof_auroc"].max()
    near = grid.with_row_index("i").filter(pl.col("oof_auroc") >= best - B1_TIE)
    near = near.sort(pl.col("analyzer") != "word", "C", pl.col("weighting") != "traj")
    return int(near["i"][0])


def b1_for_n(sets, texts, n, config):
    """시점 N의 B1: train OOF 이슈 점수 + 테스트 이슈 fold 평균 점수, 그리고 학습 이슈 전체로 다시 학습한
    모델(in-fold 누수 크기와 refit 테스트 점수 비교용) 점수."""
    rows = baselines.issue_rows(sets["train"], n, texts)
    test_rows = baselines.issue_rows(pl.concat([sets[s] for s in evaluate.TEST_SPLITS]), n, texts)
    fit, predict = baselines.b1_functions(config)
    res = oof.oof_predict(rows, fit, predict, test=test_rows)
    full = fit(rows)
    scores = pl.concat([res.train.select("instance_id", "p_fail"), res.test.select("instance_id", "p_fail")])
    in_fold = rows.select("instance_id", p_fail=pl.Series(predict(full, rows)))
    refit = test_rows.select("instance_id", p_fail=pl.Series(predict(full, test_rows)))
    fold_test = [test_rows.select("instance_id", p_fail=pl.Series(v)) for v in res.test_fold_scores]
    return {"n": n, "scores": scores, "in_fold": in_fold, "refit": refit, "fold_test": fold_test,
            "seen_issue_scores": res.seen_issue_scores, "fold_issues": res.fold_issues}


def issue_preds(sets, issue_scores, n_grid):
    """이슈 점수(N마다 하나씩 또는 모든 N에 같은 표) → 모든 분할의 궤적 예측."""
    out = []
    for n in n_grid:
        s = issue_scores[n] if isinstance(issue_scores, dict) else issue_scores
        out += [baselines.to_trajectories(t, s, n) for t in sets.values()]
    return pl.concat(out)


def build_b1(sets, texts, jobs, out_dir):
    train = sets["train"]
    rows0 = baselines.issue_rows(train, 0, texts)
    grid = pl.DataFrame(parallel(b1_grid_task, [(c, rows0, train) for c in B1_GRID], jobs))
    pick = choose_b1(grid)
    config = B1_GRID[pick]
    grid = grid.with_columns(selected=pl.int_range(pl.len()) == pick)
    grid.write_csv(out_dir / "b1_grid.csv", float_precision=FLOAT)
    assert (grid["seen_issue_scores"] == 0).all()

    per_n = {r["n"]: r for r in parallel(b1_for_n, [(sets, texts, n, config) for n in evaluate.N_GRID_ISSUE], jobs)}
    p_static = issue_preds(sets, per_n[0]["scores"], evaluate.N_GRID_ISSUE)
    p_n = issue_preds(sets, {n: r["scores"] for n, r in per_n.items()}, evaluate.N_GRID_ISSUE)

    # 누수 크기: 같은 설정을 학습 이슈 전체로 학습해 train에 채점한 in-fold AUROC vs OOF AUROC
    leak, scale = {}, []
    for n, r in per_n.items():
        oof_d = trajectory_auroc(train, r["scores"], n)
        in_d = trajectory_auroc(train, r["in_fold"], n)
        leak[n] = {"oof_auroc": round(oof_d["auroc"], 4), "in_fold_auroc": round(in_d["auroc"], 4),
                   "seen_issue_scores": r["seen_issue_scores"], "fold_fit_issues": r["fold_issues"]}
        # P4-D2 근거: 테스트 점수 fold 평균 vs 전체 재학습, 그리고 OOF 점수 분포
        for name, trajs in sets.items():
            versions = {"oof": r["scores"], "in_fold": r["in_fold"]} if name == "train" else {"fold_mean": r["scores"], "refit": r["refit"]}
            for scheme, s in versions.items():
                p = metrics.align(trajs, baselines.to_trajectories(trajs, s, n), n)
                d = metrics.discrimination(p["target"], p["p_fail"])
                q = np.quantile(p["p_fail"].to_numpy(), QUANTILES)
                scale.append({"N": n, "split": name, "scheme": scheme, "auroc": d["auroc"], "pr_auc": d["pr_auc"],
                              "mean": float(p["p_fail"].mean()), **{f"q{int(x * 100):02d}": float(v) for x, v in zip(QUANTILES, q)}})
    pl.DataFrame(scale).write_csv(out_dir / "b1_scale.csv", float_precision=FLOAT)
    info = {"config": config.as_dict(), "grid_size": len(B1_GRID), "leakage": leak}
    # 테스트를 fold 모델 하나로만 채점한 예측 (운영점 이전성 분석용, 저장하지 않음): {model: {N: [fold k 예측]}}
    test = sets["test"]
    fold_test = {
        "b1_static": {n: [baselines.to_trajectories(test, f, n) for f in per_n[0]["fold_test"]] for n in evaluate.N_GRID},
        "b1_n": {n: [baselines.to_trajectories(test, f, n) for f in per_n[n]["fold_test"]] for n in evaluate.N_GRID},
    }
    return p_static, p_n, config, info, fold_test


def build_oracles(sets, proposal_trajs):
    fill = 1 - float(sets["train"]["target"].mean())  # 점수를 못 주는 궤적: 학습 성공률
    loo = pl.concat([baselines.oracle_loo(t, n, fill) for t in sets.values() for n in evaluate.N_GRID_ISSUE])
    singles = {name: int(t.group_by("instance_id").len().filter(pl.col("len") == 1)["len"].sum()) for name, t in sets.items()}
    full = baselines.oracle_full_loo_auroc(proposal_trajs)  # 제안서 0.945 재현 (전체 80,036궤적)
    info = {"fill_p_fail": round(fill, 4), "loo_single_trajectory_issues_filled": singles,
            "full_data_loo_proposal": {**full, "auroc": round(full["auroc"], 4)}}
    return loo, info


# ---- 평가 ----

def fixed_cap_table(sets, costs):
    ks = range(1, FIXED_CAP_MAX_K + 1)
    tables = [metrics.fixed_cap_curve(t, costs[name], ks).with_columns(split=pl.lit(name)) for name, t in sets.items()]
    return pl.concat(tables).select("split", "K", "n_stopped", "stop_rate", "lost_success", "saved_steps", "saved_tokens")


def proposal_reproduction(proposal_trajs, all_costs):
    """제안서 고정 상한 표(전체 80,036궤적) 재현. 하네스 검사용이고 평가 결과가 아니다."""
    ks = [*EXPECTED["fixed_cap_all"], K_PROPOSAL]
    curve = metrics.fixed_cap_curve(proposal_trajs, all_costs, sorted(set(ks)))
    return {int(r["K"]): {c: r[c] for c in ("lost_success", "saved_steps", "saved_tokens")} for r in curve.iter_rows(named=True)}


def boot_task(trajs, p, stat, n_boot):
    return metrics.bootstrap(trajs, p, stat, n_boot=n_boot)


def cap_stat(ks, costs):
    def stat(trajs, _preds):
        out = {}
        for k in ks:
            out.update({f"K{k}.{c}": v for c, v in evaluate.cap_point(trajs, k, costs).items()})
        for k in ks[1:]:
            out.update({f"K{k}.d_{c}": out[f"K{k}.{c}"] - out[f"K{ks[0]}.{c}"] for c in ("lost_success", "saved_steps", "saved_tokens")})
        return out
    return stat


def operating_points(all_preds, sets, costs, k_star):
    """train(OOF)에서 고르고 테스트에 적용. 반환: 주 운영점 표 행(부트스트랩 구간은 나중에 붙임), 민감도 행, 주 운영점 {(model, N, policy): (τ, K)}."""
    train = sets["train"]
    main, sens, chosen = [], [], {}
    applied = {}
    ref_train = evaluate.cap_point(train, k_star, costs["train"])
    for model in OP_MODELS:
        p_all = all_preds[model]
        sp = {name: evaluate.split_preds(p_all, t) for name, t in sets.items()}
        for n in evaluate.N_GRID:
            for policy in evaluate.POLICIES:
                curves = evaluate.candidate_curves(train, sp["train"], n, policy, costs["train"])
                for by in evaluate.OBJECTIVES:
                    for max_loss in evaluate.MAX_LOSS_GRID:
                        op = metrics.select_from_curves(curves, n, max_loss=max_loss, by=by)
                        key = (model, n, op["tau"], op["cap"])
                        for name in evaluate.TEST_SPLITS:
                            if (key, name) not in applied:
                                applied[(key, name)] = evaluate.apply_point(sets[name], sp[name].filter(pl.col("N") == n), n, op["tau"], op["cap"], costs[name])
                        base = {"model": model, "policy": policy, "N": n, "objective": by, "max_loss": max_loss, "tau": op["tau"], "cap": op["cap"]}
                        sens.append({**base, "split": "train", **{c: op[c] for c in evaluate.POINT_COLUMNS}})
                        sens += [{**base, "split": name, **applied[(key, name)]} for name in evaluate.TEST_SPLITS]
                        if by == "saved_steps" and max_loss == evaluate.MAX_LOSS:
                            chosen[(model, n, policy)] = (op["tau"], op["cap"])
                            spread = evaluate.fold_spread(train, sp["train"], n, op["tau"], op["cap"], costs["train"])
                            tr = {c: op[c] for c in evaluate.POINT_COLUMNS}
                            main.append({"model": model, "policy": policy, "N": n, "tau": op["tau"], "cap": op["cap"], "split": "train",
                                         **tr, **{f"d_{c}": tr[c] - ref_train[c] for c in ("lost_success", "saved_steps", "saved_tokens")}, **spread})
                            main += [{"model": model, "policy": policy, "N": n, "tau": op["tau"], "cap": op["cap"], "split": name,
                                      **applied[(key, name)]} for name in evaluate.TEST_SPLITS]
    return main, sens, chosen


def cap_rows(sets, costs, k_star, cap_curve_train):
    rows, sens = [], []
    ref = {name: evaluate.cap_point(t, k_star, costs[name]) for name, t in sets.items()}
    for policy, k in (("cap", k_star), ("cap_proposal", K_PROPOSAL)):
        for name, t in sets.items():
            pt = evaluate.cap_point(t, k, costs[name])
            row = {"model": "b0_cap", "policy": policy, "N": None, "tau": None, "cap": k, "split": name, **pt}
            if name == "train":  # K* 행은 차이의 기준이라 d_*가 없다
                if k != k_star:
                    row.update({f"d_{c}": pt[c] - ref[name][c] for c in ("lost_success", "saved_steps", "saved_tokens")})
                row.update(evaluate.fold_spread(sets["train"], None, None, None, k, costs["train"]))
            rows.append(row)
    for max_loss in evaluate.MAX_LOSS_GRID:
        k = metrics.select_cap(cap_curve_train, max_loss)["K"]
        for name, t in sets.items():
            sens.append({"model": "b0_cap", "policy": "cap", "N": None, "objective": "saved_steps", "max_loss": max_loss,
                         "tau": None, "cap": k, "split": name, **evaluate.cap_point(t, k, costs[name])})
    return rows, sens


def transfer_task(model, n, policy, train, train_preds, test, test_preds, full_op, test_fold_preds):
    rows = evaluate.transfer_rows(train, train_preds, test, test_preds, n, policy, full_op, test_fold_preds=test_fold_preds)
    return [{"model": model, "policy": policy, "N": n, **r} for r in rows]


def tau_transfer(all_preds, sets, main_rows, k_star, fold_test, jobs):
    """회의 2026-11-05 §4.3: train 안 fold 사이 운영점 이전 vs train → test 이전 (evaluate.transfer_rows).

    fold_test[model][N] = 테스트를 fold k 모델 하나로 채점한 예측 목록 (OOF로 학습하는 모델만, 예: B1).
    """
    train, test = sets["train"], sets["test"]
    full = {(r["model"], r["N"], r["policy"]): r for r in main_rows if r["split"] == "train"}
    cap_full = {"tau": None, "cap": k_star, **evaluate.cap_point(train, k_star, None)}
    tasks = [("b0_cap", None, "cap", train, None, test, None, cap_full, None)]
    for m in OP_MODELS:
        for n in evaluate.N_GRID:
            for policy in evaluate.POLICIES:
                tasks.append((m, n, policy, train, evaluate.split_preds(all_preds[m], train, n), test,
                              evaluate.split_preds(all_preds[m], test, n), full[(m, n, policy)], fold_test.get(m, {}).get(n)))
    rows = [r for out in parallel(transfer_task, tasks, jobs) for r in out]
    cols = ["model", "policy", "N", *evaluate.TRANSFER_COLUMNS]
    return frame(rows, cols, {"N": pl.Int64, "cap": pl.Int64, "tau": pl.Float64})


def attach_ci(main_rows, boots):
    """주 운영점 행에 테스트 부트스트랩 구간을 붙인다. boots[(model, split, N)] = bootstrap 출력."""
    out = []
    for r in main_rows:
        r = dict(r)
        if r["split"] in evaluate.TEST_SPLITS and boots:
            if r["model"] == "b0_cap":
                b, prefix = boots[("b0_cap", r["split"], None)], f"K{r['cap']}."
            else:
                b, prefix = boots[(r["model"], r["split"], r["N"])], f"{r['policy']}."
            ci = {s[len(prefix):]: (e, lo, hi) for s, e, lo, hi in b.select("stat", "estimate", "lo", "hi").iter_rows() if s.startswith(prefix)}
            for c, (e, lo, hi) in ci.items():
                if c.startswith("d_") or c not in r:
                    r[c] = e
                r[f"{c}_lo"], r[f"{c}_hi"] = lo, hi
        out.append(r)
    return out


OP_COLUMNS = [
    "model", "policy", "N", "tau", "cap", "split", "stop_rate", "lost_success", "lost_success_lo", "lost_success_hi",
    "meets_max_loss", "saved_steps", "saved_steps_lo", "saved_steps_hi", "saved_tokens", "saved_tokens_lo", "saved_tokens_hi",
    "d_lost_success", "d_lost_success_lo", "d_lost_success_hi", "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi",
    "d_saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi",
    "fold_lost_success_min", "fold_lost_success_max", "fold_saved_steps_min", "fold_saved_steps_max",
]


def frame(rows, columns, schema_overrides=None):
    df = pl.DataFrame(rows, infer_schema_length=None, schema_overrides=schema_overrides)
    for c in columns:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    return df.select(columns)


def main():
    ap = argparse.ArgumentParser(description="Phase 4: 기준선 B0·B1, 오라클, 운영점")
    ap.add_argument("--force", action="store_true", help="저장된 예측 파일과 달라도 덮어씀")
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT, help=f"이슈 단위 부트스트랩 반복 수 (기본 {evaluate.N_BOOT})")
    ap.add_argument("--skip-bootstrap", action="store_true", help="부트스트랩 구간을 건너뜀 (개발용, 완료 기준 C7·C8 검사 안 함)")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1), help="B1 그리드·부트스트랩 병렬 프로세스 수")
    args = ap.parse_args()
    if args.n_boot <= 0:
        ap.error("--n-boot은 양수여야 한다")
    for p, script in ((io.TRAJECTORIES_PATH, "01_parse.py"), (io.STEPS_PATH, "01_parse.py"), (io.SPLITS_PATH, "02_split.py")):
        if not p.exists():
            sys.exit(f"{p}가 없다. 먼저 uv run --no-sync python code/{script} 를 실행한다.")
    t_start = time.time()
    timings = {}

    def lap(name):
        timings[name] = round(time.time() - t_start - sum(timings.values()), 1)
        print(f"[{timings[name]:>6.1f}s] {name}", flush=True)

    out_dir = io.RESULTS_DIR / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_stats = split.validate_splits(io.load_trajectories(split.INPUT_COLUMNS), io.load_splits())
    sets = evaluate.load_eval_sets()
    all_costs = evaluate.load_all_costs()
    costs = evaluate.load_costs(sets, all_costs)
    # 제안서 수치 재현 검사(C4·C6) 전용: 분할과 무관한 전체 80,036궤적 (8B·405B 포함). 평가에는 쓰지 않는다
    proposal_trajs = io.load_trajectories(metrics.TRAJ_COLUMNS)
    texts = baselines.issue_table()
    n_hint = io.load_trajectories(["instance_id", "issue_text"]).filter(
        pl.col("issue_text").str.contains(baselines.HINT, literal=True))["instance_id"].n_unique()
    lap("입력·평가 집합·맥락 길이")

    # 1. 예측
    built = {m: build_b0(sets, c) for m, c in B0_COLUMNS.items()}
    lap("B0 에러 개수 점수")
    built["b1_static"], built["b1_n"], b1_config, b1_info, b1_fold_test = build_b1(sets, texts, args.jobs, out_dir)
    lap("B1 그리드·OOF")
    built["ref_oracle_loo"], oracle_info = build_oracles(sets, proposal_trajs)
    lap("오라클")
    n_grid_of = {m: (evaluate.N_GRID if m.startswith("b0_") else evaluate.N_GRID_ISSUE) for m in MODELS}
    sets_of = {m: sets for m in MODELS}
    configs = {
        "b0_err": ({"column": "is_error", "rule": "앞 N스텝 개수"}, "rule (학습 없음)"),
        "b0_editfail": ({"column": "edit_failed", "rule": "앞 N스텝 개수"}, "rule (학습 없음)"),
        "b1_static": ({**b1_config.as_dict(), "fit_n": 0}, "5fold-oof/test-fold-mean"),
        "b1_n": ({**b1_config.as_dict(), "fit_n": "each"}, "5fold-oof/test-fold-mean"),
        "ref_oracle_loo": ({"rule": "1 - 같은 집합·같은 이슈 다른 궤적 성공률", "fill": oracle_info["fill_p_fail"]}, "reference"),
    }
    written = {m: preds.write(m, built[m], sets_of[m], n_grid_of[m], *configs[m], force=args.force) for m in MODELS}
    all_preds = {m: preds.read(m) for m in MODELS}
    lap("예측 저장·검증")

    # 2. 판별 지표
    disc = pl.concat([evaluate.discrimination_table(m, all_preds[m], sets_of[m], n_grid_of[m]) for m in MODELS])
    loo = pl.col("model") == "ref_oracle_loo"  # LOO 오라클의 이슈 내 AUROC는 정의상 0 (자기 라벨을 뺀 인공물) → 보고하지 않음
    disc_out = disc.with_columns([pl.when(loo).then(None).otherwise(pl.col(c)).alias(c) for c in ("within_pair", "within_issue_mean")])
    disc_out.write_csv(out_dir / "discrimination.csv", float_precision=FLOAT)

    # 3. 고정 상한 + K*
    caps = fixed_cap_table(sets, costs)
    caps.write_csv(out_dir / "fixed_cap.csv", float_precision=FLOAT)
    cap_train = caps.filter(pl.col("split") == "train")
    k_star = metrics.select_cap(cap_train, evaluate.MAX_LOSS)["K"]
    proposal = proposal_reproduction(proposal_trajs, all_costs)
    lap("판별 지표·고정 상한")

    # 4. 운영점
    main_rows, sens_rows, chosen = operating_points(all_preds, sets, costs, k_star)
    cap_main, cap_sens = cap_rows(sets, costs, k_star, cap_train)
    lap("운영점 선택·적용")

    # 4b. 운영점 이전성 (회의 §4.3): fold 사이 vs train → test
    transfer = tau_transfer(all_preds, sets, main_rows, k_star, b1_fold_test, args.jobs)
    transfer.write_csv(out_dir / "tau_transfer.csv", float_precision=FLOAT)
    transfer_sum = evaluate.transfer_summary(transfer)
    transfer_sum.write_csv(out_dir / "tau_transfer_summary.csv", float_precision=FLOAT)
    lap("운영점 이전성")

    # 5. 비용 곡선 (그림용, 상한 없는 단독 정책)
    curve_dir = out_dir / "cost_curves"
    if curve_dir.exists():
        shutil.rmtree(curve_dir)
    curve_dir.mkdir()
    for m in MODELS:
        for (name, n), c in evaluate.model_curves(all_preds[m], sets_of[m], costs, n_grid_of[m]).items():
            evaluate.thin_curve(c).write_csv(curve_dir / f"{m}_{name}_N{n}.csv", float_precision=FLOAT)
    lap("비용 곡선")

    # 6. 부트스트랩 (테스트 분할, 이슈 단위). 분할마다 같은 시드라 모델 사이에서도 같은 재표본을 쓴다
    boots = {}
    if not args.skip_bootstrap:
        tasks, keys = [], []
        for name in evaluate.TEST_SPLITS:
            t = sets[name]
            for m in MODELS:
                if name not in sets_of[m]:
                    continue
                for n in n_grid_of[m]:
                    points = {pol: chosen[(m, n, pol)] for pol in evaluate.POLICIES if (m, n, pol) in chosen}
                    caps_used = [cap for _, cap in points.values()]
                    c = evaluate.compressed_costs(costs[name], n, caps_used, k_star) if points else None
                    stat = evaluate.bootstrap_stat(n, points, k_star if points else None, c, discrimination=True)
                    tasks.append((t, evaluate.split_preds(all_preds[m], t, n), stat, args.n_boot))
                    keys.append((m, name, n))
            c = evaluate.compressed_costs(costs[name], None, [K_PROPOSAL], k_star)
            empty = pl.DataFrame(schema=preds.SCHEMA)
            ks = list(dict.fromkeys([k_star, K_PROPOSAL]))  # K*가 제안서 K와 같으면 차이(0)를 따로 만들지 않는다
            tasks.append((t, empty, cap_stat(ks, c), args.n_boot))
            keys.append(("b0_cap", name, None))
        boots = dict(zip(keys, parallel(boot_task, tasks, args.jobs)))
        ci_rows = []
        for (m, name, n), b in boots.items():
            if m == "b0_cap":
                continue
            keep = ["auroc", "pr_auc"] + ([] if m == "ref_oracle_loo" else ["within_pair", "within_issue_mean"])
            ci_rows.append(b.filter(pl.col("stat").is_in(keep)).with_columns(model=pl.lit(m), split=pl.lit(name), N=pl.lit(n)))
        pl.concat(ci_rows).select("model", "split", "N", "stat", "estimate", "lo", "hi", "n_boot") \
            .write_csv(out_dir / "discrimination_ci.csv", float_precision=FLOAT)
        lap(f"부트스트랩 ({len(tasks)}개 × {args.n_boot}회)")

    ops = attach_ci(cap_main + main_rows, boots)
    for r in ops:
        r["meets_max_loss"] = bool(r["lost_success"] <= evaluate.MAX_LOSS + 1e-12)
    ops_df = frame(ops, OP_COLUMNS, {"N": pl.Int64, "cap": pl.Int64, "tau": pl.Float64})
    ops_df.write_csv(out_dir / "operating_points.csv", float_precision=FLOAT)
    sens_cols = ["model", "policy", "N", "objective", "max_loss", "tau", "cap", "split", *evaluate.POINT_COLUMNS]
    frame(cap_sens + sens_rows, sens_cols, {"N": pl.Int64, "cap": pl.Int64, "tau": pl.Float64}) \
        .write_csv(out_dir / "operating_points_sensitivity.csv", float_precision=FLOAT)

    check(args, out_dir, split_stats, texts, n_hint, written, disc_out, caps, k_star, proposal, b1_info, oracle_info, ops_df,
          transfer, transfer_sum, boots, timings)


def check(args, out_dir, split_stats, texts, n_hint, written, disc, caps, k_star, proposal, b1_info, oracle_info, ops_df,
          transfer, transfer_sum, boots, timings):
    def pick(split_name, k, col):
        return float(caps.filter((pl.col("split") == split_name) & (pl.col("K") == k))[col][0])

    b1 = disc.filter(pl.col("model").is_in(["b1_static", "b1_n"]))
    within_dev = float(max((b1["within_pair"] - 0.5).abs().max(), (b1["within_issue_mean"] - 0.5).abs().max()))
    head = disc.filter(pl.col("split") != "train_oof").filter(
        (pl.col("N").is_in([0, 10])) & pl.col("model").is_in(["b0_err", "b1_static", "b1_n", "ref_oracle_loo"]))
    s = {
        "splits": split_stats["split_trajectories"],
        "issues_after_strip": texts.height,
        "issues_with_hint": n_hint,
        "preds": {m: {"rows": w["rows"], "written": w["written"], "coverage": w["coverage"]} for m, w in written.items()},
        "b1": b1_info,
        "b1_within_issue_max_abs_dev_from_half": within_dev,
        "oracle": oracle_info,
        "k_star": k_star,
        "fixed_cap": {name: {k: {c: round(pick(name, k, c), 4) for c in ("lost_success", "saved_steps", "saved_tokens")}
                             for k in (k_star, K_PROPOSAL)} for name in split.SPLITS},
        "proposal_reproduction_all_80036": {k: {c: round(v, 4) for c, v in r.items()} for k, r in proposal.items()},
        "discrimination_test_N0_N10": head.select("model", "split", "N", "auroc", "pr_auc", "within_pair").to_dicts(),
        "operating_points_N10": ops_df.filter((pl.col("N") == 10) | (pl.col("model") == "b0_cap")).select(
            "model", "policy", "N", "tau", "cap", "split", "lost_success", "saved_steps", "saved_tokens", "d_saved_steps",
            "d_saved_steps_lo", "d_saved_steps_hi").to_dicts(),
        "tau_transfer_N10": transfer_sum.filter((pl.col("N") == 10) | pl.col("N").is_null()).to_dicts(),
        "bootstrap": None if args.skip_bootstrap else {"n_boot": args.n_boot, "tasks": len(boots), "seed": io.SEED},
    }

    crit = {}
    crit["C1_preds_cover_running"] = all(w["coverage"] for w in written.values())  # preds.write가 검사 실패 시 이미 멈춘다
    crit["C2_b1_within_issue_half"] = within_dev < 1e-12
    crit["C3_oof_no_seen_issue_and_one_text"] = (
        all(v["seen_issue_scores"] == 0 for v in b1_info["leakage"].values()) and texts.height == EXPECTED["issues_after_strip"]
        and n_hint == EXPECTED["issues_with_hint"])
    crit["C3b_in_fold_much_higher_than_oof"] = all(v["in_fold_auroc"] > v["oof_auroc"] + 0.05 for v in b1_info["leakage"].values())
    c4 = all(abs(proposal[k]["lost_success"] - lo) <= 5e-4 + 1e-9 and abs(proposal[k]["saved_steps"] - sv) <= 5e-4 + 1e-9
             for k, (lo, sv) in EXPECTED["fixed_cap_all"].items())
    c4 &= abs(proposal[K_PROPOSAL]["saved_tokens"] - EXPECTED["fixed_cap_all_k35_tokens"]) <= 5e-4 + 1e-9
    # 분할별 고정 상한 표 = 하네스 (N=K에서 진행 중 궤적 전부 p_fail=1) — K_PROPOSAL에서 test로 확인
    t = evaluate.load_eval_sets()["test"]
    h = metrics.cost_curve(t, metrics.running(t, K_PROPOSAL).select("traj_id", N=pl.lit(K_PROPOSAL), p_fail=pl.lit(1.0)), K_PROPOSAL).row(-1, named=True)
    c4 &= all(abs(pick("test", K_PROPOSAL, c) - h[c]) <= 1e-12 for c in ("lost_success", "saved_steps"))
    crit["C4_fixed_cap_matches_phase3"] = bool(c4)
    crit["C5_k_star"] = k_star == EXPECTED["k_star"]
    full = oracle_info["full_data_loo_proposal"]
    crit["C6_oracle_full_loo"] = round(full["auroc"], 3) == EXPECTED["oracle_full_loo_auroc"] and full["n_traj"] == EXPECTED["oracle_full_loo_n_traj"]
    models_in_disc = set(disc["model"].unique())
    crit["C7_discrimination_table"] = models_in_disc == set(MODELS) and (out_dir / "discrimination_ci.csv").exists() and not args.skip_bootstrap
    test_ops = ops_df.filter(pl.col("split").is_in(evaluate.TEST_SPLITS))
    crit["C8_operating_points_with_ci"] = (
        not args.skip_bootstrap and set(test_ops["model"].unique()) == {*OP_MODELS, "b0_cap"}
        and set(test_ops["policy"].unique()) == {"single", "combined", "cap", "cap_proposal"}
        and test_ops.filter(pl.col("model") != "b0_cap")["d_saved_steps_lo"].null_count() == 0)
    n_folds = split.N_FOLDS
    n_points = 1 + len(OP_MODELS) * len(evaluate.N_GRID) * len(evaluate.POLICIES)  # b0_cap + 모델 × N × 정책
    n_fold_model = 2 * len(evaluate.N_GRID) * len(evaluate.POLICIES)  # B1 두 종류만 fold 모델별 테스트 점수가 있다
    expected_rows = (2 * n_folds + 1) * n_points + n_folds * n_fold_model
    crit["C10_tau_transfer_table"] = (
        transfer.height == expected_rows
        and set(transfer["target"].unique()) == {"test", "test_fold_model", *(f"fold{k}" for k in range(n_folds))}
        and transfer.filter(pl.col("source") == "train").height == n_points
        and set(transfer.filter(pl.col("target") == "test_fold_model")["model"].unique()) == {"b1_static", "b1_n"}
        and transfer.select("dst_stop_rate", "dst_lost_success", "src_lost_success").null_count().sum_horizontal().item() == 0)
    s["criteria"] = crit
    s["criteria_note"] = "C9(pytest 전체 통과, 재실행 시 results/ 변화 없음)는 스크립트 밖에서 확인한다"
    (out_dir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2, default=str) + "\n")

    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=200, float_precision=4):
        print(disc.filter(pl.col("N").is_in([0, 10])).select("model", "split", "N", "n_running", "base_rate", "auroc", "pr_auc", "within_pair"))
        print(ops_df.filter((pl.col("N") == 10) | (pl.col("model") == "b0_cap")).select(
            "model", "policy", "N", "tau", "cap", "split", "lost_success", "saved_steps", "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi"))
    print(json.dumps({"k_star": k_star, "b1": b1_info["config"], "criteria": crit, "timings": timings}, ensure_ascii=False, indent=2))

    if args.skip_bootstrap:
        failed = [k for k, v in crit.items() if not v and k not in ("C7_discrimination_table", "C8_operating_points_with_ci")]
        assert not failed, failed
        print("\n부트스트랩을 건너뛰어 C7·C8은 검사하지 않음. 나머지 완료 기준 통과")
        return
    failed = [k for k, v in crit.items() if not v]
    assert not failed, failed
    print("\n완료 기준 통과")


if __name__ == "__main__":
    main()
