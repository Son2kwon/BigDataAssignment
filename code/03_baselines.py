"""Phase 4: 기준선 B0·B1, 참고용 오라클, 운영점 선택(3.5) 적용

실행: uv run --no-sync python code/03_baselines.py [--force] [--n-boot 1000] [--skip-bootstrap] [--jobs 16]

입력: data/processed/trajectories.parquet, steps.parquet (01_parse.py), splits.parquet (02_split.py)
평가 집합은 70B 궤적뿐이다 (train, test). 8B·405B 궤적은 제안서 수치 재현 검사(C4·C6)에만 읽는다.
- data/processed/preds/<model>.parquet  traj_id, N, p_fail (train OOF + test)
    b0_err, b0_editfail       앞 N스텝 is_error / edit_failed 개수 (학습 없음)
    b1_static, b1_n           이슈 본문 TF-IDF + LR. static = N=0 모델 하나, n = N마다 진행 중 궤적으로 재학습
    ref_oracle_loo            같은 집합·같은 이슈 다른 궤적 성공률 (참고 천장, 피처 사용 금지)
- data/processed/preds/<model>.test_folds.parquet  B1의 fold 모델별 test 점수 (traj_id, N, fold, p_fail). 운영점 평가(P4-D6)와
    M1 입력(P4-D7)용. 주 파일의 test 행은 5개 fold 평균 점수(판별 지표용, P4-D2)
- data/processed/context_costs.parquet  스텝별 맥락 길이 캐시
- results/phase4/                       판별 표, 고정 상한 표, 운영점 표, 운영점 이전성 표, 비용 곡선, B1 그리드, 요약

예측 파일이 이미 있으면 다시 계산한 결과와 같은지 확인한다. 다르면 멈추고, --force면 덮어쓴다.
끝에 완료 기준(PLAN_4.md 4.7)을 검사하고 다르면 assert로 멈춘다.

손실 기준 (회의 2026-11-05 §4.1): 운영점·K*는 train 손실 <= metrics.SELECT_MAX_LOSS(3.5%)로 고르고, 테스트 충족은
손실 95% 구간 상한 <= metrics.JUDGE_MAX_LOSS(5%)로 판정한다 (operating_points.csv의 meets_judge_ci_hi).
OOF 모델(B1)의 test 운영점은 fold 모델별 점수에 같은 τ·K를 적용한 5개 결과의 평균이다 (P4-D6, *_fold_min/max 컬럼).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl

from afp import baselines, evaluate, io, metrics, oof, preds, split

MODELS = ("b0_err", "b0_editfail", "b1_static", "b1_n", "ref_oracle_loo")
OP_MODELS = ("b0_err", "b0_editfail", "b1_static", "b1_n")  # 운영점 표에 넣는 모델 (오라클 제외)
FOLD_MODELS = ("b1_static", "b1_n")  # OOF로 학습해 fold 모델별 test 점수가 있는 모델 (P4-D6·P4-D7)
B0_COLUMNS = {"b0_err": "is_error", "b0_editfail": "edit_failed"}
B1_GRID = tuple(
    baselines.B1Config(analyzer=a, C=c, weighting=w)
    for a in ("word", "char_wb") for c in (0.03, 0.1, 0.3, 1.0, 3.0) for w in ("traj", "issue")
)
B1_TIE = 0.002  # OOF AUROC 차이가 이보다 작으면 더 단순한 설정 (word, 작은 C, traj 가중)
K_PROPOSAL = evaluate.K_PROPOSAL  # 제안서의 고정 상한 (전체 데이터에서 고른 값, 참고 행)
FIXED_CAP_MAX_K = 80
CURVE_N = (0, 10)  # b1_scale 분위수 표의 N
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
SCORE_DIST_N = (0, *evaluate.N_GRID)  # b1_score_dist 표의 N
FLOAT = 6

EXPECTED = {
    "split_trajectories": {"train": 62_311, "test": 12_481},
    "issues_after_strip": 3_591,
    "issues_with_hint": 13,
    # train에서 손실 <= metrics.SELECT_MAX_LOSS(3.5%)인 최소 K (70B 무작위 분할, K=39는 train 손실 3.72%).
    # 선택 기준이 5%이던 때는 35 (예전 분할에서는 34)
    "k_star": 40,
    # 제안서 재현 (분할과 무관한 전체 80,036궤적, 8B·405B 포함): 하네스 검사용
    "oracle_full_loo_auroc": 0.945,
    "oracle_full_loo_n_traj": 80_026,
    # Phase 3 결과 (전체 데이터 고정 상한): K → (손실, 절약 스텝). 반올림 자릿수까지 비교
    "fixed_cap_all": {30: (0.070, 0.294), 35: (0.049, 0.246), 40: (0.037, 0.208), 50: (0.022, 0.158)},
    "fixed_cap_all_k35_tokens": 0.422,
}


parallel = evaluate.parallel  # 운영점 루프와 같이 afp/evaluate.py로 옮김 (Phase 5.0)


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
    # 테스트를 fold 모델 하나로만 채점한 예측 (traj_id, N, fold, p_fail). 저장해서 운영점 평가(P4-D6)와 M1 입력(P4-D7)에 쓴다
    test = sets["test"]
    source_n = {"b1_static": lambda n: 0, "b1_n": lambda n: n}
    fold_test = {
        m: pl.concat([baselines.to_trajectories(test, f, n).with_columns(fold=pl.lit(k, dtype=pl.Int8))
                      for n in evaluate.N_GRID_ISSUE for k, f in enumerate(per_n[src(n)]["fold_test"])])
        for m, src in source_n.items()
    }
    return p_static, p_n, config, info, fold_test


def fold_lists(fold_test: pl.DataFrame, n: int) -> list[pl.DataFrame]:
    """fold별 점수 (traj_id, N, fold, p_fail) → 시점 N의 fold k 예측 목록 (운영점 이전성 분석 입력)."""
    f = fold_test.filter(pl.col("N") == n)
    return [f.filter(pl.col("fold") == k).select("traj_id", "N", "p_fail").sort("traj_id") for k in sorted(f["fold"].unique().to_list())]


def score_dist(all_preds, fold_test, sets):
    """5.0.6 B1 점수 분포: N별 train OOF, test fold 0..4, test 평균 점수의 평균·표준편차(ddof=0)·분위수. 궤적 단위."""
    rows = []
    for m in FOLD_MODELS:
        for n in SCORE_DIST_N:
            srcs = {"train_oof": evaluate.split_preds(all_preds[m], sets["train"], n),
                    **{f"test_fold{k}": p for k, p in enumerate(fold_lists(fold_test[m], n))},
                    "test_mean": evaluate.split_preds(all_preds[m], sets["test"], n)}
            for src, p in srcs.items():
                v = p["p_fail"].to_numpy()
                q = np.quantile(v, QUANTILES)
                rows.append({"model": m, "N": n, "source": src, "n": len(v), "mean": float(v.mean()), "std": float(v.std()),
                             **{f"q{int(x * 100):02d}": float(y) for x, y in zip(QUANTILES, q)}})
    return pl.DataFrame(rows)


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


def transfer_task(model, n, policy, train, train_preds, test, test_preds, full_op, test_fold_preds):
    rows = evaluate.transfer_rows(train, train_preds, test, test_preds, n, policy, full_op, test_fold_preds=test_fold_preds)
    return [{"model": model, "policy": policy, "N": n, **r} for r in rows]


def tau_transfer(all_preds, sets, main_rows, k_star, fold_test, jobs):
    """회의 2026-11-05 §4.3: train 안 fold 사이 운영점 이전 vs train → test 이전 (evaluate.transfer_rows).

    fold_test[model] = 테스트를 fold k 모델 하나로 채점한 예측 (traj_id, N, fold, p_fail. OOF로 학습하는 모델만, 예: B1).
    """
    train, test = sets["train"], sets["test"]
    full = {(r["model"], r["N"], r["policy"]): r for r in main_rows if r["split"] == "train"}
    cap_full = {"tau": None, "cap": k_star, **evaluate.cap_point(train, k_star, None)}
    tasks = [("b0_cap", None, "cap", train, None, test, None, cap_full, None)]
    for m in OP_MODELS:
        for n in evaluate.N_GRID:
            for policy in evaluate.POLICIES:
                tasks.append((m, n, policy, train, evaluate.split_preds(all_preds[m], train, n), test,
                              evaluate.split_preds(all_preds[m], test, n), full[(m, n, policy)],
                              fold_lists(fold_test[m], n) if m in fold_test else None))
    rows = [r for out in parallel(transfer_task, tasks, jobs) for r in out]
    cols = ["model", "policy", "N", *evaluate.TRANSFER_COLUMNS]
    return evaluate.frame(rows, cols, evaluate.OP_SCHEMA)


def main():
    ap = argparse.ArgumentParser(description="Phase 4: 기준선 B0·B1, 오라클, 운영점")
    ap.add_argument("--force", action="store_true", help="저장된 예측 파일과 달라도 덮어씀")
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT, help=f"이슈 단위 부트스트랩 반복 수 (기본 {evaluate.N_BOOT})")
    ap.add_argument("--skip-bootstrap", action="store_true", help="부트스트랩 구간을 건너뜀 (개발용, 완료 기준 C7·C8 검사 안 함)")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1), help="B1 그리드·부트스트랩 병렬 프로세스 수")
    args = ap.parse_args()
    if args.n_boot <= 0:
        ap.error("--n-boot은 양수여야 한다")
    assert metrics.SELECT_MAX_LOSS in evaluate.MAX_LOSS_GRID and metrics.JUDGE_MAX_LOSS in evaluate.MAX_LOSS_GRID
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
    written_folds = {m: preds.write_test_folds(m, b1_fold_test[m], sets["test"], n_grid_of[m], split.N_FOLDS, force=args.force)
                     for m in FOLD_MODELS}
    all_preds = {m: preds.read(m) for m in MODELS}
    fold_test = {m: preds.read_test_folds(m) for m in FOLD_MODELS}
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
    k_star = metrics.select_cap(cap_train, metrics.SELECT_MAX_LOSS)["K"]
    proposal = proposal_reproduction(proposal_trajs, all_costs)
    lap("판별 지표·고정 상한")

    # 4. 운영점
    model_preds = {m: (all_preds[m], fold_test.get(m)) for m in MODELS}
    main_rows, sens_rows, chosen = evaluate.operating_points({m: model_preds[m] for m in OP_MODELS}, sets, costs, k_star)
    cap_main, cap_sens = evaluate.cap_rows(sets, costs, k_star, cap_train, K_PROPOSAL)
    lap("운영점 선택·적용")

    # 4b. 운영점 이전성 (회의 §4.3): fold 사이 vs train → test
    transfer = tau_transfer(all_preds, sets, main_rows, k_star, fold_test, args.jobs)
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
        for (name, n), c in evaluate.model_curves(all_preds[m], sets_of[m], costs, n_grid_of[m], fold_test.get(m)).items():
            evaluate.thin_curve(c).write_csv(curve_dir / f"{m}_{name}_N{n}.csv", float_precision=FLOAT)
    lap("비용 곡선")

    # 6. 부트스트랩 (테스트 분할, 이슈 단위). 분할마다 같은 시드라 모델 사이에서도 같은 재표본을 쓴다
    boots = {}
    if not args.skip_bootstrap:
        tasks, keys = evaluate.bootstrap_tasks(model_preds, sets, costs, chosen, k_star, n_grid_of, args.n_boot)
        cap_tasks, cap_keys = evaluate.cap_bootstrap_tasks(sets, costs, k_star, args.n_boot, K_PROPOSAL)
        boots = dict(zip(keys + cap_keys, parallel(evaluate.boot_task, tasks + cap_tasks, args.jobs)))
        evaluate.discrimination_ci(boots, no_within=("ref_oracle_loo",)).write_csv(out_dir / "discrimination_ci.csv", float_precision=FLOAT)
        lap(f"부트스트랩 ({len(tasks) + len(cap_tasks)}개 × {args.n_boot}회)")

    ops_df = evaluate.operating_table(cap_main + main_rows, boots)
    ops_df.write_csv(out_dir / "operating_points.csv", float_precision=FLOAT)
    dist = score_dist(all_preds, fold_test, sets)
    dist.write_csv(out_dir / "b1_score_dist.csv", float_precision=FLOAT)
    evaluate.sensitivity_table(cap_sens + sens_rows).write_csv(out_dir / "operating_points_sensitivity.csv", float_precision=FLOAT)

    check(args, out_dir, split_stats, texts, n_hint, written, disc_out, caps, k_star, proposal, b1_info, oracle_info, ops_df,
          transfer, transfer_sum, boots, timings, written_folds, dist)


def check(args, out_dir, split_stats, texts, n_hint, written, disc, caps, k_star, proposal, b1_info, oracle_info, ops_df,
          transfer, transfer_sum, boots, timings, written_folds, dist):
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
        "preds_test_folds": {m: {"rows": w["rows"], "folds": w["folds"], "max_abs_mean_diff": w["max_abs_mean_diff"]} for m, w in written_folds.items()},
        "b1": b1_info,
        "b1_within_issue_max_abs_dev_from_half": within_dev,
        "oracle": oracle_info,
        "k_star": k_star,
        "fixed_cap": {name: {k: {c: round(pick(name, k, c), 4) for c in ("lost_success", "saved_steps", "saved_tokens")}
                             for k in (k_star, K_PROPOSAL)} for name in split.SPLITS},
        "proposal_reproduction_all_80036": {k: {c: round(v, 4) for c, v in r.items()} for k, r in proposal.items()},
        "discrimination_test_N0_N10": head.select("model", "split", "N", "auroc", "pr_auc", "within_pair").to_dicts(),
        "loss_criteria": {
            "select_max_loss": metrics.SELECT_MAX_LOSS, "select_rule": "train(OOF) 점추정 손실 <= select_max_loss로 τ·K·K*를 고른다",
            "judge_max_loss": metrics.JUDGE_MAX_LOSS, "judge_rule": "test 손실의 이슈 단위 부트스트랩 95% 구간 상한(lost_success_hi) <= judge_max_loss",
        },
        "operating_points_N10": ops_df.filter((pl.col("N") == 10) | (pl.col("model") == "b0_cap")).select(
            "model", "policy", "N", "tau", "cap", "split", "lost_success", "lost_success_lo", "lost_success_hi",
            *evaluate.LOSS_FLAG_COLUMNS, "saved_steps", "saved_tokens", "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi",
            "lost_success_fold_min", "lost_success_fold_max", "saved_steps_fold_min", "saved_steps_fold_max").to_dicts(),
        "b1_score_dist_N10": dist.filter(pl.col("N") == 10).select("model", "source", "std", "q05", "q50", "q95").to_dicts(),
        # 판정 기준 충족 + K* 대비 절약 스텝 차이 구간 하한 > 0 (N=3~20, 모든 모델·정책)
        "judged_and_saves_more_than_k_star": [] if args.skip_bootstrap else evaluate.judged_and_saves_more(ops_df).to_dicts(),
        # 참고값: 테스트 손실 95% 구간 폭 (선택 기준 3.5%의 근거였던 예전 분할 ±1.4%p와 비교용, 기준을 바꾸지 않는다)
        "test_loss_ci_width": None if args.skip_bootstrap else evaluate.loss_ci_width(ops_df, k_star),
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
    # 손실 기준 표시: train에서 고른 행(제안서 K 참고 행 제외)은 모두 선택 기준 안, 테스트 행은 모두 판정 값이 있다
    train_ops = ops_df.filter(pl.col("split") == "train")
    crit["C11_loss_criteria_flags"] = bool(
        train_ops.filter(pl.col("policy") != "cap_proposal")["meets_select_point"].all()
        and train_ops["meets_judge_ci_hi"].null_count() == train_ops.height
        and test_ops["meets_select_point"].null_count() == test_ops.height
        and (args.skip_bootstrap or test_ops["meets_judge_ci_hi"].null_count() == 0))
    # P4-D6·P4-D7: fold별 test 점수 파일, fold 평균 = 주 파일 test 점수, OOF 모델 test 운영점에만 fold 범위가 있고 평균이 범위 안
    crit["C12_test_fold_scores"] = all(
        w["folds"] == list(range(split.N_FOLDS)) and w["max_abs_mean_diff"] is not None and w["max_abs_mean_diff"] <= preds.FOLD_MEAN_TOL
        for w in written_folds.values()) and set(written_folds) == set(FOLD_MODELS)
    fold_ops = test_ops.filter(pl.col("model").is_in(FOLD_MODELS))
    other_ops = ops_df.filter(~(pl.col("model").is_in(FOLD_MODELS) & pl.col("split").is_in(evaluate.TEST_SPLITS)))
    crit["C13_foldwise_operating_points"] = bool(
        fold_ops.height == len(FOLD_MODELS) * len(evaluate.N_GRID) * len(evaluate.POLICIES)
        and all(fold_ops[f"{c}_fold_min"].null_count() == 0
                and ((fold_ops[f"{c}_fold_min"] <= fold_ops[c] + 1e-12) & (fold_ops[c] <= fold_ops[f"{c}_fold_max"] + 1e-12)).all()
                for c in evaluate.POINT_COLUMNS)
        and other_ops.select([f"{c}_fold_{e}" for c in evaluate.POINT_COLUMNS for e in ("min", "max")]).null_count().sum_horizontal().item()
        == other_ops.height * 2 * len(evaluate.POINT_COLUMNS)
        and dist.height == len(FOLD_MODELS) * len(SCORE_DIST_N) * (split.N_FOLDS + 2))
    s["criteria"] = crit
    s["criteria_note"] = "C9(pytest 전체 통과, 재실행 시 results/ 변화 없음)는 스크립트 밖에서 확인한다"
    io.write_summary(out_dir / "summary.json", s)

    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=200, float_precision=4):
        print(disc.filter(pl.col("N").is_in([0, 10])).select("model", "split", "N", "n_running", "base_rate", "auroc", "pr_auc", "within_pair"))
        print(ops_df.filter((pl.col("N") == 10) | (pl.col("model") == "b0_cap")).select(
            "model", "policy", "N", "tau", "cap", "split", "lost_success", "lost_success_hi", "meets_judge_ci_hi", "saved_steps",
            "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi"))
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
