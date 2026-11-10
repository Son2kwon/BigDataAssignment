"""Phase 8.1 보조 실험: 70B로 학습한 M1을 다른 모델(Llama 8B·405B) 궤적에 그대로 적용한다 (docs/phase8.md 8.1 사전 등록).

실행: uv run --no-sync python code/07_m1_cross.py [--n-boot 1000] [--jobs 16] [--skip-bootstrap] [--smoke]

- 모델: m1_xgb, m1_lr (B1을 입력으로 쓰지 않는 M1). B1 포함판은 8B·405B용 B1 fold 점수가 없어서 뺀다
- 학습: Phase 5 fold 모델을 같은 코드(04_m1.run_oof)·같은 설정(저장된 예측 파일 메타데이터)으로 다시 만든다. fold 모델은 저장하지
  않았기 때문이다. 다시 만든 모델의 train OOF·test fold 점수가 저장된 예측과 같아야 진행한다 (C3, 같은 모델이라는 확인)
- 고르는 것 없음: τ·K는 저장된 train OOF 예측에서 04_m1.py와 같은 함수(evaluate.candidate_curves·metrics.select_from_curves)로
  다시 고른 값이고, K*는 70B train에서 고른 40이다. 8B·405B 집합에서는 아무것도 고르지 않는다
- 평가: 집합마다 판별 지표(fold 평균 점수, P4-D2)와 운영점(fold 모델별 채점 평균, P4-D6), 고정 상한 K*·K=35 행, 이슈 단위
  부트스트랩 95% 구간 (afp/evaluate.py 공용 함수). 같은 방식으로 70B test(저장된 Linux 예측)도 나란히 계산한다 (참고)
- 집합 (afp/cross.py): x8b_unseen·x405b_unseen = 70B test 이슈의 궤적 (주 결과), x8b_all·x405b_all = 전체 (참고, 70B train 이슈 포함)
- --smoke: 집합마다 이슈 일부·N=5·10·부트스트랩 20회로 끝까지 돌려 data/processed/phase8_smoke/에 쓴다. 지표는 출력하지 않는다

입력: data/processed/trajectories.parquet, steps.parquet, splits.parquet, features/m1_n.parquet (04_m1.py),
      preds/m1_xgb·m1_lr(.test_folds).parquet (04_m1.py), context_costs.parquet
출력: results/phase8/ (sets.csv, discrimination.csv, operating_points.csv, n_summary.csv, summary.json),
      data/processed/phase8/m1_cross_preds.parquet (8B·405B fold 모델별 점수, git 밖)
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from afp import cross, evaluate, features, io, metrics, preds, split

m1 = importlib.import_module("04_m1")  # 주 실험의 설계 행렬·학습기·OOF 경로를 그대로 쓴다

MODELS = ("m1_xgb", "m1_lr")
REF_SET = "test70b"  # 70B test (참고, 저장된 예측)
OUT_DIR = io.RESULTS_DIR / "phase8"
DATA_DIR = io.PROCESSED_DIR / "phase8"
SMOKE_DIR = io.PROCESSED_DIR / "phase8_smoke"
SMOKE_ISSUES = 25  # --smoke: 집합마다 앞쪽 이슈 수
SMOKE_N = (5, 10)
SMOKE_BOOT = 20
REPRO_TOL = 1e-9  # 다시 만든 fold 모델 점수와 저장된 예측의 허용 차이 (같은 머신·같은 라이브러리면 0)
FLOAT = 6
SUMMARY_COLUMNS = ["split", *evaluate.N_SUMMARY_COLUMNS]


# ---- 학습 (fold 모델 다시 만들기 + 다른 모델 궤적 채점) ----

def refit_task(model, n, kind, cfg, X, y, inst, fold, X_test, test_inst, X_cross):
    r = m1.run_oof(kind, cfg, X, y, inst, fold, X_test, test_inst, keep_models=True)
    cross_folds = np.stack([m1.predict_model(mod, X_cross) for mod in r["models"]]).astype(np.float64)
    return {"model": model, "N": n, "oof": r["oof"], "test_folds": r["test_folds"], "cross_folds": cross_folds}


def cross_matrix(xfeats, xt, n, cols):
    """시점 N에 진행 중인 다른 모델 궤적 (traj_id 정렬)과 M1 피처 행렬."""
    alive = metrics.running(xt, n).select("traj_id").sort("traj_id")
    f = alive.join(xfeats.filter(pl.col("N") == n), on="traj_id", how="left")
    if f.select(features.FEATURES).null_count().sum_horizontal().item():
        raise AssertionError(f"N={n}: 피처가 없는 다른 모델 궤적")
    return alive["traj_id"], f.select(cols).to_numpy()


def stored_scores(model, n, design):
    """저장된 예측에서 다시 만든 모델과 같은 순서의 train OOF 점수와 test fold별 점수."""
    p = preds.read(model).filter(pl.col("N") == n)
    tr = design["train"].select("traj_id").join(p, on="traj_id", how="left")["p_fail"].to_numpy()
    fp = preds.read_test_folds(model).filter(pl.col("N") == n)
    te = np.stack([design["test"].select("traj_id").join(fp.filter(pl.col("fold") == k), on="traj_id", how="left")["p_fail"].to_numpy()
                   for k in range(split.N_FOLDS)])
    return tr, te


def config_of(model):
    """저장된 예측 파일(04_m1.py)의 학습 설정: 학습기, 피처 목록, N별 고른 하이퍼파라미터."""
    c = preds.metadata(model)["config"]
    kind, use_b1 = m1.MODEL_SPECS[model]
    if c["learner"] != kind or use_b1 or c["features"] != list(features.FEATURES):
        raise AssertionError(f"{model}: 저장된 예측의 설정이 지금 코드와 다르다 ({c['learner']}, 피처 {len(c['features'])}개)")
    return kind, {int(n): cfg for n, cfg in c["selected"].items()}


# ---- 운영점 (train OOF에서만 고른다) ----

def choose_points(model, sets, costs, n_grid):
    """04_m1.py(evaluate.operating_points)와 같은 선택: 저장된 train OOF 곡선, 선택 기준 3.5%, 목적 절약 스텝."""
    train_p = evaluate.split_preds(preds.read(model), sets["train"])
    out = {}
    for n in n_grid:
        for policy in evaluate.POLICIES:
            curves = evaluate.candidate_curves(sets["train"], train_p, n, policy, costs["train"])
            out[(n, policy)] = metrics.select_from_curves(curves, n, max_loss=metrics.SELECT_MAX_LOSS, by="saved_steps")
    return out


def phase5_point_match(model, chosen):
    """참고: 커밋된 results/phase5/operating_points.csv(Mac 실행, P6-D10)의 train 운영점과 (τ, K)가 같은 개수."""
    f = io.RESULTS_DIR / "phase5" / "operating_points.csv"
    p5 = pl.read_csv(f, schema_overrides=evaluate.OP_SCHEMA).filter((pl.col("model") == model) & (pl.col("split") == "train"))
    same = 0
    for (n, policy), op in chosen.items():
        r = p5.filter((pl.col("N") == n) & (pl.col("policy") == policy))
        if r.height == 1:
            tau, cap = r["tau"][0], r["cap"][0]
            same += bool(np.isclose(tau, op["tau"], rtol=0, atol=1e-6) and cap == op["cap"])
    return {"same": same, "total": len(chosen)}


# ---- 실행 ----

def main():
    ap = argparse.ArgumentParser(description="Phase 8.1: 70B M1을 8B·405B 궤적에 적용")
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    ap.add_argument("--skip-bootstrap", action="store_true", help="부트스트랩 구간을 건너뜀 (완료 기준 일부 검사 안 함)")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--smoke", action="store_true", help=f"이슈 {SMOKE_ISSUES}개·N {SMOKE_N}·부트스트랩 {SMOKE_BOOT}회. 지표 출력 없음")
    args = ap.parse_args()
    need = [io.TRAJECTORIES_PATH, io.STEPS_PATH, io.SPLITS_PATH, m1.FEATURES_PATH, *(preds.path(m) for m in MODELS),
            *(preds.fold_path(m) for m in MODELS)]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        sys.exit(f"입력이 없다: {missing}. 01_parse.py → 02_split.py → 03_baselines.py → 04_m1.py 를 먼저 실행한다.")
    n_grid = SMOKE_N if args.smoke else evaluate.N_GRID
    n_boot = SMOKE_BOOT if args.smoke else args.n_boot
    out_dir = SMOKE_DIR / "results" if args.smoke else OUT_DIR
    data_dir = SMOKE_DIR if args.smoke else DATA_DIR
    t_start = time.time()
    timings = {}

    def lap(name):
        timings[name] = round(time.time() - t_start - sum(timings.values()), 1)
        print(f"[{timings[name]:>6.1f}s] {name}", flush=True)

    print(f"모드: {'smoke' if args.smoke else '최종'} (N {n_grid}, 부트스트랩 {0 if args.skip_bootstrap else n_boot}회)")
    sets = evaluate.load_eval_sets()
    xsets = cross.load_cross_sets()
    if args.smoke:
        xsets = {name: t.join(t.select("instance_id").unique().sort("instance_id").head(SMOKE_ISSUES), on="instance_id", how="semi")
                 for name, t in xsets.items()}
    all_costs = evaluate.load_all_costs()
    costs = evaluate.load_costs(sets, all_costs)
    xcosts = evaluate.load_costs(xsets, all_costs)
    ref = {} if args.smoke else {REF_SET: sets["test"]}  # smoke는 70B test 지표를 계산하지 않는다
    set_info = cross.describe({**xsets, **ref}, n_grid)
    lap("입력·평가 집합")

    # 1. 다른 모델 궤적의 M1 피처 (70B train·test 피처는 04_m1.py가 쓴 파일)
    xt = pl.concat(list(xsets.values())).unique("traj_id").sort("traj_id")
    values = features.load_step_values(max(n_grid), xt)
    t_issue = io.load_trajectories(["traj_id", "instance_id", "issue_text"]).join(xt.select("traj_id"), on="traj_id", how="semi")
    xfeats = features.m1_features(values, xt, n_grid, features.mentions_table(t_issue))
    feat_counts = features.check_features(xfeats, xt, n_grid)
    feats70 = pl.read_parquet(m1.FEATURES_PATH)
    lap(f"다른 모델 궤적 피처 ({xt.height}궤적)")

    # 2. fold 모델 다시 만들기 (저장된 설정) + 8B·405B 채점
    configs = {m: config_of(m) for m in MODELS}
    designs, xids, tasks = {}, {}, []
    for m in MODELS:
        kind, selected = configs[m]
        for n in n_grid:
            D = m1.design(sets, feats70, n, False)
            ids, Xc = cross_matrix(xfeats, xt, n, D["cols"])
            designs[(m, n)], xids[(m, n)] = D, ids
            tasks.append((m, n, kind, selected[n], D["X"], D["y"], D["train"]["instance_id"].to_numpy(), D["train"]["fold"].to_numpy(),
                          D["X_test"], D["test"]["instance_id"].to_numpy(), Xc))
    refits = {(r["model"], r["N"]): r for r in evaluate.parallel(refit_task, tasks, args.jobs)}
    repro = {}
    for (m, n), r in refits.items():
        tr, te = stored_scores(m, n, designs[(m, n)])
        repro[f"{m}_N{n}"] = {"oof_max_abs_diff": float(np.abs(r["oof"] - tr).max()),
                              "test_folds_max_abs_diff": float(np.abs(r["test_folds"] - te).max())}
    bad = {k: v for k, v in repro.items() if max(v.values()) > REPRO_TOL}
    if bad:
        sys.exit(f"다시 만든 fold 모델이 저장된 M1 예측과 다르다 (허용 {REPRO_TOL}): {bad}. 이 머신에서 04_m1.py를 다시 돌린 뒤 실행한다.")
    lap(f"fold 모델 다시 만들기·채점 ({len(tasks)}개, 저장된 예측과 같음)")

    # 3. 예측 (다른 모델 궤적: 평균 점수 + fold 점수 컬럼, 70B test: 저장된 예측)
    long = pl.concat([pl.DataFrame({"traj_id": xids[(m, n)], "N": pl.Series([n] * len(xids[(m, n)]), dtype=pl.Int16),
                                    "fold": pl.Series([k] * len(xids[(m, n)]), dtype=pl.Int8), "p_fail": r["cross_folds"][k]})
                      .with_columns(model=pl.lit(m))
                      for (m, n), r in refits.items() for k in range(split.N_FOLDS)]).select("model", "traj_id", "N", "fold", "p_fail") \
        .sort("model", "N", "traj_id", "fold")
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(long.to_arrow(), data_dir / "m1_cross_preds.parquet", compression="zstd")
    model_preds = {}
    for m in MODELS:
        lm = long.filter(pl.col("model") == m).drop("model")
        mean = lm.group_by("traj_id", "N").agg(pl.col("p_fail").mean()).sort("N", "traj_id")
        xp = evaluate.attach_fold_scores(mean, lm)
        model_preds[m] = {name: evaluate.split_preds(xp, t) for name, t in xsets.items()}
        if ref:
            model_preds[m][REF_SET] = evaluate.test_preds(preds.read(m), preds.read_test_folds(m), sets["test"]).filter(pl.col("N").is_in(list(n_grid)))
    eval_sets = {**xsets, **ref}
    eval_costs = {**xcosts, **({REF_SET: costs["test"]} if ref else {})}
    lap("예측 정리")

    # 4. 운영점 선택 (train OOF), K*
    cap_train = metrics.fixed_cap_curve(sets["train"], costs["train"], range(1, 81))
    k_star = metrics.select_cap(cap_train, metrics.SELECT_MAX_LOSS)["K"]
    chosen = {m: choose_points(m, sets, costs, n_grid) for m in MODELS}
    p5_match = {m: phase5_point_match(m, chosen[m]) for m in MODELS}
    lap("운영점 선택 (train OOF)")

    # 5. 점추정: 판별 지표, 운영점, 고정 상한
    disc_rows, op_rows = [], []
    ref_train = evaluate.cap_point(sets["train"], k_star, costs["train"])
    for m in MODELS:
        for (n, policy), op in chosen[m].items():
            tr = {c: op[c] for c in evaluate.POINT_COLUMNS}
            op_rows.append({"model": m, "policy": policy, "N": n, "tau": op["tau"], "cap": op["cap"], "split": "train", **tr,
                            **{f"d_{c}": tr[c] - ref_train[c] for c in evaluate.DELTA_COLUMNS}})
    for name, t in eval_sets.items():
        ref = evaluate.cap_point(t, k_star, eval_costs[name])  # 차이의 기준: 같은 집합에 K*를 적용한 결과
        for m in MODELS:
            for n in n_grid:
                p = model_preds[m][name].filter(pl.col("N") == n)
                disc_rows.append({"model": m, "split": name, "N": n, **evaluate.discrimination_row(t, p.select("traj_id", "N", "p_fail"), n)})
                for policy in evaluate.POLICIES:
                    op = chosen[m][(n, policy)]
                    pt = evaluate.test_point(t, p, n, op["tau"], op["cap"], eval_costs[name])
                    op_rows.append({"model": m, "policy": policy, "N": n, "tau": op["tau"], "cap": op["cap"], "split": name, **pt,
                                    **{f"d_{c}": pt[c] - ref[c] for c in evaluate.DELTA_COLUMNS}})
        for policy, k in (("cap", k_star), ("cap_proposal", evaluate.K_PROPOSAL)):
            pt = evaluate.cap_point(t, k, eval_costs[name])
            op_rows.append({"model": "b0_cap", "policy": policy, "N": None, "tau": None, "cap": k, "split": name, **pt,
                            **({f"d_{c}": pt[c] - ref[c] for c in evaluate.DELTA_COLUMNS} if k != k_star else {})})
    lap("판별 지표·운영점 적용")

    # 6. 부트스트랩 (이슈 단위, 집합마다 같은 재표본을 모델·N 사이에서 공유)
    boots = {}
    if not args.skip_bootstrap:
        btasks, bkeys = [], []
        for name, t in eval_sets.items():
            for m in MODELS:
                for n in n_grid:
                    points = {pol: (chosen[m][(n, pol)]["tau"], chosen[m][(n, pol)]["cap"]) for pol in evaluate.POLICIES}
                    c = evaluate.compressed_costs(eval_costs[name], n, [cap for _, cap in points.values()], k_star)
                    stat = evaluate.bootstrap_stat(n, points, k_star, c, discrimination=True)
                    btasks.append((t, model_preds[m][name].filter(pl.col("N") == n), stat, n_boot))
                    bkeys.append((m, name, n))
            ks = [k_star, evaluate.K_PROPOSAL]
            c = evaluate.compressed_costs(eval_costs[name], None, ks[1:], k_star)
            btasks.append((t, pl.DataFrame(schema=preds.SCHEMA), evaluate.cap_stat(ks, c), n_boot))
            bkeys.append(("b0_cap", name, None))
        boots = dict(zip(bkeys, evaluate.parallel(evaluate.boot_task, btasks, args.jobs)))
        lap(f"부트스트랩 ({len(btasks)}개 × {n_boot}회)")

    # 7. 표
    out_dir.mkdir(parents=True, exist_ok=True)
    set_info.write_csv(out_dir / "sets.csv", float_precision=FLOAT)
    disc = pl.DataFrame(disc_rows)
    if boots:
        ci = evaluate.discrimination_ci(boots).pivot(on="stat", index=["model", "split", "N"], values=["lo", "hi"])
        ci = ci.rename({c: c.split("_", 1)[1] + "_" + c.split("_", 1)[0] for c in ci.columns if c.startswith(("lo_", "hi_"))})
        disc = disc.join(ci.with_columns(pl.col("N").cast(pl.Int64)), on=["model", "split", "N"], how="left")
    keep = ["model", "split", "N", "n_running", "base_rate", "within_n_issues", "within_n_pairs"]
    for c in ("auroc", "pr_auc", "within_pair", "within_issue_mean"):
        keep += [x for x in (c, f"{c}_lo", f"{c}_hi") if x in disc.columns]
    disc = disc.select(keep).sort("split", "N", "model")
    disc.write_csv(out_dir / "discrimination.csv", float_precision=FLOAT)
    ops = evaluate.operating_table(op_rows, boots, test_splits=tuple(eval_sets))
    ops.write_csv(out_dir / "operating_points.csv", float_precision=FLOAT)
    nsum = pl.concat([ops.filter((pl.col("split") == s) & (pl.col("model") == "b0_cap")).select(SUMMARY_COLUMNS) for s in eval_sets]
                     + [ops.filter(pl.col("split").is_in(list(eval_sets)) & (pl.col("model") != "b0_cap"))
                        .sort("split", "model", "policy", "N").select(SUMMARY_COLUMNS)])
    nsum.write_csv(out_dir / "n_summary.csv", float_precision=FLOAT)
    lap("표 저장")

    check(args, out_dir, xsets, eval_sets, sets, feat_counts, repro, k_star, chosen, p5_match, disc, ops, boots, n_grid, n_boot, timings)


def check(args, out_dir, xsets, eval_sets, sets, feat_counts, repro, k_star, chosen, p5_match, disc, ops, boots, n_grid, n_boot, timings):
    model_rows = ops.filter((pl.col("model") != "b0_cap") & (pl.col("split") != "train"))
    test_issues = set(sets["test"]["instance_id"].unique().to_list())
    crit = {
        "C1_cross_sets": all(t.height > 0 for t in xsets.values())
        and all(set(xsets[f"{k}_unseen"]["instance_id"].unique().to_list()) <= test_issues for k in cross.MODEL_KEYS)
        and not any(set(t["traj_id"].to_list()) & set(sets["train"]["traj_id"].to_list()) for t in xsets.values()),
        "C2_features_one_to_one_finite": set(feat_counts) == set(n_grid),  # check_features가 실패 시 이미 멈춘다
        "C3_refit_reproduces_stored_preds": all(max(v.values()) <= REPRO_TOL for v in repro.values()),
        "C4_k_star": k_star == 40,
        "C5_points_complete": model_rows.height == len(MODELS) * len(n_grid) * len(evaluate.POLICIES) * len(eval_sets)
        and all(model_rows[f"{c}_fold_min"].null_count() == 0 for c in evaluate.POINT_COLUMNS)
        and (args.skip_bootstrap or (model_rows["lost_success_hi"].null_count() == 0 and disc["auroc_lo"].null_count() == 0)),
    }
    s = {
        "mode": "smoke" if args.smoke else "final",
        "models": list(MODELS), "n_grid": list(n_grid), "k_star": k_star, "k_proposal": evaluate.K_PROPOSAL,
        "sets": {name: {"traj": t.height, "issues": t["instance_id"].n_unique()} for name, t in xsets.items()},
        "features_rows_per_N": {str(k): v for k, v in feat_counts.items()},
        "refit_vs_stored": repro, "repro_tol": REPRO_TOL,
        "chosen_points": {f"{m}_N{n}_{pol}": {"tau": op["tau"], "cap": op["cap"]} for m in MODELS for (n, pol), op in chosen[m].items()},
        "chosen_points_same_as_committed_phase5_csv": p5_match,
        "bootstrap": None if args.skip_bootstrap else {"n_boot": n_boot, "tasks": len(boots), "seed": io.SEED},
        "criteria": crit,
        "criteria_note": "pytest 통과와 재실행 시 results/phase8/ 바이트 동일은 스크립트 밖에서 확인한다. 실행 시간은 표준 출력에만 남긴다",
    }
    io.write_summary(out_dir / "summary.json", s)
    if not args.smoke:
        with pl.Config(tbl_rows=80, tbl_cols=16, tbl_width_chars=220, float_precision=3):
            print(disc.filter(pl.col("model") == "m1_xgb").select("split", "N", "n_running", "base_rate", "auroc",
                                                               *[c for c in ("auroc_lo", "auroc_hi") if c in disc.columns], "within_pair"))
            print(ops.filter((pl.col("split") != "train") & (pl.col("model").is_in(["m1_xgb", "b0_cap"])) & (pl.col("policy") != "single"))
                  .select("split", "model", "policy", "N", "cap", "stop_rate", "lost_success", "lost_success_hi", "meets_judge_ci_hi",
                          "saved_steps", "d_saved_steps", "saved_tokens", "d_saved_tokens"))
    print(json.dumps({"criteria": crit, "timings": timings, "chosen_points_same_as_committed_phase5_csv": p5_match},
                     ensure_ascii=False, indent=2, default=str))
    failed = [k for k, v in crit.items() if not v]
    assert not failed, failed
    print("\n완료 기준 통과")


if __name__ == "__main__":
    main()
