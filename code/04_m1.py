"""Phase 5: M1 피처 모델 (로지스틱 회귀·XGBoost × B1 포함/제외) + SHAP

실행: uv run --no-sync python code/04_m1.py [--force] [--n-boot 1000] [--skip-bootstrap] [--skip-shap] [--jobs 12]

입력: data/processed/trajectories.parquet, steps.parquet (01_parse.py), splits.parquet (02_split.py),
      preds/b1_n.parquet + b1_n.test_folds.parquet (03_baselines.py, M1 입력 B1 점수), results/phase4/ (비교 표)
출력:
- data/processed/features/m1_n.parquet       traj_id, N, M1 피처 (70B train + test, 시점 N 진행 중 궤적)
- data/processed/preds/m1_*.parquet           traj_id, N, p_fail (train OOF + test 5개 fold 모델 평균, 판별 지표용)
- data/processed/preds/m1_*.test_folds.parquet  test fold 모델별 점수 (운영점 평가 P4-D6)
- results/phase5/                             그리드, 판별 표, 운영점 표, 비교 표, ② 민감도, 피처 분포, SHAP 표·그림, 요약

모델 4종 (N마다 따로 학습, D2): m1_lr, m1_lr_b1, m1_xgb, m1_xgb_b1. _b1은 B1 점수(b1_n)를 피처로 더한다.
- 학습: oof.oof_predict (train 5-fold OOF + test fold 모델별 점수). 하이퍼파라미터는 train OOF AUROC로 그리드에서 고른다
- B1 입력 (P4-D7): train = preds.load_feature("b1_n", "train") (OOF), test fold k 모델 = load_feature("b1_n", "test", fold=k) (①).
  ② 민감도 = M1 fold k를 load_feature_other_folds("b1_n", k)의 j≠k 4개로 각각 채점 (보고만)
- 평가: afp/evaluate.py 공용 함수 (판별 표, 운영점 선택·적용·부트스트랩, P4-D6) — 03_baselines.py와 같은 코드
- 운영점·하이퍼파라미터·좋은 N은 train OOF에서만 고른다. 테스트는 모든 N을 보고한다
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from afp import evaluate, features, io, metrics, oof, preds, split

MODEL_SPECS = {  # model_id → (학습기, B1 점수 포함)
    "m1_lr": ("lr", False),
    "m1_lr_b1": ("lr", True),
    "m1_xgb": ("xgb", False),
    "m1_xgb_b1": ("xgb", True),
}
B1_MODEL = "b1_n"  # P4-D3 주 결과
# 작은 그리드. 클래스 불균형 처리(balanced: 성공·실패 무게를 같게)를 그리드에 넣는다. 순서 = 단순한 것부터 (동률이면 앞의 것)
LR_GRID = tuple({"C": c, "class_weight": w} for w in (None, "balanced") for c in (0.01, 0.1, 1.0, 10.0))
XGB_GRID = tuple({"max_depth": d, "n_estimators": t, "balance": b} for b in (False, True) for d in (3, 6) for t in (200, 600))
XGB_FIXED = {"learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "reg_lambda": 1.0,
             "tree_method": "hist", "n_jobs": 1, "random_state": io.SEED, "eval_metric": "logloss"}
LR_TOL = 1e-6
LR_MAX_ITER = 5000
GRID_TIE = 0.002  # OOF AUROC 차이가 이보다 작으면 그리드 앞쪽(단순한) 설정 (B1과 같은 값)
GOOD_N_TIE = 0.005  # 좋은 N: OOF AUROC가 최대값에서 이 안이면 train 절약 스텝(결합 정책)이 큰 N
SHAP_N = (5, 10, 20)
SHAP_TOP = 10
SHAP_PLOT_TOP = 15
DEPENDENCE_TOP = 3
FLOAT = 6
FEATURES_PATH = io.PROCESSED_DIR / "features" / "m1_n.parquet"
COMPARE_MODELS = ("b0_err", "b0_editfail", "b1_static", "b1_n")  # Phase 4 비교 대상 (results/phase4/에서 읽음)


# ---- 피처 ----

def build_features(sets):
    trajs = pl.concat([t.select("traj_id", "instance_id", "n_steps") for t in sets.values()])
    values = features.load_step_values(max(evaluate.N_GRID), trajs)
    t = io.load_trajectories(["traj_id", "instance_id", "issue_text"]).join(trajs.select("traj_id"), on="traj_id", how="semi")
    feats = features.m1_features(values, trajs, evaluate.N_GRID, features.mentions_table(t))
    counts = features.check_features(feats, trajs, evaluate.N_GRID)
    return feats, values, counts


def write_features(feats: pl.DataFrame, force: bool) -> bool:
    """내용이 같으면 다시 쓰지 않는다 (바이트 재현). 다르면 --force 없이는 멈춘다 (preds.write와 같은 방식)."""
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FEATURES_PATH.exists():
        if pl.read_parquet(FEATURES_PATH).equals(feats):
            return False
        if not force:
            raise SystemExit(f"{FEATURES_PATH}가 다시 계산한 피처와 다르다. 확인 후 --force로 덮어쓴다.")
    tmp = FEATURES_PATH.with_name(FEATURES_PATH.name + ".tmp")
    pq.write_table(feats.to_arrow(), tmp, compression="zstd")
    tmp.replace(FEATURES_PATH)
    return True


def feature_dist(feats, train):
    """피처별 train 분포 (N마다): 평균·표준편차·분위수·0이 아닌 비율, 성공·실패 평균."""
    rows = []
    for n in evaluate.N_GRID:
        f = feats.filter(pl.col("N") == n).join(train.select("traj_id", "target"), on="traj_id", how="inner")
        succ = f["target"].to_numpy()
        for c in features.FEATURES:
            v = f[c].to_numpy()
            q = np.quantile(v, (0.05, 0.5, 0.95))
            rows.append({"N": n, "feature": c, "n": len(v), "mean": float(v.mean()), "std": float(v.std()),
                         "q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2]), "nonzero": float((v != 0).mean()),
                         "mean_success": float(v[succ].mean()), "mean_failure": float(v[~succ].mean())})
    return pl.DataFrame(rows)


# ---- 설계 행렬 ----

def design(sets, feats, n, use_b1):
    """시점 N의 train·test 행렬. test는 fold k 모델용 행렬 목록 (B1 포함이면 B1 fold k 점수, P4-D7 ①)."""
    cols = [*features.FEATURES, *([features.B1_FEATURE] if use_b1 else [])]
    f = feats.filter(pl.col("N") == n)
    out = {}
    for name in ("train", "test"):
        t = metrics.running(sets[name], n).select("traj_id", "instance_id", "target", "fold").sort("traj_id")
        out[name] = t.join(f, on="traj_id", how="left")
        if out[name].select(features.FEATURES).null_count().sum_horizontal().item():
            raise AssertionError(f"N={n} {name}: 피처가 없는 진행 중 궤적")
    train, test = out["train"], out["test"]
    b1_check = {}
    if use_b1:
        b1 = preds.load_feature(B1_MODEL, "train").filter(pl.col("N") == n).select("traj_id", pl.col("p_fail").alias(features.B1_FEATURE))
        train = train.join(b1, on="traj_id", how="left")
        if train[features.B1_FEATURE].null_count():
            raise AssertionError(f"N={n}: B1 OOF 점수가 없는 train 궤적")
        xs, fold_b1 = [], []
        for k in range(split.N_FOLDS):
            bk = preds.load_feature(B1_MODEL, "test", fold=k).filter(pl.col("N") == n).select("traj_id", pl.col("p_fail").alias(features.B1_FEATURE))
            tk = test.join(bk, on="traj_id", how="left")
            if tk[features.B1_FEATURE].null_count():
                raise AssertionError(f"N={n}: B1 fold {k} 점수가 없는 test 궤적")
            xs.append(tk.select(cols).to_numpy())
            fold_b1.append(tk[features.B1_FEATURE].to_numpy())
        fb = np.stack(fold_b1)
        # ③ 금지 확인: fold k 입력은 fold마다 다르다 (평균 점수 하나를 모든 fold에 넣지 않았다)
        b1_check = {"test_fold_input_max_spread": float((fb.max(0) - fb.min(0)).max())}
    else:
        x = test.select(cols).to_numpy()
        xs = [x] * split.N_FOLDS
    return {"cols": cols, "train": train, "test": test, "X": train.select(cols).to_numpy(), "y": (~train["target"]).to_numpy(),
            "X_test": xs, "b1_check": b1_check}


def other_fold_test(test, n, cols):
    """② 민감도 입력: {(k, j): fold k 모델에 넣을 B1 fold j 점수 행렬} (j≠k, load_feature_other_folds)."""
    out = {}
    for k in range(split.N_FOLDS):
        o = preds.load_feature_other_folds(B1_MODEL, k).filter(pl.col("N") == n)
        for j in sorted(o["src_fold"].unique().to_list()):
            bj = o.filter(pl.col("src_fold") == j).select("traj_id", pl.col("p_fail").alias(features.B1_FEATURE))
            tj = test.join(bj, on="traj_id", how="left")
            if tj[features.B1_FEATURE].null_count():
                raise AssertionError(f"N={n}: ② B1 fold {j} 점수가 없는 test 궤적")
            out[(k, j)] = tj.select(cols).to_numpy()
    return out


# ---- 학습기 ----

def fit_model(kind, cfg, X, y):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits
        Z = np.log1p(X)  # 피처는 모두 0 이상 (횟수·길이·B1 확률). 꼬리가 긴 횟수·길이를 줄인 뒤 표준화
        scaler = StandardScaler().fit(Z)
        with threadpool_limits(1):
            lr = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], solver="lbfgs", tol=LR_TOL,
                                    max_iter=LR_MAX_ITER).fit(scaler.transform(Z), y)
        return ("lr", scaler, lr)
    if kind == "xgb":
        from xgboost import XGBClassifier
        n_fail, n_succ = int(y.sum()), int((~y.astype(bool)).sum())
        spw = n_succ / n_fail if cfg["balance"] else 1.0  # 실패(y=1)가 다수라 balanced면 실패 무게를 줄인다
        m = XGBClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], scale_pos_weight=spw, **XGB_FIXED)
        return ("xgb", m.fit(X, y))
    raise ValueError(kind)


def predict_model(model, X):
    if model[0] == "lr":
        _, scaler, lr = model
        return lr.predict_proba(scaler.transform(np.log1p(X)))[:, 1]
    return model[1].predict_proba(X)[:, 1]


def run_oof(kind, cfg, X, y, inst, fold, X_test, test_inst, keep_models=False):
    """oof.oof_predict로 train OOF와 test fold 모델별 점수. fold k 모델은 X_test[k]로 채점한다 (P4-D7 ①)."""
    train_rows = pl.DataFrame({"instance_id": inst, "fold": fold, "_i": np.arange(len(y))})
    test_rows = pl.DataFrame({"instance_id": test_inst, "_i": np.arange(len(test_inst)), "_src": np.full(len(test_inst), -1)})

    def fit(rows):
        idx = rows["_i"].to_numpy()
        return fit_model(kind, cfg, X[idx], y[idx])

    def predict(model, rows):
        idx = rows["_i"].to_numpy()
        if "_src" in rows.columns:
            return predict_model(model, X_test[int(rows["_src"][0])][idx])
        return predict_model(model, X[idx])

    res = oof.oof_predict(train_rows, fit, predict, test=test_rows, keep_models=keep_models,
                          test_for_fold=lambda k, t: t.with_columns(_src=pl.lit(k)))
    from sklearn.metrics import roc_auc_score
    p = res.train["p_fail"].to_numpy()
    return {"oof": p, "test_folds": res.test_fold_scores, "oof_auroc": float(roc_auc_score(y, p)),
            "seen_issue_scores": res.seen_issue_scores, "models": res.models}


def grid_task(model_id, n, i, kind, cfg, X, y, inst, fold, X_test, test_inst):
    r = run_oof(kind, cfg, X, y, inst, fold, X_test, test_inst)
    return {"model": model_id, "N": n, "i": i, **r}


def refit_task(model_id, n, kind, cfg, X, y, inst, fold, X_test, test_inst):
    r = run_oof(kind, cfg, X, y, inst, fold, X_test, test_inst, keep_models=True)
    return {"model": model_id, "N": n, **r}


def choose(rows):
    """그리드 결과(같은 model·N) → 고른 인덱스. 최고 OOF AUROC에서 GRID_TIE 안이면 그리드 앞쪽 설정."""
    best = max(r["oof_auroc"] for r in rows)
    return min(r["i"] for r in rows if r["oof_auroc"] >= best - GRID_TIE)


def cfg_str(cfg):
    return json.dumps(cfg, sort_keys=True)


# ---- SHAP ----

def shap_analysis(model_id, n, models, D, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from scipy.stats import spearmanr

    cols = D["cols"]
    svs = [shap.TreeExplainer(m[1]).shap_values(D["X_test"][k]) for k, m in enumerate(models)]
    mean_abs = np.stack([np.abs(s).mean(0) for s in svs])  # fold × 피처
    imp = mean_abs.mean(0)
    order = np.argsort(-imp, kind="stable")
    top = [cols[i] for i in order[:SHAP_TOP]]
    fold_ranks = np.stack([np.argsort(np.argsort(-m, kind="stable"), kind="stable") for m in mean_abs])
    fold_top = [set(np.array(cols)[np.argsort(-m, kind="stable")[:SHAP_TOP]]) for m in mean_abs]
    pair_rho = [spearmanr(fold_ranks[a], fold_ranks[b]).statistic for a in range(len(models)) for b in range(a + 1, len(models))]
    imp_rows = [{"model": model_id, "N": n, "feature": cols[i], "rank": r + 1, "mean_abs_shap": float(imp[i]),
                 "fold_min": float(mean_abs[:, i].min()), "fold_max": float(mean_abs[:, i].max()),
                 "fold_rank_min": int(fold_ranks[:, i].min()) + 1, "fold_rank_max": int(fold_ranks[:, i].max()) + 1,
                 "mean_shap_when_nonzero": float(np.mean([s[:, i][D["X_test"][k][:, i] != 0].mean() if (D["X_test"][k][:, i] != 0).any() else np.nan
                                                          for k, s in enumerate(svs)]))}
                for r, i in enumerate(order)]
    stab = {"model": model_id, "N": n, "top_k": SHAP_TOP,
            "fold_top_overlap_min": min(len(t & set(top)) for t in fold_top), "fold_top_overlap_max": max(len(t & set(top)) for t in fold_top),
            "fold_top_intersection": len(set.intersection(*fold_top)),
            "fold_rank_spearman_min": float(min(pair_rho)), "fold_rank_spearman_mean": float(np.mean(pair_rho))}
    # 그림: fold 모델 5개의 SHAP 값을 표본마다 평균 (= 5개 모델 출력 평균(log-odds)의 분해). 색은 피처 값 (B1은 fold 평균)
    avg = np.mean(svs, axis=0)
    x_plot = np.mean(D["X_test"], axis=0)
    np.random.seed(io.SEED)  # 점 위치 지터가 전역 난수를 쓴다
    shap.summary_plot(avg, x_plot, feature_names=cols, max_display=SHAP_PLOT_TOP, show=False, plot_size=(8, 6))
    plt.title(f"{model_id} N={n} (test, SHAP averaged over 5 fold models, log-odds of failure)", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / f"shap_summary_{model_id}_N{n}.png", dpi=100, metadata={"Software": None})
    plt.close("all")
    for c in top[:DEPENDENCE_TOP]:
        np.random.seed(io.SEED)
        shap.dependence_plot(c, avg, x_plot, feature_names=cols, interaction_index=None, show=False)
        plt.title(f"{model_id} N={n}: {c}", fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / f"shap_dependence_{model_id}_N{n}_{c}.png", dpi=100, metadata={"Software": None})
        plt.close("all")
    return imp_rows, stab


# ---- 실행 ----

def main():
    ap = argparse.ArgumentParser(description="Phase 5: M1 피처 모델 + SHAP")
    ap.add_argument("--force", action="store_true", help="저장된 피처·예측 파일과 달라도 덮어씀")
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    ap.add_argument("--skip-bootstrap", action="store_true", help="부트스트랩 구간을 건너뜀 (개발용, 완료 기준 일부 검사 안 함)")
    ap.add_argument("--skip-shap", action="store_true", help="SHAP을 건너뜀 (개발용)")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    for p, script in ((io.TRAJECTORIES_PATH, "01_parse.py"), (io.STEPS_PATH, "01_parse.py"), (io.SPLITS_PATH, "02_split.py"),
                      (preds.fold_path(B1_MODEL), "03_baselines.py"), (io.RESULTS_DIR / "phase4" / "operating_points.csv", "03_baselines.py")):
        if not p.exists():
            sys.exit(f"{p}가 없다. 먼저 uv run --no-sync python code/{script} 를 실행한다.")
    t_start = time.time()
    timings = {}

    def lap(name):
        timings[name] = round(time.time() - t_start - sum(timings.values()), 1)
        print(f"[{timings[name]:>6.1f}s] {name}", flush=True)

    out_dir = io.RESULTS_DIR / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = evaluate.load_eval_sets()
    costs = evaluate.load_costs(sets)
    lap("입력·평가 집합·맥락 길이")

    # 1. 피처
    feats, values, feat_counts = build_features(sets)
    feats_written = write_features(feats, args.force)
    feat_dist = feature_dist(feats, sets["train"])
    feat_dist.write_csv(out_dir / "feature_dist.csv", float_precision=FLOAT)
    lap(f"피처 ({len(features.FEATURES)}개, 행 {feats.height})")

    # 2. 설계 행렬 + 그리드 (train OOF AUROC로 고름)
    designs = {(m, n): design(sets, feats, n, b1) for m, (_, b1) in MODEL_SPECS.items() for n in evaluate.N_GRID}
    tasks = []
    for (m, n), D in designs.items():
        kind = MODEL_SPECS[m][0]
        args_d = (D["X"], D["y"], D["train"]["instance_id"].to_numpy(), D["train"]["fold"].to_numpy(), D["X_test"],
                  D["test"]["instance_id"].to_numpy())
        for i, cfg in enumerate(LR_GRID if kind == "lr" else XGB_GRID):
            tasks.append((m, n, i, kind, cfg, *args_d))
    # 무거운 작업(XGBoost 큰 설정)부터 넣어 병렬 대기를 줄인다 (결과는 순서와 무관)
    order = sorted(range(len(tasks)), key=lambda j: (tasks[j][3] != "xgb", -tasks[j][4].get("n_estimators", 0) * tasks[j][4].get("max_depth", 0)))
    res = parallel_ordered(grid_task, tasks, order, args.jobs)
    grid_rows, selected = [], {}
    for (m, n) in designs:
        rs = sorted((r for r in res if r["model"] == m and r["N"] == n), key=lambda r: r["i"])
        assert all(r["seen_issue_scores"] == 0 for r in rs)
        pick = choose(rs)
        selected[(m, n)] = rs[pick]
        grid = LR_GRID if MODEL_SPECS[m][0] == "lr" else XGB_GRID
        grid_rows += [{"model": m, "N": n, "i": r["i"], "config": cfg_str(grid[r["i"]]), "oof_auroc": r["oof_auroc"],
                       "selected": r["i"] == pick} for r in rs]
    grid_df = pl.DataFrame(grid_rows)
    grid_df.write_csv(out_dir / "m1_grid.csv", float_precision=FLOAT)
    lap(f"그리드 OOF ({len(tasks)}개)")

    # 3. 고른 설정을 fold 모델을 남기고 다시 학습 (② 민감도·SHAP용). 같은 점수가 나와야 한다 (결정성 확인)
    need = [(m, n) for (m, n) in designs if MODEL_SPECS[m][1] or (MODEL_SPECS[m][0] == "xgb" and n in SHAP_N)]
    if args.skip_shap:
        need = [(m, n) for (m, n) in need if MODEL_SPECS[m][1]]
    rtasks = []
    for (m, n) in need:
        D, kind = designs[(m, n)], MODEL_SPECS[m][0]
        cfg = (LR_GRID if kind == "lr" else XGB_GRID)[selected[(m, n)]["i"]]
        rtasks.append((m, n, kind, cfg, D["X"], D["y"], D["train"]["instance_id"].to_numpy(), D["train"]["fold"].to_numpy(),
                       D["X_test"], D["test"]["instance_id"].to_numpy()))
    refits = {(r["model"], r["N"]): r for r in evaluate.parallel(refit_task, rtasks, args.jobs)}
    refit_same = all(np.array_equal(r["oof"], selected[k]["oof"]) and np.array_equal(r["test_folds"], selected[k]["test_folds"])
                     for k, r in refits.items())
    lap(f"고른 설정 다시 학습 ({len(rtasks)}개)")

    # 4. 예측 저장
    written, written_folds = {}, {}
    for m, (kind, _) in MODEL_SPECS.items():
        parts, fparts = [], []
        for n in evaluate.N_GRID:
            D, r = designs[(m, n)], selected[(m, n)]
            parts.append(D["train"].select("traj_id", N=pl.lit(n, dtype=pl.Int16), p_fail=pl.Series(r["oof"])))
            parts.append(D["test"].select("traj_id", N=pl.lit(n, dtype=pl.Int16), p_fail=pl.Series(r["test_folds"].mean(axis=0))))
            fparts += [D["test"].select("traj_id", N=pl.lit(n, dtype=pl.Int16), fold=pl.lit(k, dtype=pl.Int8), p_fail=pl.Series(r["test_folds"][k]))
                       for k in range(split.N_FOLDS)]
        grid = LR_GRID if kind == "lr" else XGB_GRID
        config = {"learner": kind, "features": designs[(m, evaluate.N_GRID[0])]["cols"],
                  "selected": {str(n): grid[selected[(m, n)]["i"]] for n in evaluate.N_GRID},
                  "fixed": XGB_FIXED if kind == "xgb" else {"transform": "log1p+standardize", "tol": LR_TOL, "max_iter": LR_MAX_ITER},
                  "b1_input": (f"train={B1_MODEL} OOF, test fold k={B1_MODEL} fold k (P4-D7 ①)" if MODEL_SPECS[m][1] else None)}
        written[m] = preds.write(m, pl.concat(parts), sets, evaluate.N_GRID, config, "5fold-oof/test-fold-mean", force=args.force)
        written_folds[m] = preds.write_test_folds(m, pl.concat(fparts), sets["test"], evaluate.N_GRID, split.N_FOLDS, force=args.force)
    all_preds = {m: preds.read(m) for m in MODEL_SPECS}
    fold_test = {m: preds.read_test_folds(m) for m in MODEL_SPECS}
    model_preds = {m: (all_preds[m], fold_test[m]) for m in MODEL_SPECS}
    lap("예측 저장·검증")

    # 5. 판별 지표 + 운영점 (03_baselines.py와 같은 evaluate 함수)
    disc = pl.concat([evaluate.discrimination_table(m, all_preds[m], sets, evaluate.N_GRID) for m in MODEL_SPECS])
    disc.write_csv(out_dir / "discrimination.csv", float_precision=FLOAT)
    cap_train = metrics.fixed_cap_curve(sets["train"], costs["train"], range(1, 81))
    k_star = metrics.select_cap(cap_train, metrics.SELECT_MAX_LOSS)["K"]
    main_rows, sens_rows, chosen = evaluate.operating_points(model_preds, sets, costs, k_star)
    cap_main, cap_sens = evaluate.cap_rows(sets, costs, k_star, cap_train)
    lap("판별 지표·운영점 선택·적용")

    curve_dir = out_dir / "cost_curves"
    if curve_dir.exists():
        shutil.rmtree(curve_dir)
    curve_dir.mkdir()
    for m in MODEL_SPECS:
        for (name, n), c in evaluate.model_curves(all_preds[m], sets, costs, evaluate.N_GRID, fold_test[m]).items():
            evaluate.thin_curve(c).write_csv(curve_dir / f"{m}_{name}_N{n}.csv", float_precision=FLOAT)

    boots = {}
    if not args.skip_bootstrap:
        tasks_b, keys = evaluate.bootstrap_tasks(model_preds, sets, costs, chosen, k_star, {m: evaluate.N_GRID for m in MODEL_SPECS}, args.n_boot)
        cap_tasks, cap_keys = evaluate.cap_bootstrap_tasks(sets, costs, k_star, args.n_boot)
        boots = dict(zip(keys + cap_keys, evaluate.parallel(evaluate.boot_task, tasks_b + cap_tasks, args.jobs)))
        evaluate.discrimination_ci(boots).write_csv(out_dir / "discrimination_ci.csv", float_precision=FLOAT)
        lap(f"부트스트랩 ({len(tasks_b) + len(cap_tasks)}개 × {args.n_boot}회)")
    ops_df = evaluate.operating_table(cap_main + main_rows, boots)
    ops_df.write_csv(out_dir / "operating_points.csv", float_precision=FLOAT)
    evaluate.sensitivity_table(cap_sens + sens_rows).write_csv(out_dir / "operating_points_sensitivity.csv", float_precision=FLOAT)

    # 6. ② 민감도 (P4-D7): M1 fold k를 B1 fold j≠k 4개로 각각 채점. 판별 = 20개 점수 평균, 운영점 = 20개 결과 평균
    sens2 = b1_input_sensitivity(designs, refits, sets, costs, chosen, model_preds)
    sens2.write_csv(out_dir / "b1_input_sensitivity.csv", float_precision=FLOAT)
    lap("② 민감도")

    # 7. SHAP
    shap_imp, shap_stab = [], []
    if not args.skip_shap:
        fig_dir = out_dir / "figures"
        if fig_dir.exists():
            shutil.rmtree(fig_dir)
        fig_dir.mkdir()
        for m in ("m1_xgb", "m1_xgb_b1"):
            for n in SHAP_N:
                imp, stab = shap_analysis(m, n, refits[(m, n)]["models"], designs[(m, n)], fig_dir)
                shap_imp += imp
                shap_stab.append(stab)
        pl.DataFrame(shap_imp).write_csv(out_dir / "shap_importance.csv", float_precision=FLOAT)
        pl.DataFrame(shap_stab).write_csv(out_dir / "shap_stability.csv", float_precision=FLOAT)
        lap("SHAP")

    # 8. 비교 표 (B0·B1은 results/phase4/에서 읽는다), 좋은 N, 요약
    compare = comparison_tables(out_dir, disc, ops_df, boots)
    good_n = choose_good_n(disc, ops_df)
    pl.DataFrame(good_n).write_csv(out_dir / "good_n.csv", float_precision=FLOAT)
    check(args, out_dir, sets, feat_counts, feats_written, grid_df, selected, designs, written, written_folds, disc, ops_df, k_star,
          sens2, shap_imp, shap_stab, compare, good_n, refit_same, timings, boots)


def parallel_ordered(fn, tasks, order, jobs):
    out = evaluate.parallel(fn, [tasks[j] for j in order], jobs)
    res = [None] * len(tasks)
    for j, r in zip(order, out):
        res[j] = r
    return res


def b1_input_sensitivity(designs, refits, sets, costs, chosen, model_preds):
    """P4-D7 ② vs ①: 판별 지표(test)와 고른 운영점의 test 손실·절약 (점추정). ①은 저장된 fold 점수 그대로."""
    test = sets["test"]
    rows = []
    for m, (_, use_b1) in MODEL_SPECS.items():
        if not use_b1:
            continue
        for n in evaluate.N_GRID:
            D, models = designs[(m, n)], refits[(m, n)]["models"]
            other = other_fold_test(D["test"], n, D["cols"])
            long = pl.concat([D["test"].select("traj_id", N=pl.lit(n, dtype=pl.Int16), fold=pl.lit(10 * k + j, dtype=pl.Int16),
                                               p_fail=pl.Series(predict_model(models[k], X)))
                              for (k, j), X in other.items()])
            mean2 = long.group_by("traj_id", "N").agg(pl.col("p_fail").mean()).sort("traj_id")
            p1 = evaluate.test_preds(*model_preds[m], test, n)
            p2 = evaluate.attach_fold_scores(mean2, long)
            d1 = evaluate.discrimination_row(test, p1.select("traj_id", "N", "p_fail"), n)
            d2 = evaluate.discrimination_row(test, mean2, n)
            for c in ("auroc", "pr_auc", "within_pair", "within_issue_mean"):
                rows.append({"model": m, "N": n, "kind": "discrimination", "policy": None, "metric": c, "main_1": d1[c], "sens_2": d2[c]})
            for policy in evaluate.POLICIES:
                tau, cap = chosen[(m, n, policy)]
                a = evaluate.test_point(test, p1, n, tau, cap, costs["test"], spread=False)
                b = evaluate.test_point(test, p2, n, tau, cap, costs["test"], spread=False)
                rows += [{"model": m, "N": n, "kind": "operating_point", "policy": policy, "metric": c, "main_1": a[c], "sens_2": b[c]}
                         for c in evaluate.POINT_COLUMNS]
    return pl.DataFrame(rows).with_columns(diff=pl.col("sens_2") - pl.col("main_1"))


def comparison_tables(out_dir, disc, ops_df, boots):
    """B0·B1(Phase 4 결과 파일) + M1 → 판별 비교, 운영점 비교, 손실 구간 폭 비교."""
    p4 = io.RESULTS_DIR / "phase4"
    d4 = pl.read_csv(p4 / "discrimination.csv", schema_overrides={"N": pl.Int64}).filter(pl.col("model").is_in(COMPARE_MODELS))
    d4 = d4.filter(pl.col("N").is_in(list(evaluate.N_GRID)))
    dall = pl.concat([d4.select(disc.columns).cast(disc.schema), disc])
    ci_parts = [pl.read_csv(p4 / "discrimination_ci.csv", schema_overrides={"N": pl.Int64}).filter(pl.col("model").is_in(COMPARE_MODELS))]
    if boots:
        ci_parts.append(pl.read_csv(out_dir / "discrimination_ci.csv", schema_overrides={"N": pl.Int64}))
    ci = pl.concat([c.select("model", "split", "N", "stat", "lo", "hi") for c in ci_parts]) \
        .pivot(on="stat", index=["model", "split", "N"], values=["lo", "hi"])
    ci = ci.rename({c: (c.split("_", 1)[1] + "_" + c.split("_", 1)[0]) for c in ci.columns if c.startswith(("lo_", "hi_"))})
    dcomp = dall.join(ci, on=["model", "split", "N"], how="left")
    keep = ["model", "split", "N", "n_running", "base_rate", "within_n_issues"]
    for c in ("auroc", "pr_auc", "within_pair", "within_issue_mean"):
        keep += [c] + [x for x in (f"{c}_lo", f"{c}_hi") if x in dcomp.columns]
    dcomp = dcomp.select(keep).sort("split", "N", "model")
    dcomp.write_csv(out_dir / "discrimination_compare.csv", float_precision=FLOAT)

    o4 = pl.read_csv(p4 / "operating_points.csv", schema=ops_df.schema)
    cap4 = o4.filter(pl.col("model") == "b0_cap")
    cap5 = ops_df.filter(pl.col("model") == "b0_cap")
    # 같은 평가 집합·시드라 K*·K=35 행(부트스트랩 구간 포함)은 Phase 4와 같아야 한다 (CSV 6자리 기준)
    num = [c for c, t in ops_df.schema.items() if t == pl.Float64]
    cap_same = cap4.height == cap5.height and all(
        np.allclose(cap4[c].fill_null(np.nan).to_numpy(), cap5[c].fill_null(np.nan).to_numpy(), atol=1e-6, equal_nan=True) for c in num)
    ocomp = pl.concat([o4, ops_df.filter(pl.col("model") != "b0_cap")])
    ocomp.write_csv(out_dir / "operating_points_compare.csv", float_precision=FLOAT)

    t = ocomp.filter(pl.col("split") == "test").with_columns(half_width=(pl.col("lost_success_hi") - pl.col("lost_success_lo")) / 2)
    width = t.select("model", "policy", "N", "tau", "cap", "stop_rate", "lost_success", "lost_success_lo", "lost_success_hi", "half_width",
                     "saved_steps", "meets_judge_ci_hi")
    width.write_csv(out_dir / "loss_ci_width_compare.csv", float_precision=FLOAT)
    width_sum = width.filter(pl.col("model") != "b0_cap").group_by("model", "policy").agg(
        n=pl.len(), half_width_median=pl.col("half_width").median(), half_width_min=pl.col("half_width").min(),
        half_width_max=pl.col("half_width").max(), stop_rate_median=pl.col("stop_rate").median(),
        lost_success_median=pl.col("lost_success").median()).sort("policy", "model")
    width_sum.write_csv(out_dir / "loss_ci_width_summary.csv", float_precision=FLOAT)
    return {"discrimination": dcomp, "operating": ocomp, "width": width, "width_summary": width_sum, "cap_rows_match_phase4": bool(cap_same)}


def choose_good_n(disc, ops_df):
    """좋은 N (train OOF만): OOF AUROC 최대에서 GOOD_N_TIE 안의 N 중 train 절약 스텝(결합 정책 운영점)이 가장 큰 N."""
    out = []
    for m in MODEL_SPECS:
        d = disc.filter((pl.col("model") == m) & (pl.col("split") == "train_oof")).select("N", "auroc")
        s = ops_df.filter((pl.col("model") == m) & (pl.col("split") == "train") & (pl.col("policy") == "combined")).select("N", "saved_steps")
        j = d.join(s, on="N").sort("N")
        best = j["auroc"].max()
        cand = j.filter(pl.col("auroc") >= best - GOOD_N_TIE).sort(["saved_steps", "N"], descending=[True, False])
        pick = int(cand["N"][0])
        out += [{"model": m, "N": int(r["N"]), "train_oof_auroc": r["auroc"], "train_saved_steps_combined": r["saved_steps"],
                 "in_tie_band": r["auroc"] >= best - GOOD_N_TIE, "selected": int(r["N"]) == pick} for r in j.iter_rows(named=True)]
    return out


def check(args, out_dir, sets, feat_counts, feats_written, grid_df, selected, designs, written, written_folds, disc, ops_df, k_star,
          sens2, shap_imp, shap_stab, compare, good_n, refit_same, timings, boots):
    test_ops = ops_df.filter(pl.col("split") == "test")
    m1_test = test_ops.filter(pl.col("model").is_in(list(MODEL_SPECS)))
    judged = [] if args.skip_bootstrap else evaluate.judged_and_saves_more(ops_df).to_dicts()
    s2 = sens2.with_columns(abs_diff=pl.col("diff").abs()).group_by("kind", "metric").agg(
        max_abs_diff=pl.col("abs_diff").max(), mean_abs_diff=pl.col("abs_diff").mean()).sort("kind", "metric")
    shap_top = {f"{r['model']}_N{r['N']}": r["feature"] for r in shap_imp if r["rank"] == 1}
    s = {
        "features": {"n_features": len(features.FEATURES), "names": features.FEATURES, "b1_feature": features.B1_FEATURE,
                     "rows_per_N": {str(k): v for k, v in feat_counts.items()}, "written": feats_written},
        "grid": {"lr": list(LR_GRID), "xgb": list(XGB_GRID), "xgb_fixed": XGB_FIXED, "tie": GRID_TIE,
                 "selected": {f"{m}_N{n}": {"i": r["i"], "oof_auroc": r["oof_auroc"]} for (m, n), r in selected.items()}},
        "b1_input": {"model": B1_MODEL, "train": "load_feature(train) OOF", "test": "load_feature(test, fold=k) for M1 fold k (P4-D7 ①)",
                     "test_fold_input_max_spread": {f"{m}_N{n}": D["b1_check"]["test_fold_input_max_spread"]
                                                    for (m, n), D in designs.items() if D["b1_check"]}},
        "preds": {m: {"rows": w["rows"], "written": w["written"]} for m, w in written.items()},
        "preds_test_folds": {m: {"rows": w["rows"], "max_abs_mean_diff": w["max_abs_mean_diff"]} for m, w in written_folds.items()},
        "k_star": k_star,
        "discrimination_N10": disc.filter(pl.col("N") == 10).select("model", "split", "auroc", "pr_auc", "within_pair", "within_issue_mean").to_dicts(),
        "operating_points_N10": ops_df.filter((pl.col("N") == 10) | (pl.col("model") == "b0_cap")).select(
            "model", "policy", "N", "tau", "cap", "split", "stop_rate", "lost_success", "lost_success_lo", "lost_success_hi",
            *evaluate.LOSS_FLAG_COLUMNS, "saved_steps", "saved_tokens", "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi").to_dicts(),
        "judged_and_saves_more_than_k_star": judged,
        "test_loss_ci_width_m1": None if args.skip_bootstrap else evaluate.loss_ci_width(ops_df, k_star, models=MODEL_SPECS),
        "loss_ci_width_summary": compare["width_summary"].to_dicts(),
        "b1_input_sensitivity_summary": s2.to_dicts(),
        "good_n": [r for r in good_n if r["selected"]],
        "shap_top1": shap_top,
        "shap_stability": shap_stab,
        "bootstrap": None if args.skip_bootstrap else {"n_boot": args.n_boot, "tasks": len(boots), "seed": io.SEED},
    }
    crit = {
        "C1_features_one_to_one_finite": set(feat_counts) == set(evaluate.N_GRID),  # check_features가 실패 시 이미 멈춘다
        "C2_grid_no_seen_issue": True,  # main에서 assert
        "C3_selected_refit_reproduces_scores": refit_same,
        "C4_k_star": k_star == 40,
        "C5_b1_test_input_differs_by_fold": all(D["b1_check"]["test_fold_input_max_spread"] > 0 for D in designs.values() if D["b1_check"]),
        "C6_preds_and_fold_files": all(w["max_abs_mean_diff"] is not None and w["max_abs_mean_diff"] <= preds.FOLD_MEAN_TOL for w in written_folds.values()),
        "C7_operating_points": m1_test.height == len(MODEL_SPECS) * len(evaluate.N_GRID) * len(evaluate.POLICIES)
        and all(m1_test[f"{c}_fold_min"].null_count() == 0 for c in evaluate.POINT_COLUMNS)
        and (args.skip_bootstrap or (m1_test["lost_success_hi"].null_count() == 0 and m1_test["meets_judge_ci_hi"].null_count() == 0)),
        "C8_cap_rows_match_phase4": args.skip_bootstrap or compare["cap_rows_match_phase4"],
        "C9_b1_sensitivity": sens2.height > 0 and sens2["sens_2"].null_count() == 0,
        "C10_shap": args.skip_shap or (len(shap_stab) == 2 * len(SHAP_N) and len(list((out_dir / "figures").glob("*.png"))) == 2 * len(SHAP_N) * (1 + DEPENDENCE_TOP)),
    }
    s["criteria"] = crit
    s["criteria_note"] = "pytest 전체 통과(누수 테스트 포함)와 재실행 시 results/·preds/ 변화 없음은 스크립트 밖에서 확인한다. 실행 시간은 표준 출력에만 남긴다"
    io.write_summary(out_dir / "summary.json", s)

    with pl.Config(tbl_rows=60, tbl_cols=14, tbl_width_chars=220, float_precision=4):
        print(compare["discrimination"].filter((pl.col("N") == 10)).select("model", "split", "auroc", "pr_auc", "within_pair", "within_issue_mean"))
        print(compare["operating"].filter(((pl.col("N") == 10) | (pl.col("model") == "b0_cap")) & (pl.col("split") == "test")).select(
            "model", "policy", "tau", "cap", "stop_rate", "lost_success", "lost_success_hi", "meets_judge_ci_hi", "saved_steps",
            "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi"))
        print(compare["width_summary"])
    print(json.dumps({"criteria": crit, "timings": timings, "good_n": s["good_n"], "shap_top1": shap_top}, ensure_ascii=False, indent=2, default=str))
    failed = [k for k, v in crit.items() if not v]
    assert not failed, failed
    print("\n완료 기준 통과")


if __name__ == "__main__":
    main()
