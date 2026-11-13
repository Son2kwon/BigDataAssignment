"""Phase 5b: M1 개선 보조 실험 — 마지막 k스텝 창 피처 (PLAN "Phase 5b~5e", 11절 P5b-D1~D11, docs/phase5b.md)

실행:
  uv run --no-sync python code/04e_m1_window.py [--jobs 24] [--n-boot 1000] [--skip-bootstrap]
  uv run --no-sync python code/04e_m1_window.py --smoke     # 점검: train 이슈 1/10·그리드 2개·부트스트랩 20회, 결과는 data/processed/phase5b_smoke/

**train OOF만 쓴다.** test 궤적은 피처도 만들지 않고 채점하지 않는다. 5b~5e의 test 평가(--final)는 5e를 마친 뒤
한 번만 따로 실행한다 (P5b-D1). 공식 M1(Phase 5) 결과는 바꾸지 않는다.

묶음 (P5b-D7, 이 순서가 보고 순서이고 피처 수가 같을 때의 동률 순서):
  ① official = 공식 M1 피처 73개 / ② w3 = ① + 창 k=3 / ③ w5 = ① + 창 k=5 / ④ w35 = ① + 창 k=3·5 / ⑤ w35_rec = ④ + 최근성 3개
  확인용 (P5b-D8, 선택 후보 아님): wprop = ① + 창 k=⌈N/4⌉ (k=1·k ≥ N이면 창 없음)
  창 피처는 afp/features.py window_features (k마다 7개, k ≥ N이면 없음). N=3은 창이 없어서 ②~④가 ①과 같다 (같은 열이면 한 번만 학습)
모델 (P5b-D2): m1_xgb, m1_xgb_b1 (04_m1.py의 학습기·XGB_GRID 8개·GRID_TIE 선택 규칙 그대로). B1 입력은 train OOF b1_n (P4-D7)
S1 선택 (P5b-D3): m1_xgb의 train OOF 이슈 내 AUROC(쌍 가중) N∈{8, 10, 15, 20} 평균 최대. 0.002 안이면 피처가 적은 묶음
운영점 (train OOF): 공용 evaluate 함수로 고른다 (손실 ≤ 3.5%, 단독·결합). 구간은 train 이슈 단위 부트스트랩
(선택 불확실성은 들어 있지 않다). 짝지은 비교 = 같은 재표본에서 묶음 − ①(같은 모델)

입력: data/processed/ (01·02), preds/b1_n.parquet (03), features/m1_n.parquet (04_m1, 같은 피처인지 확인), preds/m1_xgb*.parquet (①이
이 머신의 공식 M1 OOF와 같은지 확인)
출력: results/phase5b/ (표·요약, git 안), data/processed/phase5b/ (묶음별 OOF 점수, git 밖)
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
from sklearn.metrics import roc_auc_score

from afp import evaluate, features, io, metrics, oof, preds

m1 = importlib.import_module("04_m1")  # 공식 M1의 학습기·그리드·선택 규칙을 그대로 쓴다

MODELS = {"m1_xgb": False, "m1_xgb_b1": True}  # model → B1 점수 포함
BUNDLES = {  # 이름 → (표기, N → 창 k 목록, 최근성). 선택 후보
    "official": ("①", lambda n: (), False),
    "w3": ("②", lambda n: (3,), False),
    "w5": ("③", lambda n: (5,), False),
    "w35": ("④", lambda n: (3, 5), False),
    "w35_rec": ("⑤", lambda n: (3, 5), True),
}
CHECK_BUNDLES = {"wprop": ("비례 k", lambda n: (features.proportional_k(n),), False)}  # P5b-D8 확인용
ALL_BUNDLES = {**BUNDLES, **CHECK_BUNDLES}
BASE = "official"
SELECT_MODEL = "m1_xgb"  # S1을 고르는 모델 ([결정 필요] docs/phase5b.md). m1_xgb_b1의 선택은 같이 보고한다
SELECT_N = (8, 10, 15, 20)
SELECT_TIE = 0.002
PROP_TIE = 0.002  # 비례 k가 고정 k와 이보다 크게 다르면 [결정 필요]
SMOKE_GRID = 2
FLOAT = 6
OUT_DIR = io.RESULTS_DIR / "phase5b"
DATA_DIR = io.PROCESSED_DIR / "phase5b"
SMOKE_DIR = io.PROCESSED_DIR / "phase5b_smoke"
REFERENCE = (  # 다른 Phase의 train OOF 판별 (읽기만, 참고 표)
    ("phase5", io.RESULTS_DIR / "phase5" / "discrimination.csv", ("m1_xgb", "m1_xgb_b1")),
    ("phase6", io.RESULTS_DIR / "phase6" / "discrimination.csv", ("m2_lstm", "m2_lstm_b1")),
    ("phase6_supp", io.RESULTS_DIR / "phase6" / "supplementary" / "discrimination.csv", ("m2x_nocum", "m2x_nocum_b1")),
    ("phase7", io.RESULTS_DIR / "phase7" / "discrimination.csv", ("m3_bert", "m3_bert_b1")),
)


def bundle_cols(bundle: str, n: int, use_b1: bool) -> list[str]:
    _, ks, rec = ALL_BUNDLES[bundle]
    return [*features.FEATURES, *features.window_columns(n, ks(n)), *(features.RECENCY_FEATURES if rec else []),
            *([features.B1_FEATURE] if use_b1 else [])]


def n_features(bundle: str, n: int = max(evaluate.N_GRID)) -> int:
    return len(bundle_cols(bundle, n, False))


# ---- 피처 (train만) ----

def build_features(train: pl.DataFrame) -> tuple[dict, dict]:
    """N → (traj_id, 공식 피처, 창 피처(k=3, 5, 비례 k 중 k < N), 최근성). 공식 피처가 04_m1.py 캐시와 같은지도 본다."""
    trajs = train.select("traj_id", "instance_id", "n_steps")
    values = features.load_step_values(max(evaluate.N_GRID), trajs)
    t = io.load_trajectories(["traj_id", "instance_id", "issue_text"]).join(trajs.select("traj_id"), on="traj_id", how="semi")
    base = features.m1_features(values, trajs, evaluate.N_GRID, features.mentions_table(t))
    features.check_features(base, trajs, evaluate.N_GRID)
    cache_same = None
    if m1.FEATURES_PATH.exists():
        cache = pl.read_parquet(m1.FEATURES_PATH).join(trajs.select("traj_id"), on="traj_id", how="semi").sort("N", "traj_id")
        cache_same = bool(cache.equals(base))
    out = {}
    for n in evaluate.N_GRID:
        ks = {*features.WINDOW_KS, features.proportional_k(n)}
        extra = features.window_features(values, trajs, n, ks=ks, recency=True)
        f = base.filter(pl.col("N") == n).join(extra, on="traj_id", how="left")
        cols = [c for c in f.columns if c not in ("traj_id", "N")]
        features.check_features(f, trajs, [n], columns=cols)
        out[n] = f
    return out, {"official_matches_04_m1_cache": cache_same, "rows_per_N": {str(n): f.height for n, f in out.items()}}


def design(train: pl.DataFrame, feats: dict, b1: pl.DataFrame | None, n: int, cols: list[str]) -> dict:
    """04_m1.design의 train 부분과 같은 행 순서(traj_id)·열 순서. 공식 묶음이면 같은 행렬이 나온다."""
    t = metrics.running(train, n).select("traj_id", "instance_id", "target", "fold").sort("traj_id")
    t = t.join(feats[n], on="traj_id", how="left")
    if b1 is not None:
        t = t.join(b1.filter(pl.col("N") == n).select("traj_id", pl.col("p_fail").alias(features.B1_FEATURE)), on="traj_id", how="left")
    x = t.select(cols)
    if x.null_count().sum_horizontal().item():
        raise AssertionError(f"N={n}: 피처·B1이 없는 진행 중 궤적")
    return {"traj_id": t["traj_id"], "X": x.to_numpy(), "y": (~t["target"]).to_numpy(), "inst": t["instance_id"].to_numpy(),
            "fold": t["fold"].to_numpy()}


# ---- 학습 (train OOF만) ----

def oof_task(key, cfg, X, y, inst, fold):
    rows = pl.DataFrame({"instance_id": inst, "fold": fold, "_i": np.arange(len(y))})

    def fit(r):
        idx = r["_i"].to_numpy()
        return m1.fit_model("xgb", cfg, X[idx], y[idx])

    def predict(model, r):
        return m1.predict_model(model, X[r["_i"].to_numpy()])

    res = oof.oof_predict(rows, fit, predict)
    p = res.train["p_fail"].to_numpy()
    return {"key": key, "oof": p, "oof_auroc": float(roc_auc_score(y, p)), "seen_issue_scores": res.seen_issue_scores}


# ---- 운영점·부트스트랩 (train) ----

def ops_task(bundle, model, n, train, p, costs_train, k_star):
    """train OOF 곡선에서 단독·결합 운영점 (공식과 같은 선택 함수)과 K* 대비 차이, fold별 범위."""
    ref = evaluate.cap_point(train, k_star, costs_train)
    out = []
    for policy in evaluate.POLICIES:
        op = metrics.select_from_curves(evaluate.candidate_curves(train, p, n, policy, costs_train), n)
        pt = {c: op[c] for c in evaluate.POINT_COLUMNS}
        out.append({"bundle": bundle, "model": model, "policy": policy, "N": n, "tau": op["tau"], "cap": op["cap"], **pt,
                    **{f"d_{c}": pt[c] - ref[c] for c in evaluate.DELTA_COLUMNS},
                    **evaluate.fold_spread(train, p, n, op["tau"], op["cap"], costs_train)})
    return out


def oof_stat(n, points_a, points_o, k_ref, costs):
    """부트스트랩 stat: 묶음 a의 판별·운영점, K* 대비 차이(dk_), 같은 모델 ①(o) 대비 차이(do_). 같은 재표본."""
    def stat(trajs, p):
        a, o = evaluate.unpair(p)
        da = evaluate.discrimination_row(trajs, a, n)
        do = evaluate.discrimination_row(trajs, o, n)
        out = {}
        for c in evaluate.PAIR_DISC:
            out[c] = da[c]
            out[f"do_{c}"] = da[c] - do[c]
        ref = evaluate.cap_point(trajs, k_ref, costs)
        for pol, (tau, cap) in points_a.items():
            pa = evaluate.apply_point(trajs, a, n, tau, cap, costs)
            po = evaluate.apply_point(trajs, o, n, *points_o[pol], costs)
            for c in evaluate.POINT_COLUMNS:
                out[f"{pol}.{c}"] = pa[c]
                out[f"{pol}.do_{c}"] = pa[c] - po[c]
                if c != "stop_rate":
                    out[f"{pol}.dk_{c}"] = pa[c] - ref[c]
        return out
    return stat


def boot_task(key, train, p_pair, stat, n_boot):
    return {"key": key, "boot": metrics.bootstrap(train, p_pair, stat, n_boot=n_boot)}


# ---- 출력 ----

def write_parquet(df: pl.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and pl.read_parquet(path).equals(df):
        return
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(df.to_arrow(), tmp, compression="zstd")
    tmp.replace(path)


def feature_dist(train: pl.DataFrame, feats: dict) -> pl.DataFrame:
    """새 피처의 train 분포 (N마다): 평균·0이 아닌 비율·성공·실패 평균 (해석용)."""
    rows = []
    for n, f in feats.items():
        f = f.join(train.select("traj_id", "target"), on="traj_id", how="inner")
        succ = f["target"].to_numpy()
        for c in [c for c in f.columns if c not in ("traj_id", "N", "target", *features.FEATURES)]:
            v = f[c].to_numpy()
            rows.append({"N": n, "feature": c, "n": len(v), "mean": float(v.mean()), "std": float(v.std()), "nonzero": float((v != 0).mean()),
                         "mean_success": float(v[succ].mean()), "mean_failure": float(v[~succ].mean())})
    return pl.DataFrame(rows)


def reference_table(disc: pl.DataFrame) -> pl.DataFrame:
    """이번 실행의 ①·각 묶음과 다른 Phase의 train OOF 판별 (참고, 같은 train 이슈·같은 지표 함수)."""
    rows = [disc.select(pl.lit("phase5b").alias("source"), pl.concat_str(pl.col("model"), pl.lit(":"), pl.col("bundle")).alias("model"),
                        "N", "auroc", "within_pair", "within_issue_mean")]
    for src, path, models in REFERENCE:
        if path.exists():
            d = pl.read_csv(path, schema_overrides={"N": pl.Int64})
            d = d.filter((pl.col("split") == "train_oof") & pl.col("model").is_in(list(models)))
            rows.append(d.select(pl.lit(src).alias("source"), "model", "N", "auroc", "within_pair", "within_issue_mean"))
    return pl.concat(rows, how="vertical_relaxed").sort("N", "source", "model")


def main():
    ap = argparse.ArgumentParser(description="Phase 5b: M1 창 피처 (train OOF만)")
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    ap.add_argument("--skip-bootstrap", action="store_true", help="부트스트랩 구간을 건너뜀 (개발용, 완료 기준 C6 불충족)")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--smoke", action="store_true", help="train 이슈 1/10, 그리드 앞 2개, 부트스트랩 20회. results/를 쓰지 않는다")
    args = ap.parse_args()
    for p, script in ((io.TRAJECTORIES_PATH, "01_parse.py"), (io.STEPS_PATH, "01_parse.py"), (io.SPLITS_PATH, "02_split.py"),
                      (preds.path(m1.B1_MODEL), "03_baselines.py")):
        if not p.exists():
            sys.exit(f"{p}가 없다. 먼저 uv run --no-sync python code/{script} 를 실행한다.")
    t0 = time.time()
    timings = {}

    def lap(name):
        timings[name] = round(time.time() - t0 - sum(timings.values()), 1)
        print(f"[{timings[name]:>7.1f}s] {name}", flush=True)

    out_dir, data_dir = (SMOKE_DIR, SMOKE_DIR / "oof") if args.smoke else (OUT_DIR, DATA_DIR)
    n_boot = 20 if args.smoke else args.n_boot
    grid = m1.XGB_GRID[:SMOKE_GRID] if args.smoke else m1.XGB_GRID
    train = evaluate.load_eval_sets()["train"]  # test는 읽지 않는다
    if args.smoke:
        issues = train["instance_id"].unique().sort()
        train = train.join(pl.DataFrame({"instance_id": issues.gather(np.arange(0, len(issues), 10))}), on="instance_id", how="semi")
    costs_train = evaluate.load_costs({"train": train})["train"]
    cap_train = metrics.fixed_cap_curve(train, costs_train, range(1, 81))
    k_star = metrics.select_cap(cap_train, metrics.SELECT_MAX_LOSS)["K"]
    b1_train = preds.load_feature(m1.B1_MODEL, "train").join(train.select("traj_id"), on="traj_id", how="semi")
    lap(f"입력 (train 궤적 {train.height}, 이슈 {train['instance_id'].n_unique()}, K*={k_star})")

    # 1. 피처
    feats, feat_info = build_features(train)
    lap(f"피처 (N별 행 {feat_info['rows_per_N']})")

    # 2. 그리드 OOF. 같은 (모델, N, 열)은 한 번만 학습한다 (N=3의 ②~④ = ① 등)
    cols_of, key_of, designs = {}, {}, {}
    for b in ALL_BUNDLES:
        for m, use_b1 in MODELS.items():
            for n in evaluate.N_GRID:
                cols = bundle_cols(b, n, use_b1)
                key = (m, n, tuple(cols))
                cols_of[(b, m, n)], key_of[(b, m, n)] = cols, key
                if key not in designs:
                    designs[key] = design(train, feats, b1_train if use_b1 else None, n, cols)
    keys = list(designs)
    tasks = [((j, i), cfg, designs[k]["X"], designs[k]["y"], designs[k]["inst"], designs[k]["fold"])
             for j, k in enumerate(keys) for i, cfg in enumerate(grid)]
    tasks.sort(key=lambda t: -t[1]["n_estimators"] * t[1]["max_depth"])  # 무거운 작업부터 (결과는 순서와 무관)
    res = {r["key"]: r for r in evaluate.parallel(oof_task, tasks, args.jobs)}
    assert all(r["seen_issue_scores"] == 0 for r in res.values())
    selected = {}
    for j, k in enumerate(keys):
        rs = [{"i": i, **res[(j, i)]} for i in range(len(grid))]
        selected[k] = rs[m1.choose(rs)]
    grid_rows = [{"bundle": b, "model": m, "N": n, "n_features": len(cols_of[(b, m, n)]), "i": i, "config": m1.cfg_str(grid[i]),
                  "oof_auroc": res[(keys.index(key_of[(b, m, n)]), i)]["oof_auroc"], "selected": i == selected[key_of[(b, m, n)]]["i"]}
                 for (b, m, n) in key_of for i in range(len(grid))]
    grid_df = pl.DataFrame(grid_rows)
    lap(f"그리드 OOF (설계 {len(keys)}개 × 설정 {len(grid)}개 = {len(tasks)}개)")

    # 3. 묶음별 OOF 점수와 판별 지표
    oof_p = {}
    for (b, m, n), key in key_of.items():
        D = designs[key]
        oof_p.setdefault((b, m), []).append(pl.DataFrame({"traj_id": D["traj_id"], "N": np.full(len(D["y"]), n, dtype=np.int16),
                                                          "p_fail": selected[key]["oof"]}))
    oof_p = {k: pl.concat(v).sort("N", "traj_id") for k, v in oof_p.items()}
    for (b, m), p in oof_p.items():
        write_parquet(p, data_dir / f"{m}__{b}.parquet")
    disc = pl.concat([evaluate.discrimination_table(m, p, {"train": train}, evaluate.N_GRID).with_columns(bundle=pl.lit(b))
                      for (b, m), p in oof_p.items()])
    disc = disc.with_columns(n_features=pl.struct("bundle", "N").map_elements(lambda s: n_features(s["bundle"], s["N"]), return_dtype=pl.Int64))
    disc = disc.select("bundle", "model", "N", "n_features", *[c for c in disc.columns if c not in ("bundle", "model", "N", "n_features", "split")])
    # ① = 이 머신의 공식 M1 OOF (04_m1.py 결과 파일)인지
    repro = {}
    for m in MODELS:
        if args.smoke or not preds.path(m).exists():
            repro[m] = None
            continue
        ref = preds.read(m).join(train.select("traj_id"), on="traj_id", how="semi").sort("N", "traj_id")
        mine = oof_p[(BASE, m)]
        same_rows = ref.select("traj_id", "N").equals(mine.select("traj_id", pl.col("N").cast(ref["N"].dtype)))
        repro[m] = {"same_rows": bool(same_rows),
                    "max_abs_diff": float(np.abs(ref["p_fail"].to_numpy() - mine["p_fail"].to_numpy()).max()) if same_rows else None}
    lap("OOF 점수·판별 지표")

    # 4. train 운영점
    otasks = [(b, m, n, train, oof_p[(b, m)].filter(pl.col("N") == n), costs_train, k_star)
              for (b, m) in oof_p for n in evaluate.N_GRID]
    op_rows = [r for out in evaluate.parallel(ops_task, otasks, args.jobs) for r in out]
    chosen = {(r["bundle"], r["model"], r["N"], r["policy"]): (r["tau"], r["cap"]) for r in op_rows}
    lap(f"train 운영점 ({len(otasks)}개)")

    # 5. 부트스트랩 (train 이슈 단위, 모든 작업이 같은 재표본: 시드 io.SEED, 같은 train 이슈)
    boots = {}
    if not args.skip_bootstrap:
        btasks = []
        for (b, m), p in oof_p.items():
            for n in evaluate.N_GRID:
                pa, po = p.filter(pl.col("N") == n), oof_p[(BASE, m)].filter(pl.col("N") == n)
                pts_a = {pol: chosen[(b, m, n, pol)] for pol in evaluate.POLICIES}
                pts_o = {pol: chosen[(BASE, m, n, pol)] for pol in evaluate.POLICIES}
                c = evaluate.compressed_costs(costs_train, n, [cap for _, cap in (*pts_a.values(), *pts_o.values())], k_star)
                btasks.append(((b, m, n), train, evaluate.pair_preds(pa, po), oof_stat(n, pts_a, pts_o, k_star, c), n_boot))
        boots = {r["key"]: r["boot"] for r in evaluate.parallel(boot_task, btasks, args.jobs)}
        lap(f"부트스트랩 ({len(btasks)}개 × {n_boot}회)")

    def ci(key, stat):
        if key not in boots:
            return (None, None)
        r = boots[key].filter(pl.col("stat") == stat)
        return (r["lo"][0], r["hi"][0])

    # 판별 표에 구간 (묶음의 쌍 가중 이슈 내 AUROC, ① 대비 차이)
    drows = []
    for r in disc.iter_rows(named=True):
        k = (r["bundle"], r["model"], r["N"])
        extra = {}
        for c in ("auroc", "within_pair"):
            extra[f"{c}_lo"], extra[f"{c}_hi"] = ci(k, c)
        ref = disc.filter((pl.col("bundle") == BASE) & (pl.col("model") == r["model"]) & (pl.col("N") == r["N"])).row(0, named=True)
        for c in ("auroc", "within_pair", "within_issue_mean"):
            extra[f"d_{c}"] = r[c] - ref[c]
        for c in ("auroc", "within_pair"):
            extra[f"d_{c}_lo"], extra[f"d_{c}_hi"] = ci(k, f"do_{c}")
        drows.append({**r, **extra})
    disc_df = pl.DataFrame(drows, infer_schema_length=None)

    # 운영점 표: K* 대비 차이 [95%], ① 대비 차이 [95%]
    orows = []
    for r in op_rows:
        k, pol = (r["bundle"], r["model"], r["N"]), r["policy"]
        e = {}
        for c in ("lost_success", "saved_steps", "saved_tokens"):
            e[f"{c}_lo"], e[f"{c}_hi"] = ci(k, f"{pol}.{c}")
            e[f"d_{c}_lo"], e[f"d_{c}_hi"] = ci(k, f"{pol}.dk_{c}")
        ref = next(x for x in op_rows if x["bundle"] == BASE and x["model"] == r["model"] and x["N"] == r["N"] and x["policy"] == pol)
        for c in ("lost_success", "saved_steps", "saved_tokens"):
            e[f"do_{c}"] = r[c] - ref[c]
            e[f"do_{c}_lo"], e[f"do_{c}_hi"] = ci(k, f"{pol}.do_{c}")
        orows.append({**r, **e})
    cap_rows = []
    ref_k = evaluate.cap_point(train, k_star, costs_train)
    for policy, kk in (("cap", k_star), ("cap_proposal", evaluate.K_PROPOSAL)):
        pt = evaluate.cap_point(train, kk, costs_train)
        cap_rows.append({"bundle": None, "model": "b0_cap", "policy": policy, "N": None, "tau": None, "cap": kk, **pt,
                         **{f"d_{c}": pt[c] - ref_k[c] for c in evaluate.DELTA_COLUMNS},
                         **evaluate.fold_spread(train, None, None, None, kk, costs_train)})
    op_cols = ["bundle", "model", "policy", "N", "tau", "cap", "stop_rate", "lost_success", "lost_success_lo", "lost_success_hi",
               "saved_steps", "saved_steps_lo", "saved_steps_hi", "saved_tokens", "saved_tokens_lo", "saved_tokens_hi",
               "d_lost_success", "d_lost_success_lo", "d_lost_success_hi", "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi",
               "d_saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi",
               "do_lost_success", "do_lost_success_lo", "do_lost_success_hi", "do_saved_steps", "do_saved_steps_lo", "do_saved_steps_hi",
               "do_saved_tokens", "do_saved_tokens_lo", "do_saved_tokens_hi",
               "fold_lost_success_min", "fold_lost_success_max", "fold_saved_steps_min", "fold_saved_steps_max"]
    ops_df = evaluate.frame(cap_rows + orows, op_cols, {**evaluate.OP_SCHEMA, "bundle": pl.String})
    ops_df = ops_df.with_columns(meets_select_point=pl.col("lost_success") <= metrics.SELECT_MAX_LOSS + metrics.LOSS_EPS)

    # 6. S1 선택 (P5b-D3) + 비례 k 확인 (P5b-D8)
    def sel_metric(b, m):
        d = disc_df.filter((pl.col("bundle") == b) & (pl.col("model") == m) & pl.col("N").is_in(list(SELECT_N)))
        return float(d["within_pair"].mean())

    def pick(m):
        score = {b: sel_metric(b, m) for b in BUNDLES}
        best = max(score.values())
        cand = [b for b in BUNDLES if score[b] >= best - SELECT_TIE]
        return min(cand, key=lambda b: (n_features(b), list(BUNDLES).index(b))), score, best

    s1, score, best = pick(SELECT_MODEL)
    alt = {m: pick(m)[0] for m in MODELS}
    sel_rows = []
    for m in MODELS:
        sc, bst = pick(m)[1], pick(m)[2]
        for b in BUNDLES:
            row = {"model": m, "bundle": b, "label": BUNDLES[b][0], "n_features_N20": n_features(b),
                   **{f"within_pair_N{n}": disc_df.filter((pl.col("bundle") == b) & (pl.col("model") == m) & (pl.col("N") == n))["within_pair"][0]
                      for n in evaluate.N_GRID},
                   "select_mean": sc[b], "d_select_mean_vs_official": sc[b] - sc[BASE], "in_tie_band": sc[b] >= bst - SELECT_TIE,
                   "picked_by_this_model": b == alt[m], "selected_S1": m == SELECT_MODEL and b == s1}
            sel_rows.append(row)
    sel_df = pl.DataFrame(sel_rows)
    prop_rows = []
    for m in MODELS:
        fixed = max(("w3", "w5"), key=lambda b: sel_metric(b, m))
        d = sel_metric("wprop", m) - sel_metric(fixed, m)
        prop_rows.append({"model": m, "wprop_select_mean": sel_metric("wprop", m), "best_fixed_single": fixed,
                          "best_fixed_select_mean": sel_metric(fixed, m), "d_wprop_minus_fixed": d,
                          "d_wprop_minus_S1": sel_metric("wprop", m) - sel_metric(s1, m), "exceeds_tie": abs(d) > PROP_TIE})
    prop_df = pl.DataFrame(prop_rows)
    # 가설 (P5b-D9): S1은 N ≥ 8에서 ①보다 OOF 이슈 내 AUROC가 높다
    hyp = [{"model": m, "N": n, **{c: r[c] for c in ("d_within_pair", "d_within_pair_lo", "d_within_pair_hi")}}
           for m in MODELS for n in SELECT_N
           for r in disc_df.filter((pl.col("bundle") == s1) & (pl.col("model") == m) & (pl.col("N") == n)).iter_rows(named=True)]
    lap(f"S1 선택: {s1} ({BUNDLES[s1][0]})")

    # 7. 저장
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_df.write_csv(out_dir / "grid.csv", float_precision=FLOAT)
    disc_df.write_csv(out_dir / "discrimination_oof.csv", float_precision=FLOAT)
    ops_df.write_csv(out_dir / "operating_points_oof.csv", float_precision=FLOAT)
    sel_df.write_csv(out_dir / "selection.csv", float_precision=FLOAT)
    prop_df.write_csv(out_dir / "prop_k_check.csv", float_precision=FLOAT)
    feature_dist(train, feats).write_csv(out_dir / "feature_dist.csv", float_precision=FLOAT)
    if not args.smoke:
        reference_table(disc_df).write_csv(out_dir / "oof_reference.csv", float_precision=FLOAT)

    n_designs_unique = len(keys)
    crit = {
        "C1_features_finite_and_official_same_as_04_m1": args.smoke or feat_info["official_matches_04_m1_cache"] is not False,
        "C2_no_seen_issue_scores": True,  # 위 assert
        "C3_official_reproduces_04_m1_oof": args.smoke or all(r is not None and r["same_rows"] and r["max_abs_diff"] == 0 for r in repro.values()),
        "C4_k_star": args.smoke or k_star == 40,
        "C5_tables_complete": disc_df.height == len(ALL_BUNDLES) * len(MODELS) * len(evaluate.N_GRID)
        and ops_df.filter(pl.col("model") != "b0_cap").height == len(ALL_BUNDLES) * len(MODELS) * len(evaluate.N_GRID) * len(evaluate.POLICIES),
        "C6_bootstrap_ci": (not args.skip_bootstrap) and ops_df.filter(pl.col("model") != "b0_cap")["d_saved_tokens_lo"].null_count() == 0,
    }
    summary = {
        "mode": "smoke" if args.smoke else "train OOF only (test not read)",
        "bundles": {b: {"label": v[0], "n_features_by_N": {str(n): n_features(b, n) for n in evaluate.N_GRID}} for b, v in ALL_BUNDLES.items()},
        "models": list(MODELS), "grid": [dict(c) for c in grid], "xgb_fixed": m1.XGB_FIXED, "grid_tie": m1.GRID_TIE,
        "k_star": k_star, "features": feat_info, "reproduce_official_oof": repro,
        "unique_designs": n_designs_unique, "grid_fits": len(tasks),
        "selection": {"rule": "P5b-D3", "model": SELECT_MODEL, "N": list(SELECT_N), "tie": SELECT_TIE,
                      "scores": score, "best": best, "S1": s1, "S1_label": BUNDLES[s1][0], "pick_by_model": alt,
                      "models_agree": len(set(alt.values())) == 1},
        "candidates_seen": {"bundles": len(BUNDLES), "check_bundles": len(CHECK_BUNDLES),
                            "grid_configs_per_bundle_model_N": len(grid)},
        "prop_k_check": prop_rows,
        "hypothesis_P5b_D9": {"rows": hyp, "n_positive": sum(r["d_within_pair"] > 0 for r in hyp),
                              "n_ci_lo_positive": sum((r["d_within_pair_lo"] or -1) > 0 for r in hyp), "n": len(hyp)},
        "bootstrap": None if args.skip_bootstrap else {"n_boot": n_boot, "seed": io.SEED, "unit": "train issue",
                                                       "note": "운영점 선택은 다시 하지 않는다 (선택 불확실성 미포함)"},
        "criteria": crit,
        "criteria_note": "pytest 전체 통과와 재실행 시 results/phase5b/ 바이트 동일은 스크립트 밖에서 확인한다. 실행 시간은 표준 출력에만 남긴다",
    }
    io.write_summary(out_dir / "summary.json", summary)

    with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=250, float_precision=4):
        print(sel_df.select("model", "bundle", "label", "n_features_N20", *[f"within_pair_N{n}" for n in evaluate.N_GRID],
                            "select_mean", "d_select_mean_vs_official", "in_tie_band", "picked_by_this_model"))
        print(disc_df.filter(pl.col("model") == SELECT_MODEL).select("bundle", "N", "auroc", "d_auroc", "within_pair", "d_within_pair",
                                                                    "d_within_pair_lo", "d_within_pair_hi"))
        print(ops_df.filter((pl.col("policy") == "combined") & (pl.col("model") == SELECT_MODEL) | (pl.col("model") == "b0_cap")).select(
            "bundle", "N", "tau", "cap", "lost_success", "saved_steps", "d_saved_steps", "saved_tokens", "d_saved_tokens",
            "d_saved_tokens_lo", "d_saved_tokens_hi", "do_saved_steps", "do_saved_tokens"))
        print(prop_df)
    print(json.dumps({"S1": s1, "pick_by_model": alt, "reproduce": repro, "criteria": crit, "timings": timings},
                     ensure_ascii=False, indent=2, default=str))
    failed = [k for k, v in crit.items() if not v]
    if failed:
        print(f"\n완료 기준 불충족: {failed}")
        sys.exit(1)
    print("\n완료 기준 통과")


if __name__ == "__main__":
    main()
