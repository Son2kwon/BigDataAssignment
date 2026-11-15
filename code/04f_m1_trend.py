"""Phase 5c: M1 개선 보조 실험 — N스텝과 k스텝 가중 (PLAN "Phase 5b~5e", 11절 P5c-D1~D8, docs/phase5c.md)

실행:
  uv run --no-sync python code/04f_m1_trend.py [--device cuda|cpu] [--gpu-jobs 4] [--jobs 24] [--n-boot 1000] [--skip-bootstrap]
  uv run --no-sync python code/04f_m1_trend.py --smoke     # 점검: train 이슈 1/10·그리드 2개·부트스트랩 20회, 결과는 data/processed/phase5c_smoke/

**train OOF만 쓴다.** test 궤적은 피처도 만들지 않는다 (P5b-D1, 5b.0). 학습·운영점·부트스트랩 함수는 04e_m1_window.py
(5b)를 그대로 쓴다 (장치 P5b-D15: 기본 cuda, 비교는 모두 같은 장치로 학습한 S1 대비).

출발점 S1 = 5b ② = 공식 M1 피처 73개 + 창 k=3 7개 (results/phase5b/summary.json의 S1이 w3인지 확인하고, 아니면 멈춘다).
S1은 이 스크립트가 같은 장치로 다시 학습한다. 5b의 OOF 파일(data/processed/phase5b/)이 있으면 같은지 확인만 한다.

묶음 (P5c-D5, 피처는 afp/features.py trend_features. k 집합은 S1을 따라 {3} (P5c-D1), 기울기만 k=5·3 (P5c-D8)):
  1 s1    = S1
  2 a{α}  = S1 + (a) 가중합 창 k=3, α ∈ {2, 4, 8} 중 OOF로 하나 (1단계 선택)
  3 chg   = S1 + (c) 변화율 창 k=3
  4 g{γ}  = S1 + (b) 지수 감쇠, γ ∈ {0.5, 0.7, 0.9} 중 OOF로 하나 (1단계 선택). N=3에도 있다
  5 sl5   = S1 + obs_len 기울기 k=5 (N ≥ 8)
  6 sl35  = S1 + 기울기 k=5·3 (k=3은 N ≥ 5)
  7 combo = S1 + (2와 3 중 나은 것) + (5와 6 중 나은 것, 0.002 안이면 5) (2단계 선택)
1단계 선택·S2 선택 규칙 (P5b-D3·D12, P5c-D9): m1_xgb의 train OOF 이슈 내 AUROC(쌍 가중) N∈{8, 10, 15, 20} 평균.
0.002 안이면 피처가 적은 쪽, 같으면 앞 번호 (α·γ는 작은 값, 2와 3이면 2). m1_xgb_b1의 선택도 같이 보고한다.
운영점·부트스트랩은 최종 묶음 7개만 한다. 고르지 않은 α·γ 후보는 그리드·판별 지표만 낸다 (P5c-D10).

입력: data/processed/ (01·02), preds/b1_n.parquet (03), features/m1_n.parquet (04_m1, 있으면 같은 피처인지 확인)
출력: results/phase5c/ (표·요약, git 안), data/processed/phase5c/ (최종 묶음별 OOF 점수, git 밖, 5d 출발점)
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import polars as pl

from afp import evaluate, features, io, metrics, preds

m1 = importlib.import_module("04_m1")  # 학습기·그리드·선택 규칙
m1w = importlib.import_module("04e_m1_window")  # 5b: 설계 행렬, 장치, OOF·운영점·부트스트랩 작업

MODELS = m1w.MODELS
S1_KS = (3,)  # S1(5b ②)의 창 k 집합 (P5c-D1)
S1_NAME_5B = "w3"
ALPHA_VARIANTS = {f"a{a}": a for a in features.ALPHAS}
GAMMA_VARIANTS = {features._gamma_tag(g): g for g in features.GAMMAS}
STAGE1 = ["s1", *ALPHA_VARIANTS, "chg", *GAMMA_VARIANTS, "sl5", "sl35"]
BASE = "s1"
SELECT_MODEL = m1w.SELECT_MODEL  # m1_xgb (P5b-D12)
SELECT_N = m1w.SELECT_N
SELECT_TIE = m1w.SELECT_TIE
SMOKE_GRID = m1w.SMOKE_GRID
FLOAT = m1w.FLOAT
OUT_DIR = io.RESULTS_DIR / "phase5c"
DATA_DIR = io.PROCESSED_DIR / "phase5c"
SMOKE_DIR = io.PROCESSED_DIR / "phase5c_smoke"


def extra_cols(variant: str, n: int, combo: tuple = ()) -> list[str]:
    """S1에 더하는 5c 열. combo = 7번 묶음을 이루는 변형 이름 2개."""
    if variant == "s1":
        return []
    if variant in ALPHA_VARIANTS:
        return features.alpha_columns(n, S1_KS, ALPHA_VARIANTS[variant])
    if variant == "chg":
        return features.change_columns(n, S1_KS)
    if variant in GAMMA_VARIANTS:
        return features.decay_columns(GAMMA_VARIANTS[variant])
    if variant == "sl5":
        return features.slope_columns(n, (5,))
    if variant == "sl35":
        return features.slope_columns(n, (3, 5))
    if variant == "combo":
        return [c for v in combo for c in extra_cols(v, n)]
    raise KeyError(variant)


def bundle_cols(variant: str, n: int, use_b1: bool, combo: tuple = ()) -> list[str]:
    return [*features.FEATURES, *features.window_columns(n, S1_KS), *extra_cols(variant, n, combo),
            *([features.B1_FEATURE] if use_b1 else [])]


def n_features(variant: str, n: int = max(evaluate.N_GRID), combo: tuple = ()) -> int:
    return len(bundle_cols(variant, n, False, combo))


# ---- 피처 (train만) ----

def build_features(train: pl.DataFrame) -> tuple[dict, dict]:
    """N → (traj_id, 공식 피처, 창 k=3, 5c 피처 전부). 공식 피처가 04_m1.py 캐시와 같은지도 본다."""
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
        win = features.window_features(values, trajs, n, ks=S1_KS, recency=False)
        trend = features.trend_features(values, trajs, n, ks=S1_KS, slope_ks=features.SLOPE_KS)
        f = base.filter(pl.col("N") == n).join(win, on="traj_id", how="left").join(trend, on="traj_id", how="left")
        features.check_features(f, trajs, [n], columns=[c for c in f.columns if c not in ("traj_id", "N")])
        out[n] = f
    return out, {"official_matches_04_m1_cache": cache_same, "rows_per_N": {str(n): f.height for n, f in out.items()}}


# ---- 선택 ----

def sel_metric(disc: pl.DataFrame, variant: str, model: str) -> float:
    d = disc.filter((pl.col("bundle") == variant) & (pl.col("model") == model) & pl.col("N").is_in(list(SELECT_N)))
    return float(d["within_pair"].mean())


def pick(disc: pl.DataFrame, cands: list[str], model: str, n_feat: dict, tie: float = SELECT_TIE) -> tuple[str, dict]:
    """선택 지표 최대에서 tie 안이면 피처가 적은 쪽, 같으면 cands 앞쪽 (P5b-D3, P5c-D9)."""
    score = {v: sel_metric(disc, v, model) for v in cands}
    best = max(score.values())
    band = [v for v in cands if score[v] >= best - tie]
    return min(band, key=lambda v: (n_feat[v], cands.index(v))), score


def feature_dist(train: pl.DataFrame, feats: dict) -> pl.DataFrame:
    """5c 새 피처의 train 분포 (N마다): 평균·표준편차·0이 아닌 비율·성공·실패 평균 (해석용)."""
    rows = []
    skip = {"traj_id", "N", "target", *features.FEATURES}
    for n, f in feats.items():
        f = f.join(train.select("traj_id", "target"), on="traj_id", how="inner")
        succ = f["target"].to_numpy()
        for c in [c for c in f.columns if c not in skip and c not in features.window_columns(n, S1_KS)]:
            v = f[c].to_numpy()
            rows.append({"N": n, "feature": c, "n": len(v), "mean": float(v.mean()), "std": float(v.std()), "nonzero": float((v != 0).mean()),
                         "mean_success": float(v[succ].mean()), "mean_failure": float(v[~succ].mean())})
    return pl.DataFrame(rows)


def s1_check() -> dict:
    """5b가 고른 S1이 이 스크립트의 S1(창 k=3)과 같은지. results/phase5b/summary.json 기준."""
    path = m1w.OUT_DIR / "summary.json"
    if not path.exists():
        sys.exit(f"{path}가 없다. 5b(04e_m1_window.py) 결과가 필요하다 (S1 확인).")
    s = json.loads(path.read_text())
    if s["selection"]["S1"] != S1_NAME_5B:
        sys.exit(f"5b S1 = {s['selection']['S1']}인데 이 스크립트는 S1 = {S1_NAME_5B}(창 k=3)로 짜여 있다. 멈춘다 (PLAN P9-D2 (2)).")
    return {"phase5b_S1": s["selection"]["S1"], "phase5b_device": s.get("device")}


def main():
    ap = argparse.ArgumentParser(description="Phase 5c: M1 가중·변화율·감쇠·기울기 피처 (train OOF만)")
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    ap.add_argument("--skip-bootstrap", action="store_true", help="부트스트랩 구간을 건너뜀 (개발용, 완료 기준 C6 불충족)")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1), help="CPU 작업 수 (운영점·부트스트랩, --device cpu면 학습도)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="XGBoost 학습 장치 (기본 cuda, P5b-D15)")
    ap.add_argument("--gpu-jobs", type=int, default=4, help="--device cuda일 때 동시에 학습하는 프로세스 수")
    ap.add_argument("--smoke", action="store_true", help="train 이슈 1/10, 그리드 앞 2개, 부트스트랩 20회. results/를 쓰지 않는다")
    args = ap.parse_args()
    for p, script in ((io.TRAJECTORIES_PATH, "01_parse.py"), (io.STEPS_PATH, "01_parse.py"), (io.SPLITS_PATH, "02_split.py"),
                      (preds.path(m1.B1_MODEL), "03_baselines.py")):
        if not p.exists():
            sys.exit(f"{p}가 없다. 먼저 uv run --no-sync python code/{script} 를 실행한다.")
    s1_info = s1_check()
    m1w.check_device(args.device)
    train_jobs = args.gpu_jobs if args.device == "cuda" else args.jobs
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

    feats, feat_info = build_features(train)
    lap(f"피처 (N별 행 {feat_info['rows_per_N']})")

    # ---- 그리드 OOF (같은 (모델, N, 열)은 한 번만 학습) ----
    designs, results, selected = {}, {}, {}
    cols_of, key_of = {}, {}

    def run_grid(variants, combo=()):
        new = []
        for v in variants:
            for m, use_b1 in MODELS.items():
                for n in evaluate.N_GRID:
                    cols = bundle_cols(v, n, use_b1, combo)
                    key = (m, n, tuple(cols))
                    cols_of[(v, m, n)], key_of[(v, m, n)] = cols, key
                    if key not in designs:
                        designs[key] = m1w.design(train, feats, b1_train if use_b1 else None, n, cols)
                        new.append(key)
        tasks = [((key, i), cfg, designs[key]["X"], designs[key]["y"], designs[key]["inst"], designs[key]["fold"], args.device)
                 for key in new for i, cfg in enumerate(grid)]
        tasks.sort(key=lambda t: -t[1]["n_estimators"] * t[1]["max_depth"])  # 무거운 작업부터 (결과는 순서와 무관)
        for r in evaluate.parallel(m1w.oof_task, tasks, train_jobs):
            assert r["seen_issue_scores"] == 0
            results[r["key"]] = r
        for key in new:
            rs = [{"i": i, **results[(key, i)]} for i in range(len(grid))]
            selected[key] = rs[m1.choose(rs)]
        return len(new), len(tasks)

    def oof_scores(variants):
        out = {}
        for v in variants:
            for m in MODELS:
                parts = []
                for n in evaluate.N_GRID:
                    D, key = designs[key_of[(v, m, n)]], key_of[(v, m, n)]
                    parts.append(pl.DataFrame({"traj_id": D["traj_id"], "N": np.full(len(D["y"]), n, dtype=np.int16), "p_fail": selected[key]["oof"]}))
                out[(v, m)] = pl.concat(parts).sort("N", "traj_id")
        return out

    def disc_of(p_by):
        return pl.concat([evaluate.discrimination_table(m, p, {"train": train}, evaluate.N_GRID).with_columns(bundle=pl.lit(v))
                          for (v, m), p in p_by.items()])

    # 1단계: 묶음 1~6 (α·γ 후보 포함)
    d1, f1 = run_grid(STAGE1)
    oof_p = oof_scores(STAGE1)
    disc = disc_of(oof_p)
    lap(f"1단계 그리드 OOF ({args.device}, 설계 {d1}개, 학습 {f1}개)")

    nf = {v: n_features(v) for v in STAGE1}
    alpha_pick = {m: pick(disc, list(ALPHA_VARIANTS), m, nf) for m in MODELS}
    gamma_pick = {m: pick(disc, list(GAMMA_VARIANTS), m, nf) for m in MODELS}
    a_star, g_star = alpha_pick[SELECT_MODEL][0], gamma_pick[SELECT_MODEL][0]
    ac_pick = {m: pick(disc, [a_star, "chg"], m, nf) for m in MODELS}
    # 5와 6: 0.002 안이면 k=5 (P5c-D5). 피처도 5가 적어 pick 규칙과 같은 결과다
    sl_pick = {m: pick(disc, ["sl5", "sl35"], m, nf) for m in MODELS}
    combo = (ac_pick[SELECT_MODEL][0], sl_pick[SELECT_MODEL][0])

    # 2단계: 묶음 7
    d2, f2 = run_grid(["combo"], combo)
    oof_p.update(oof_scores(["combo"]))
    disc = pl.concat([disc, disc_of({k: v for k, v in oof_p.items() if k[0] == "combo"})])
    lap(f"2단계 그리드 OOF (묶음 7 = S1 + {combo[0]} + {combo[1]}, 설계 {d2}개, 학습 {f2}개)")

    final = {"s1": "1", a_star: "2", "chg": "3", g_star: "4", "sl5": "5", "sl35": "6", "combo": "7"}
    nf["combo"] = n_features("combo", combo=combo)

    def nfeat(v, n):
        return n_features(v, n, combo if v == "combo" else ())

    disc = disc.with_columns(n_features=pl.struct("bundle", "N").map_elements(lambda s: nfeat(s["bundle"], s["N"]), return_dtype=pl.Int64),
                             final_label=pl.col("bundle").replace_strict(final, default=None, return_dtype=pl.String))
    disc = disc.select("bundle", "final_label", "model", "N", "n_features",
                       *[c for c in disc.columns if c not in ("bundle", "final_label", "model", "N", "n_features", "split")])
    for v in final:
        for m in MODELS:
            m1w.write_parquet(oof_p[(v, m)], data_dir / f"{m}__{v}.parquet")
    # S1 = 5b ② OOF인지 (5b OOF 파일이 이 머신에 있을 때만)
    repro = {}
    for m in MODELS:
        ref_path = m1w.DATA_DIR / f"{m}__{S1_NAME_5B}.parquet"
        if args.smoke or not ref_path.exists():
            repro[m] = None
            continue
        ref, mine = pl.read_parquet(ref_path).sort("N", "traj_id"), oof_p[(BASE, m)]
        same_rows = ref.select("traj_id", "N").equals(mine.select("traj_id", pl.col("N").cast(ref["N"].dtype)))
        repro[m] = {"same_rows": bool(same_rows),
                    "max_abs_diff": float(np.abs(ref["p_fail"].to_numpy() - mine["p_fail"].to_numpy()).max()) if same_rows else None}
    lap("OOF 점수·판별 지표")

    # ---- train 운영점 (최종 묶음 7개) ----
    otasks = [(v, m, n, train, oof_p[(v, m)].filter(pl.col("N") == n), costs_train, k_star)
              for v in final for m in MODELS for n in evaluate.N_GRID]
    op_rows = [r for out in evaluate.parallel(m1w.ops_task, otasks, args.jobs) for r in out]
    chosen = {(r["bundle"], r["model"], r["N"], r["policy"]): (r["tau"], r["cap"]) for r in op_rows}
    lap(f"train 운영점 ({len(otasks)}개)")

    # ---- 부트스트랩 (train 이슈 단위, 같은 재표본. 묶음 − S1 짝지은 차이) ----
    boots = {}
    if not args.skip_bootstrap:
        btasks = []
        for v in final:
            for m in MODELS:
                for n in evaluate.N_GRID:
                    pa, po = oof_p[(v, m)].filter(pl.col("N") == n), oof_p[(BASE, m)].filter(pl.col("N") == n)
                    pts_a = {pol: chosen[(v, m, n, pol)] for pol in evaluate.POLICIES}
                    pts_o = {pol: chosen[(BASE, m, n, pol)] for pol in evaluate.POLICIES}
                    c = evaluate.compressed_costs(costs_train, n, [cap for _, cap in (*pts_a.values(), *pts_o.values())], k_star)
                    btasks.append(((v, m, n), train, evaluate.pair_preds(pa, po), m1w.oof_stat(n, pts_a, pts_o, k_star, c), n_boot))
        boots = {r["key"]: r["boot"] for r in evaluate.parallel(m1w.boot_task, btasks, args.jobs)}
        lap(f"부트스트랩 ({len(btasks)}개 × {n_boot}회)")

    def ci(key, stat):
        if key not in boots:
            return (None, None)
        r = boots[key].filter(pl.col("stat") == stat)
        return (r["lo"][0], r["hi"][0])

    drows = []
    for r in disc.iter_rows(named=True):
        k = (r["bundle"], r["model"], r["N"])
        e = {}
        for c in ("auroc", "within_pair"):
            e[f"{c}_lo"], e[f"{c}_hi"] = ci(k, c)
        ref = disc.filter((pl.col("bundle") == BASE) & (pl.col("model") == r["model"]) & (pl.col("N") == r["N"])).row(0, named=True)
        for c in ("auroc", "within_pair", "within_issue_mean"):
            e[f"d_{c}"] = r[c] - ref[c]
        for c in ("auroc", "within_pair"):
            e[f"d_{c}_lo"], e[f"d_{c}_hi"] = ci(k, f"do_{c}")
        drows.append({**r, **e})
    disc_df = pl.DataFrame(drows, infer_schema_length=None)

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
        orows.append({**r, "final_label": final[r["bundle"]], **e})
    cap_rows = []
    ref_k = evaluate.cap_point(train, k_star, costs_train)
    for policy, kk in (("cap", k_star), ("cap_proposal", evaluate.K_PROPOSAL)):
        pt = evaluate.cap_point(train, kk, costs_train)
        cap_rows.append({"bundle": None, "final_label": None, "model": "b0_cap", "policy": policy, "N": None, "tau": None, "cap": kk, **pt,
                         **{f"d_{c}": pt[c] - ref_k[c] for c in evaluate.DELTA_COLUMNS},
                         **evaluate.fold_spread(train, None, None, None, kk, costs_train)})
    op_cols = ["bundle", "final_label", "model", "policy", "N", "tau", "cap", "stop_rate", "lost_success", "lost_success_lo", "lost_success_hi",
               "saved_steps", "saved_steps_lo", "saved_steps_hi", "saved_tokens", "saved_tokens_lo", "saved_tokens_hi",
               "d_lost_success", "d_lost_success_lo", "d_lost_success_hi", "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi",
               "d_saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi",
               "do_lost_success", "do_lost_success_lo", "do_lost_success_hi", "do_saved_steps", "do_saved_steps_lo", "do_saved_steps_hi",
               "do_saved_tokens", "do_saved_tokens_lo", "do_saved_tokens_hi",
               "fold_lost_success_min", "fold_lost_success_max", "fold_saved_steps_min", "fold_saved_steps_max"]
    ops_df = evaluate.frame(cap_rows + orows, op_cols, {**evaluate.OP_SCHEMA, "bundle": pl.String, "final_label": pl.String})
    ops_df = ops_df.with_columns(meets_select_point=pl.col("lost_success") <= metrics.SELECT_MAX_LOSS + metrics.LOSS_EPS)

    # ---- S2 선택 (P5b-D3) ----
    order = list(final)
    s2_pick = {m: pick(disc_df, order, m, nf) for m in MODELS}
    s2 = s2_pick[SELECT_MODEL][0]
    sel_rows = []

    def add_rows(stage, picks, cands):
        for m in MODELS:
            v_pick, score = picks[m]
            best = max(score.values())
            for v in cands:
                sel_rows.append({"stage": stage, "model": m, "bundle": v, "final_label": final.get(v), "n_features_N20": nf[v],
                                 **{f"within_pair_N{n}": disc_df.filter((pl.col("bundle") == v) & (pl.col("model") == m) & (pl.col("N") == n))["within_pair"][0]
                                    for n in evaluate.N_GRID},
                                 "select_mean": score[v], "d_select_mean_vs_s1": score[v] - sel_metric(disc_df, BASE, m),
                                 "in_tie_band": score[v] >= best - SELECT_TIE, "picked_by_this_model": v == v_pick,
                                 "used": m == SELECT_MODEL and v == v_pick})
    add_rows("alpha", alpha_pick, list(ALPHA_VARIANTS))
    add_rows("gamma", gamma_pick, list(GAMMA_VARIANTS))
    add_rows("combo_a_or_c", ac_pick, [a_star, "chg"])
    add_rows("combo_slope", sl_pick, ["sl5", "sl35"])
    add_rows("S2", s2_pick, order)
    sel_df = pl.DataFrame(sel_rows)
    # 가설 (P5c-D7): S2는 N ≥ 8에서 S1보다 OOF 이슈 내 AUROC가 높다 (방향만)
    hyp = [{"model": m, "N": n, **{c: r[c] for c in ("d_within_pair", "d_within_pair_lo", "d_within_pair_hi")}}
           for m in MODELS for n in SELECT_N
           for r in disc_df.filter((pl.col("bundle") == s2) & (pl.col("model") == m) & (pl.col("N") == n)).iter_rows(named=True)]
    lap(f"S2 선택: {s2} (묶음 {final[s2]})")

    # ---- 저장 ----
    grid_rows = [{"bundle": v, "model": m, "N": n, "n_features": len(cols_of[(v, m, n)]), "i": i, "config": m1.cfg_str(grid[i]),
                  "oof_auroc": results[(key_of[(v, m, n)], i)]["oof_auroc"], "selected": i == selected[key_of[(v, m, n)]]["i"]}
                 for (v, m, n) in key_of for i in range(len(grid))]
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(grid_rows).write_csv(out_dir / "grid.csv", float_precision=FLOAT)
    disc_df.write_csv(out_dir / "discrimination_oof.csv", float_precision=FLOAT)
    ops_df.write_csv(out_dir / "operating_points_oof.csv", float_precision=FLOAT)
    sel_df.write_csv(out_dir / "selection.csv", float_precision=FLOAT)
    feature_dist(train, feats).write_csv(out_dir / "feature_dist.csv", float_precision=FLOAT)
    if not args.smoke:
        m1w.reference_table(disc_df.filter(pl.col("final_label").is_not_null())).write_csv(out_dir / "oof_reference.csv", float_precision=FLOAT)

    n_final = len(final) * len(MODELS) * len(evaluate.N_GRID)
    crit = {
        "C1_features_finite_and_official_same_as_04_m1": args.smoke or feat_info["official_matches_04_m1_cache"] is not False,
        "C2_no_seen_issue_scores": True,  # run_grid의 assert
        # S1 재학습 = 5b ② OOF (5b 파일이 있고 같은 장치일 때만 비트 단위로 요구, 아니면 기록만)
        "C3_s1_reproduces_5b": args.smoke or all(
            r is None or (r["same_rows"] and (s1_info["phase5b_device"] != args.device or r["max_abs_diff"] == 0)) for r in repro.values()),
        "C4_k_star": args.smoke or k_star == 40,
        "C5_tables_complete": disc_df.filter(pl.col("final_label").is_not_null()).height == n_final
        and ops_df.filter(pl.col("model") != "b0_cap").height == n_final * len(evaluate.POLICIES),
        "C6_bootstrap_ci": (not args.skip_bootstrap) and ops_df.filter(pl.col("model") != "b0_cap")["d_saved_tokens_lo"].null_count() == 0,
    }
    summary = {
        "mode": "smoke" if args.smoke else "train OOF only (test not read)",
        "device": args.device, "s1": {**s1_info, "definition": "official 73 + window k=3 (7)", "reproduce_5b_oof": repro},
        "variants": {v: {"n_features_by_N": {str(n): nfeat(v, n) for n in evaluate.N_GRID}} for v in [*STAGE1, "combo"]},
        "final_bundles": final, "combo": list(combo),
        "models": list(MODELS), "grid": [dict(c) for c in grid], "xgb_fixed": m1.XGB_FIXED, "grid_tie": m1.GRID_TIE,
        "k_star": k_star, "features": feat_info,
        "unique_designs": len(designs), "grid_fits": len(results),
        "selection": {"rule": "P5b-D3, P5c-D5·D9", "model": SELECT_MODEL, "N": list(SELECT_N), "tie": SELECT_TIE,
                      "alpha": a_star, "gamma": g_star, "combo": list(combo), "S2": s2, "S2_label": final[s2],
                      "pick_by_model": {m: {"alpha": alpha_pick[m][0], "gamma": gamma_pick[m][0], "a_or_c": ac_pick[m][0],
                                            "slope": sl_pick[m][0], "S2": s2_pick[m][0]} for m in MODELS},
                      "S2_scores": s2_pick[SELECT_MODEL][1]},
        "candidates_seen": {"alpha": len(ALPHA_VARIANTS), "gamma": len(GAMMA_VARIANTS), "combo_parts": 2 * 2,
                            "final_bundles": len(final), "grid_configs_per_design": len(grid)},
        "hypothesis_P5c_D7": {"rows": hyp, "n_positive": sum(r["d_within_pair"] > 0 for r in hyp),
                              "n_ci_lo_positive": sum((r["d_within_pair_lo"] or -1) > 0 for r in hyp), "n": len(hyp)},
        "bootstrap": None if args.skip_bootstrap else {"n_boot": n_boot, "seed": io.SEED, "unit": "train issue", "base": BASE,
                                                       "note": "운영점 선택은 다시 하지 않는다 (선택 불확실성 미포함, P5b-D13)"},
        "criteria": crit,
        "criteria_note": "pytest 전체 통과와 재실행 시 results/phase5c/ 바이트 동일은 스크립트 밖에서 확인한다. 실행 시간은 표준 출력에만 남긴다",
    }
    io.write_summary(out_dir / "summary.json", summary)

    with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=250, float_precision=4):
        print(sel_df.select("stage", "model", "bundle", "final_label", "n_features_N20", *[f"within_pair_N{n}" for n in SELECT_N],
                            "select_mean", "d_select_mean_vs_s1", "in_tie_band", "picked_by_this_model"))
        print(ops_df.filter((pl.col("policy") == "combined") & (pl.col("model") == SELECT_MODEL) | (pl.col("model") == "b0_cap")).select(
            "final_label", "bundle", "N", "tau", "cap", "lost_success", "saved_steps", "d_saved_steps", "saved_tokens", "d_saved_tokens",
            "d_saved_tokens_lo", "d_saved_tokens_hi", "do_saved_steps", "do_saved_tokens"))
    print(json.dumps({"S2": s2, "combo": combo, "alpha": a_star, "gamma": g_star, "reproduce_5b": repro, "criteria": crit, "timings": timings},
                     ensure_ascii=False, indent=2, default=str))
    failed = [k for k, v in crit.items() if not v]
    if failed:
        print(f"\n완료 기준 불충족: {failed}")
        sys.exit(1)
    print("\n완료 기준 통과")


if __name__ == "__main__":
    main()
