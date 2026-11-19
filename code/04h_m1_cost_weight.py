"""Phase 5e: S3 + 비용 가중 학습 (학습 가중치만 바꿈), train OOF만. P5e-D1~D8.

S3 피처(5d 선택)는 그대로 두고 XGBoost sample_weight만 바꾼다.
- 묶음: S3 / +스텝 가중(원값) / +스텝 가중(√) / +토큰 가중(원값)
- 실패 가중치 = 결합 정책에서 N에서 끊었을 때 fit fold K* 상한 대비 더 아끼는 양 (평균 1로 맞춤)
- 성공 가중치 = λ ∈ {1, λ0/2, λ0, 2λ0}
- 선택 지표 = N∈{8,10,15,20} train OOF 결합 정책 운영점 절약 스텝 평균 (P5e-D6)

전체 실행 예: uv run --no-sync python code/04h_m1_cost_weight.py --device cpu --jobs 8
점검 실행: 위 명령에 --smoke. 정식 results에는 쓰지 않는다.
--final은 제공하지 않는다. test는 5e 이후 공통 최종 평가에서만 사용한다.
"""
import argparse
import hashlib
import importlib
import json
import math
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from afp import evaluate, features, io, metrics, preds, thought

m1 = importlib.import_module("04_m1")
m1w = importlib.import_module("04e_m1_window")
m1g = importlib.import_module("04g_m1_thought")
MODELS = m1w.MODELS
WEIGHTINGS = ("step", "step_sqrt", "token")  # P5e-D1 순서 (동률 마지막 기준)
BUNDLES = ("s3", *WEIGHTINGS)
BASE = "s3"
LAMBDA_TAGS = ("lam1", "lam0half", "lam0", "lam0x2")  # λ = 1, λ0/2, λ0, 2λ0 (P5e-D3)
LAMBDA_FACTOR = {"lam0half": 0.5, "lam0": 1.0, "lam0x2": 2.0}
LAMBDA0_KIND = {"step": "step", "step_sqrt": "step", "token": "token"}  # √ 묶음은 스텝 λ0 (P5e-D4, P5e-D9)
LAMBDA0_LOSS = (0.030, 0.040)
VARIANTS = (BASE, *[f"{w}_{t}" for w in WEIGHTINGS for t in LAMBDA_TAGS])
SELECT_MODEL = m1w.SELECT_MODEL
SELECT_N = m1w.SELECT_N
TIE = m1w.SELECT_TIE
CAP_KS = range(1, 81)
MEAN_TOL = 1e-9
OUT_DIR = io.RESULTS_DIR / "phase5e"
DATA_DIR = io.PROCESSED_DIR / "phase5e"
SMOKE_DIR = io.PROCESSED_DIR / "phase5e_smoke"
FLOAT = m1w.FLOAT


def lambda_value(tag, lam0):
    return 1.0 if tag == "lam1" else LAMBDA_FACTOR[tag] * lam0


# ---- 상류 확인 ----

def upstream_check():
    """5c S2 → 5d S3 연결 확인. 5d가 현재 5c S2로 돌았는지, handoff가 summary와 같은지 본다."""
    s2 = m1g.s2_check()
    path = m1g.OUT_DIR / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"S3 metadata missing: {path}; do not regenerate automatically")
    summary = json.loads(path.read_text())
    if summary.get("status") != "complete" or not summary.get("criteria") or not all(summary["criteria"].values()):
        raise ValueError("Phase 5d recorded incomplete criteria")
    if summary["s2"]["phase5c_S2"] != s2["phase5c_S2"]:
        raise ValueError(f"5d used S2={summary['s2']['phase5c_S2']} but 5c now selects {s2['phase5c_S2']}; rerun 5d (P9-D2)")
    sel = summary["selection"]
    if sel["S3"] not in m1g.BUNDLES:
        raise KeyError(sel["S3"])
    handoff = json.loads((m1g.DATA_DIR / "handoff.json").read_text())
    if handoff["summary"]["selection"] != sel or handoff["summary"]["s2"]["phase5c_S2"] != s2["phase5c_S2"]:
        raise ValueError("5d handoff does not match results/phase5d/summary.json")
    return {**s2, "phase5d_S3": sel["S3"], "phase5d_t": sel["t"], "phase5d_vocab_k": sel["vocab_k"],
            "phase5d_device": summary["device"]}, handoff


def fingerprint(frame, sort):
    return hashlib.sha256(frame.sort(sort).write_json().encode()).hexdigest()


# ---- 가중치 (fit fold 전용, P5e-D2·D7) ----

def fold_k_star(train, held_fold):
    """fit fold(= held fold 밖 train 궤적 전체)에서 다시 고른 K*. held fold 라벨은 쓰지 않는다."""
    rows = train.filter(pl.col("fold") != held_fold)
    return int(metrics.select_cap(metrics.fixed_cap_curve(rows, None, CAP_KS), metrics.SELECT_MAX_LOSS)["K"])


def raw_saving(rows, n, k, kind, costs=None):
    """N에서 끊었을 때 고정 상한 K 대비 더 아끼는 양 (결합 정책, metrics.cost_curve와 같은 식). rows 순서."""
    if k <= n:
        raise ValueError(f"K={k} must exceed N={n}")
    if kind == "step":
        return (np.minimum(rows["n_steps"].to_numpy(), k) - n).astype(np.float64)
    if kind == "token":
        tok = (costs.join(rows.select("traj_id"), on="traj_id", how="semi")
               .filter((pl.col("step_idx") > n) & (pl.col("step_idx") <= k))
               .group_by("traj_id").agg(pl.col("ctx").sum().alias("_tok")))
        out = (rows.select("traj_id").with_row_index("_i").join(tok, on="traj_id", how="left")
               .sort("_i").fill_null(0))
        return out["_tok"].to_numpy().astype(np.float64)
    raise KeyError(kind)


def fit_weights(fit_rows, scope, held_fold, n, weighting, lam, k_star, costs=None):
    """fit fold 행의 학습 가중치. 실패 = 결합 정책 절약량(원값 또는 √, 평균 1), 성공 = λ.

    가중치는 n_steps(N 이후 정보)로 만들므로 held fold·test 행이 들어오면 멈춘다 (P5e-D7 ①).
    """
    scope.validate(fit_rows, held_fold)
    if fit_rows.filter(pl.col("n_steps") <= n).height:
        raise ValueError("finished trajectories cannot be fit rows at this N")
    if not (math.isfinite(lam) and lam > 0):
        raise ValueError("lambda must be positive and finite")
    fail = ~fit_rows["target"].to_numpy().astype(bool)
    if not fail.any():
        raise ValueError("no failures in fit rows")
    kind = "token" if weighting == "token" else "step"
    raw = raw_saving(fit_rows.filter(pl.Series(fail)), n, k_star, kind, costs)
    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("nonpositive failure saving")
    if weighting == "step_sqrt":
        raw = np.sqrt(raw)
    elif weighting not in ("step", "token"):
        raise KeyError(weighting)
    w = np.full(fit_rows.height, float(lam))
    w[fail] = raw / raw.mean()
    return w


# ---- 학습 ----

def fit_weighted(cfg, X, y, w, device):
    """w=None이면 5d와 같은 학습기 그대로 (S3 재현). 아니면 같은 설정에 sample_weight만 더한다 (balance 축 유지, P5e-D5)."""
    if w is None:
        return m1w.fit_xgb(cfg, X, y, device)
    from xgboost import XGBClassifier
    if len(w) != len(y) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("invalid sample weights")
    n_fail, n_succ = int(y.sum()), int((~y.astype(bool)).sum())
    spw = n_succ / n_fail if cfg["balance"] else 1.0
    m = XGBClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], scale_pos_weight=spw,
                      **{**m1.XGB_FIXED, "device": device}).fit(X, y, sample_weight=w)
    if device != "cpu":
        m.set_params(device="cpu")
    return ("xgb", m)


def predict(model, X):
    """예측에는 가중치가 없다 (P5e-D7 ③)."""
    return m1.predict_model(model, X)


def oof_task(key, cfg, y, blocks, weights, device):
    p, seen = np.full(len(y), np.nan), np.zeros(len(y), dtype=np.int8)
    for k, (fit, held, X_fit, X_held) in enumerate(blocks):
        if np.intersect1d(fit, held).size:
            raise ValueError("OOF row overlap")
        w = None if weights is None else weights[k]
        if w is not None and len(w) != len(fit):
            raise ValueError("weights do not match fit rows")
        model = fit_weighted(cfg, X_fit, y[fit], w, device)
        p[held] = predict(model, X_held)
        seen[held] += 1
    if not np.all(seen == 1) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid OOF coverage or probabilities")
    return {"key": key, "oof": p, "oof_auroc": float(roc_auc_score(y, p))}


def fit_grid(D, extras, weights, grid, device, jobs, key):
    blocks = m1g.fold_blocks(D, extras)
    rows = evaluate.parallel(oof_task, [((key, i), cfg, D["y"], blocks, weights, device) for i, cfg in enumerate(grid)], jobs)
    rows = [{"i": i, **r} for i, r in enumerate(rows)]
    return rows[m1.choose(rows)], rows


# ---- λ0 (P5e-D4) ----

def lambda0(train, costs, p_s3, k_star):
    """kind(스텝·토큰)별 λ0 = S3 OOF 결합 정책 곡선의 손실 3~4% 구간 교환비(절약량/잃은 성공 수)의 N 평균
    ÷ 실패 원가중(결합 정책 절약량) 평균. 곡선은 그 N에서 train OOF로 고른 결합 정책 운영점
    (손실 ≤3.5%, 절약 스텝 최대)의 상한 K다 (P5e-D4, P5e-D10). 반환: {kind: (λ0, 행 목록)}.
    """
    units = {"step": "saved_steps", "token": "saved_tokens"}
    totals = {"step": float(train["n_steps"].sum()),
              "token": float(costs.join(train.select("traj_id"), on="traj_id", how="semi")["ctx"].sum())}
    n_succ = int(train["target"].sum())
    rows = {kind: [] for kind in units}
    for n in SELECT_N:
        curves = evaluate.candidate_curves(train, p_s3, n, "combined", costs)
        op = metrics.select_from_curves(curves, n)
        band = curves[op["cap"]].filter(pl.col("lost_success").is_between(LAMBDA0_LOSS[0] - metrics.LOSS_EPS,
                                                                          LAMBDA0_LOSS[1] + metrics.LOSS_EPS))
        if band.height < 2:
            raise ValueError(f"N={n}: fewer than two curve rows with loss in {LAMBDA0_LOSS}")
        first, last = band.row(0, named=True), band.row(-1, named=True)
        d_lost = (last["lost_success"] - first["lost_success"]) * n_succ
        if d_lost <= 0:
            raise ValueError(f"N={n}: no lost-success change inside {LAMBDA0_LOSS}")
        fails = metrics.running(train, n).filter(~pl.col("target")).sort("traj_id")
        for kind, unit in units.items():
            d_saved = (last[unit] - first[unit]) * totals[kind]
            raw = raw_saving(fails, n, k_star, kind, costs)
            rows[kind].append({"kind": kind, "N": n, "cap": op["cap"], "band_rows": band.height,
                               "lost_first": first["lost_success"], "lost_last": last["lost_success"],
                               "d_lost_successes": d_lost, "d_saved": d_saved, "exchange_ratio": d_saved / d_lost,
                               "mean_raw_failure_weight": float(raw.mean())})
    out = {}
    for kind, rs in rows.items():
        lam0 = float(np.mean([r["exchange_ratio"] for r in rs]) / np.mean([r["mean_raw_failure_weight"] for r in rs]))
        if not (math.isfinite(lam0) and lam0 > 0):
            raise ValueError(f"invalid lambda0 {lam0} ({kind})")
        out[kind] = (lam0, rs)
    return out


# ---- 선택 (P5e-D6) ----

def select_score(ops, variant, model):
    rows = ops.filter((pl.col("bundle") == variant) & (pl.col("model") == model) & (pl.col("policy") == "combined")
                      & pl.col("N").is_in(list(SELECT_N)))
    if rows.height != len(SELECT_N) or not np.isfinite(rows["saved_steps"].to_numpy()).all():
        raise ValueError(f"selection metric incomplete: {variant}, {model}")
    return float(rows["saved_steps"].mean())


def pick(scores, cands, lams):
    """최대에서 TIE 안이면 가중 없는 쪽, 다음은 λ가 작은 쪽, 다음은 후보 순서."""
    best = max(scores[v] for v in cands)
    band = [v for v in cands if scores[v] >= best - TIE]
    return min(band, key=lambda v: (lams[v] is not None, lams[v] or 0.0, cands.index(v)))


# ---- 실행 ----

def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="XGBoost 학습 기본 cuda; 사용 불가 시 중단")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--gpu-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    ap.add_argument("--skip-bootstrap", action="store_true", help="개발 전용, 완료 기준 불충족")
    ap.add_argument("--smoke", action="store_true", help="train 이슈 1/10·그리드 2개·부트스트랩 20회; 임시 경로만")
    return ap


def load_lexical(full_train, train, n, t, vocab_k):
    """5d가 fold k 어휘(fit fold만으로 고름)로 만든 행렬. 행 = 전체 train의 N 진행 중 궤적(traj_id 순)."""
    full_ids = metrics.running(full_train, n).sort("traj_id")["traj_id"]
    ids = metrics.running(train, n).sort("traj_id")["traj_id"]
    mask = full_ids.is_in(ids.implode()).to_numpy()
    width = len(thought.lexical_columns(n, vocab_k, m1g.KS))
    out = {}
    for f in range(5):
        X = np.load(m1g.DATA_DIR / f"lex_N{n}_f{f}_t{t}_k{vocab_k}.npy", allow_pickle=False)
        if X.shape != (full_ids.len(), width):
            raise ValueError(f"5d lexical matrix shape mismatch N={n} fold={f}: {X.shape}")
        out[f] = X[mask]
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.jobs, args.gpu_jobs, args.n_boot) < 1:
        raise ValueError("jobs and n_boot must be positive")
    upstream, handoff = upstream_check()
    started = time.monotonic()
    s3, t_star, vocab_k = upstream["phase5d_S3"], upstream["phase5d_t"], upstream["phase5d_vocab_k"]
    train, scope, steps, _, b1 = m1g.load_train_inputs()
    full_train = train
    inputs_same = {"train": fingerprint(train, "traj_id") == handoff["fingerprints"]["train"],
                   "prefix_steps": fingerprint(steps, ["traj_id", "step_idx"]) == handoff["fingerprints"]["prefix_steps"],
                   "b1_train": fingerprint(b1, ["N", "traj_id"]) == handoff["fingerprints"]["b1_train"]}
    if not all(inputs_same.values()):
        raise ValueError(f"inputs differ from 5d handoff: {inputs_same}")
    if args.smoke:
        issues = train["instance_id"].unique().sort().gather(np.arange(0, train["instance_id"].n_unique(), 10))
        train = train.filter(pl.col("instance_id").is_in(issues.implode()))
        steps = steps.join(train.select("traj_id"), on="traj_id", how="semi")
        scope = thought.TrainScope(train)
    costs = metrics.eval_costs(train)
    k_star = metrics.select_cap(metrics.fixed_cap_curve(train, costs, CAP_KS), metrics.SELECT_MAX_LOSS)["K"]
    fold_k = {k: fold_k_star(train, k) for k in range(5)}
    feats, feat_info = m1g.base_features(train)
    base_bundle = "length" if "length" in s3 else "s2"
    lex = {}
    for n in evaluate.N_GRID:
        if "length" in s3:
            lengths = features.thought_length_features(steps, train, n, m1g.KS)
            features.check_features(lengths.with_columns(N=pl.lit(n)), train, [n], features.thought_length_columns(n, m1g.KS))
            feats[n] = feats[n].join(lengths, on="traj_id", how="left")
        if "lexical" in s3:
            lex[n] = load_lexical(full_train, train, n, t_star, vocab_k)
        if m1g.columns(s3, n, False, vocab_k) != handoff["columns"][s3][str(n)]:
            raise ValueError(f"N={n}: S3 columns differ from 5d handoff")
    designs = {}
    for n in evaluate.N_GRID:
        ids = metrics.running(train, n).sort("traj_id")["traj_id"].to_list()
        for model, use_b1 in MODELS.items():
            D = m1w.design(train, feats, b1 if use_b1 else None, n, m1g.columns(base_bundle, n, use_b1))
            if D["traj_id"].to_list() != ids:
                raise ValueError("lexical/base row ordering mismatch")
            designs[(model, n)] = D
    grid = m1.XGB_GRID[:m1w.SMOKE_GRID] if args.smoke else m1.XGB_GRID
    jobs = args.gpu_jobs if args.device == "cuda" else args.jobs
    out_dir, data_dir = (SMOKE_DIR, SMOKE_DIR / "oof") if args.smoke else (OUT_DIR, DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    io.write_summary(out_dir / "summary.json", {"mode": "smoke" if args.smoke else "train OOF only",
                                               "status": "incomplete", "criteria": {"execution_finished": False}})
    m1w.check_device(args.device)
    grid_rows, parts = [], {}

    def run(variant, model, n, weights, lam):
        D = designs[(model, n)]
        best, rs = fit_grid(D, lex.get(n), weights, grid, args.device, jobs, (variant, model, n))
        grid_rows.extend({"bundle": variant, "model": model, "N": n, "lambda": lam, "i": r["i"],
                          "config": m1.cfg_str(grid[r["i"]]), "oof_auroc": r["oof_auroc"], "selected": r["i"] == best["i"]}
                         for r in rs)
        parts.setdefault((variant, model), []).append(m1g.prediction_frame(D, best["oof"], n))

    def collect():
        return {key: pl.concat(p).sort("N", "traj_id") for key, p in parts.items()}

    # 1) 가중 없는 S3 (5d와 같은 설계·학습기 → 같은 장치면 비트 단위로 같아야 한다)
    for n in evaluate.N_GRID:
        for model in MODELS:
            run(BASE, model, n, None, None)
    print(f"S3 refit complete ({time.monotonic() - started:.1f}s)", flush=True)
    p = collect()

    # 2) λ0: 모델별 S3 OOF에서 (스텝·토큰)
    lam0, lam0_rows = {}, []
    for model in MODELS:
        for kind, (value, rows) in lambda0(train, costs, p[(BASE, model)], k_star).items():
            lam0[(model, kind)] = value
            lam0_rows.extend({"model": model, **r, "lambda0": value} for r in rows)
    print(f"lambda0 = {lam0}", flush=True)

    # 3) 가중 묶음 × λ 4개. 가중치는 fold k 모델의 fit 행에만 쓴다.
    weight_rows = []
    lams = {(BASE, m): None for m in MODELS}
    for weighting in WEIGHTINGS:
        for tag in LAMBDA_TAGS:
            variant = f"{weighting}_{tag}"
            for model in MODELS:
                lam = lambda_value(tag, lam0[(model, LAMBDA0_KIND[weighting])])
                lams[(variant, model)] = lam
                for n in evaluate.N_GRID:
                    D = designs[(model, n)]
                    alive = metrics.running(train, n).sort("traj_id")
                    weights = []
                    for k in range(5):
                        fit_rows = alive.filter(pl.col("fold") != k)
                        if fit_rows["traj_id"].to_list() != D["traj_id"].gather(np.flatnonzero(D["fold"] != k)).to_list():
                            raise ValueError("weight rows differ from fit rows")
                        w = fit_weights(fit_rows, scope, k, n, weighting, lam, fold_k[k], costs)
                        fail = D["y"][D["fold"] != k].astype(bool)
                        weight_rows.append({"bundle": variant, "model": model, "N": n, "fold": k, "k_star_fold": fold_k[k],
                                            "lambda": lam, "n_fail": int(fail.sum()), "n_success": int((~fail).sum()),
                                            "fail_w_mean": float(w[fail].mean()), "fail_w_min": float(w[fail].min()),
                                            "fail_w_max": float(w[fail].max()), "success_w": float(lam)})
                        weights.append(w)
                    run(variant, model, n, weights, lam)
            print(f"{variant}: complete ({time.monotonic() - started:.1f}s)", flush=True)
    p = collect()
    for (v, m), prediction in p.items():
        preds.validate(prediction, {"train": train}, evaluate.N_GRID)
        m1w.write_parquet(prediction, data_dir / f"{m}__{v}.parquet")

    # 4) 모든 후보의 판별·운영점 (점추정), λ·묶음 선택
    search_disc = pl.concat([evaluate.discrimination_table(m, pr, {"train": train}, evaluate.N_GRID).with_columns(bundle=pl.lit(v))
                             for (v, m), pr in p.items()])
    tasks = [(v, m, n, train, p[(v, m)].filter(pl.col("N") == n), costs, k_star)
             for v in VARIANTS for m in MODELS for n in evaluate.N_GRID]
    search_ops = pl.DataFrame([r for rs in evaluate.parallel(m1w.ops_task, tasks, jobs) for r in rs],
                              infer_schema_length=None, schema_overrides={"N": pl.Int64, "cap": pl.Int64, "tau": pl.Float64})
    scores = {(v, m): select_score(search_ops, v, m) for v in VARIANTS for m in MODELS}
    lam_pick, bundle_pick = {}, {}
    for m in MODELS:
        for w in WEIGHTINGS:
            cands = [f"{w}_{t}" for t in LAMBDA_TAGS]
            lam_pick[(w, m)] = pick({v: scores[(v, m)] for v in cands}, cands, {v: lams[(v, m)] for v in cands})
    rep = {BASE: BASE, **{w: lam_pick[(w, SELECT_MODEL)] for w in WEIGHTINGS}}  # 5d 어휘처럼 주 모델의 λ를 두 모델에 쓴다
    for m in MODELS:
        own = {BASE: BASE, **{w: lam_pick[(w, m)] for w in WEIGHTINGS}}
        b_scores = {b: scores[(own[b], m)] for b in BUNDLES}
        bundle_pick[m] = pick(b_scores, list(BUNDLES), {b: lams[(own[b], m)] for b in BUNDLES})
    s4 = bundle_pick[SELECT_MODEL]
    print(f"S4={s4} ({rep[s4]}); lambda picks={lam_pick}", flush=True)

    # 5) 최종 묶음 4개 (주 모델 λ) 운영점·부트스트랩, S3 대비
    final_p = {(b, m): p[(rep[b], m)] for b in BUNDLES for m in MODELS}
    final_disc = pl.concat([evaluate.discrimination_table(m, pr, {"train": train}, evaluate.N_GRID).with_columns(bundle=pl.lit(b))
                            for (b, m), pr in final_p.items()])
    disc, ops = m1g.evaluate_final(train, costs, k_star, final_p, final_disc, args.jobs,
                                   20 if args.smoke else args.n_boot, args.skip_bootstrap, bundles=BUNDLES, base=BASE)
    disc = disc.with_columns(variant=pl.col("bundle").replace_strict(rep, default=None))
    ops = ops.with_columns(variant=pl.col("bundle").replace_strict(rep, default=None))

    # 선택 표
    sel_rows = []
    for m in MODELS:
        for w in WEIGHTINGS:
            cands = [f"{w}_{t}" for t in LAMBDA_TAGS]
            best = max(scores[(v, m)] for v in cands)
            for v in cands:
                sel_rows.append({"stage": "lambda", "model": m, "bundle": w, "variant": v, "lambda": lams[(v, m)],
                                 **{f"saved_steps_N{n}": search_ops.filter((pl.col("bundle") == v) & (pl.col("model") == m)
                                    & (pl.col("policy") == "combined") & (pl.col("N") == n))["saved_steps"][0] for n in SELECT_N},
                                 "select_mean": scores[(v, m)], "d_select_mean_vs_s3": scores[(v, m)] - scores[(BASE, m)],
                                 "within_pair_mean": m1g.mean_score(search_disc, v, m),
                                 "in_tie_band": scores[(v, m)] >= best - TIE, "picked_by_this_model": v == lam_pick[(w, m)],
                                 "at_grid_boundary": v == lam_pick[(w, m)] and v.endswith(("lam0half", "lam0x2")),
                                 "used": m == SELECT_MODEL and v == rep[w]})
        b_scores = {b: scores[(rep[b], m)] for b in BUNDLES}
        best = max(b_scores.values())
        for b in BUNDLES:
            v = rep[b]
            sel_rows.append({"stage": "S4", "model": m, "bundle": b, "variant": v, "lambda": lams[(v, m)],
                             **{f"saved_steps_N{n}": search_ops.filter((pl.col("bundle") == v) & (pl.col("model") == m)
                                & (pl.col("policy") == "combined") & (pl.col("N") == n))["saved_steps"][0] for n in SELECT_N},
                             "select_mean": b_scores[b], "d_select_mean_vs_s3": b_scores[b] - b_scores[BASE],
                             "within_pair_mean": m1g.mean_score(search_disc, v, m),
                             "in_tie_band": b_scores[b] >= best - TIE, "picked_by_this_model": b == bundle_pick[m],
                             "at_grid_boundary": None, "used": m == SELECT_MODEL and b == s4})

    # 5c·5d처럼 같은 장치에서 S3 OOF 재현을 확인한다.
    repro = {}
    for model in MODELS:
        path = m1g.DATA_DIR / f"{model}__{s3}.parquet"
        if args.smoke or not path.exists():
            repro[model] = None
            continue
        old, new = pl.read_parquet(path).sort("N", "traj_id"), p[(BASE, model)]
        same_rows = old.select("traj_id", "N").cast({"N": pl.Int64}).equals(new.select("traj_id", "N").cast({"N": pl.Int64}))
        same_scores = bool(same_rows and np.array_equal(old["p_fail"].to_numpy(), new["p_fail"].to_numpy()))
        repro[model] = {"same_rows": same_rows, "same_scores": same_scores,
                        "matches_when_same_device": same_rows and (upstream["phase5d_device"] != args.device or same_scores)}
    hyp = ops.filter((pl.col("bundle") == s4) & (pl.col("policy") == "combined") & pl.col("N").is_in(list(SELECT_N))).select(
        "model", "N", "do_saved_steps", "do_saved_steps_lo", "do_saved_steps_hi").to_dicts()
    expected = len(BUNDLES) * len(MODELS) * len(evaluate.N_GRID)
    model_ops = ops.filter(pl.col("model") != "b0_cap")
    crit = {"C1_upstream_and_inputs": all(inputs_same.values()),
            "C2_s3_reproduced_when_available": all(r is None or r["matches_when_same_device"] for r in repro.values()),
            "C3_weights_valid": all(abs(r["fail_w_mean"] - 1) <= MEAN_TOL and r["fail_w_min"] > 0 for r in weight_rows),
            "C4_tables_complete": disc.height == expected and model_ops.height == expected * len(evaluate.POLICIES)
            and search_disc.height == len(VARIANTS) * len(MODELS) * len(evaluate.N_GRID),
            "C5_lambda0_valid": all(math.isfinite(v) and v > 0 for v in lam0.values()),
            "C6_bootstrap_ci": not args.skip_bootstrap and (args.smoke or args.n_boot == evaluate.N_BOOT)
            and m1g.intervals_complete(ops, [f"{prefix}{c}" for prefix in ("", "d_") for c in evaluate.DELTA_COLUMNS])
            and m1g.intervals_complete(model_ops, [f"do_{c}" for c in evaluate.DELTA_COLUMNS])
            and m1g.intervals_complete(disc, [f"{prefix}{c}" for prefix in ("", "d_") for c in ("auroc", "within_pair")])}

    tables = {"grid": grid_rows, "selection": sel_rows, "lambda0": lam0_rows, "weights": weight_rows}
    for name, rows in tables.items():
        pl.DataFrame(rows, infer_schema_length=None).write_csv(out_dir / f"{name}.csv", float_precision=FLOAT)
    pl.DataFrame([{"fold": k, "k_star_fold": v} for k, v in fold_k.items()]).write_csv(out_dir / "fold_k_star.csv")
    disc.write_csv(out_dir / "discrimination_oof.csv", float_precision=FLOAT)
    search_disc.write_csv(out_dir / "lambda_search_discrimination_oof.csv", float_precision=FLOAT)
    search_ops.write_csv(out_dir / "lambda_search_operating_points_oof.csv", float_precision=FLOAT)
    ops.write_csv(out_dir / "operating_points_oof.csv", float_precision=FLOAT)
    ops.filter(pl.col("policy") == "combined").select(
        "bundle", "variant", "model", "N", "lost_success", "saved_steps", "saved_tokens", "d_saved_steps", "d_saved_steps_lo",
        "d_saved_steps_hi", "d_saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi", "do_saved_steps", "do_saved_steps_lo",
        "do_saved_steps_hi", "do_saved_tokens", "do_saved_tokens_lo", "do_saved_tokens_hi").write_csv(
        out_dir / "n_summary.csv", float_precision=FLOAT)
    summary = {"mode": "smoke" if args.smoke else "train OOF only", "status": "complete" if all(crit.values()) else "incomplete",
               "device": args.device, "seed": io.SEED, "upstream": {**upstream, "reproduce_5d_s3_oof": repro},
               "models": list(MODELS), "N": list(evaluate.N_GRID), "grid": list(grid), "xgb_fixed": m1.XGB_FIXED,
               "grid_tie": m1.GRID_TIE, "k_star": k_star, "fold_k_star": {str(k): v for k, v in fold_k.items()},
               "features": feat_info, "lambda0": {f"{m}.{kind}": v for (m, kind), v in lam0.items()},
               "lambda0_loss_band": list(LAMBDA0_LOSS),
               "selection": {"model": SELECT_MODEL, "N": list(SELECT_N), "tie": TIE, "metric": "combined saved_steps mean",
                             "S4": s4, "S4_variant": rep[s4], "lambda_by_bundle": {w: rep[w] for w in WEIGHTINGS},
                             "pick_by_model": {m: {"S4": bundle_pick[m], "lambda": {w: lam_pick[(w, m)] for w in WEIGHTINGS}}
                                               for m in MODELS}},
               "candidates_seen": {"variants": len(VARIANTS), "lambda_per_bundle": len(LAMBDA_TAGS),
                                   "final_bundles": len(BUNDLES), "xgb_configs": len(grid)},
               "hypothesis": {"rows": hyp, "n_positive": sum(r["do_saved_steps"] > 0 for r in hyp),
                              "n_ci_lo_positive": sum(r["do_saved_steps_lo"] is not None and r["do_saved_steps_lo"] > 0 for r in hyp)},
               "bootstrap": {"n_boot": 20 if args.smoke else args.n_boot, "skipped": args.skip_bootstrap,
                             "unit": "train issue", "seed": io.SEED, "base": BASE, "bundles": list(BUNDLES),
                             "lambda_candidates_point_only": True, "selection_uncertainty_included": False},
               "criteria": crit, "external_validation": "pytest and byte reproducibility are separate; not implied by script criteria"}
    io.write_summary(out_dir / "summary.json", summary)
    metadata = {"summary": summary, "train_rows": train.select("traj_id", "instance_id", "fold").to_dicts(),
                "fingerprints": handoff["fingerprints"],
                "columns": {str(n): m1g.columns(s3, n, False, vocab_k) for n in evaluate.N_GRID},
                "selected_xgb": [r for r in grid_rows if r["selected"]]}
    io.write_summary(data_dir / "handoff.json", metadata)
    print(pl.DataFrame(sel_rows, infer_schema_length=None))
    print(f"S4={s4} ({rep[s4]}); criteria={crit}; elapsed={time.monotonic() - started:.1f}s", flush=True)
    if not all(crit.values()):
        raise SystemExit("Phase 5e script criteria incomplete")


if __name__ == "__main__":
    main()
