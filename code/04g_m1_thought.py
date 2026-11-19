"""Phase 5d: S2 + 생각 길이/어휘, train OOF만. 전체 실행은 사용자만 한다.

사용자용 전체 실행 예: uv run --no-sync python code/04g_m1_thought.py --device cuda --gpu-jobs 4 --jobs 24
점검 실행(별도 허용 필요): 위 명령에 --smoke. 정식 results에는 쓰지 않는다.
--final은 제공하지 않는다. test는 5e 이후 공통 최종 평가에서만 사용한다.
"""
import argparse
import hashlib
import importlib
import json
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from afp import evaluate, features, io, metrics, oof, preds, thought

m1 = importlib.import_module("04_m1")
m1w = importlib.import_module("04e_m1_window")
m1t = importlib.import_module("04f_m1_trend")
MODELS = m1w.MODELS
BUNDLES = ("s2", "length", "lexical", "length_lexical")
BASE = "s2"
SELECT_MODEL = m1w.SELECT_MODEL
SELECT_N = m1w.SELECT_N
TIE = m1w.SELECT_TIE
KS = (3,)
SEARCH = tuple((t, k) for t in thought.SEARCH_T for k in thought.SEARCH_K)
OUT_DIR = io.RESULTS_DIR / "phase5d"
DATA_DIR = io.PROCESSED_DIR / "phase5d"
SMOKE_DIR = io.PROCESSED_DIR / "phase5d_smoke"
FLOAT = m1w.FLOAT
# (변형, combo). s2_check가 results/phase5c/summary.json의 selection.S2에서 채운다 (P9-D2 (a)).
S2 = None


def s2_cols(n, use_b1):
    if S2 is None:
        raise RuntimeError("S2 not loaded; call s2_check() first")
    return m1t.bundle_cols(S2[0], n, use_b1, S2[1])


def columns(bundle, n, use_b1, vocab_k=0):
    return [*s2_cols(n, use_b1),
            *(features.thought_length_columns(n, KS) if "length" in bundle else []),
            *(thought.lexical_columns(n, vocab_k, KS) if "lexical" in bundle else [])]


def s2_check():
    """5c가 고른 S2를 읽는다. 고정 비교 없이 재실행의 선택을 따른다 (P9-D2 (a))."""
    global S2
    s1 = m1t.s1_check()
    path = m1t.OUT_DIR / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"S2 metadata missing: {path}; do not regenerate automatically")
    summary = json.loads(path.read_text())
    if not all(summary["criteria"].values()):
        raise ValueError("Phase 5c recorded incomplete criteria")
    variant = summary["selection"]["S2"]
    combo = tuple(summary["selection"]["combo"]) if variant == "combo" else ()
    m1t.extra_cols(variant, max(evaluate.N_GRID), combo)  # 모르는 변형이면 KeyError로 멈춘다
    S2 = (variant, combo)
    return {**s1, "phase5c_S2": variant, **({"phase5c_combo": list(combo)} if combo else {}),
            "phase5c_device": summary["device"]}


def load_train_inputs():
    """test 집합·점수는 요청하지 않는다. 비용은 train만 계산; 전체 비용 캐시도 안 쓴다."""
    train = metrics.eval_set("train")
    scope = thought.TrainScope(train)
    steps = (io.scan_steps(["traj_id", "step_idx", "action_text"])
             .filter(pl.col("step_idx") <= max(evaluate.N_GRID))
             .join(train.select("traj_id").lazy(), on="traj_id", how="semi").collect())
    texts = io.load_trajectories(["traj_id", "instance_id", "issue_text"]).join(train.select("traj_id"), on="traj_id", how="semi")
    b1 = preds.load_feature(m1.B1_MODEL, "train").join(train.select("traj_id"), on="traj_id", how="semi")
    return train, scope, steps, texts, b1


def issue_texts(texts):
    from afp.baselines import strip_hint
    clean = texts.with_columns(pl.col("issue_text").map_elements(strip_hint, return_dtype=pl.String))
    if clean.group_by("instance_id").agg(pl.col("issue_text").n_unique().alias("n")).filter(pl.col("n") != 1).height:
        raise ValueError("fit issue body is not unique after strip_hint")
    return dict(clean.select("instance_id", "issue_text").unique().sort("instance_id").iter_rows())


def base_features(train):
    """S2 열만 보존. 5c의 검증된 같은 정의로 재구성한다."""
    feats, info = m1t.build_features(train)
    return {n: f.select("traj_id", "N", *s2_cols(n, False)) for n, f in feats.items()}, info


def fold_blocks(D, extras=None):
    """extras[k]는 fold k 어휘로 모든 행을 변환한 행렬. 서로 다른 어휘 열을 합치지 않는다."""
    rows = pl.DataFrame({"instance_id": D["inst"], "fold": D["fold"]})
    oof.check_folds(rows, 5)
    blocks = []
    for fold in range(5):
        fit, held = np.flatnonzero(D["fold"] != fold), np.flatnonzero(D["fold"] == fold)
        if set(D["inst"][fit]) & set(D["inst"][held]):
            raise ValueError("OOF fit/held issue overlap")
        X = D["X"] if extras is None else np.column_stack((D["X"], extras[fold]))
        if not np.isfinite(X).all():
            raise ValueError("nonfinite thought features")
        blocks.append((fit, held, X[fit], X[held]))
    return blocks


def oof_task(key, cfg, y, blocks, device):
    """OOF fold 전용 행렬을 사용. 학습기는 공식 M1과 동일, early stopping 없음."""
    p, seen = np.full(len(y), np.nan), np.zeros(len(y), dtype=np.int8)
    for fit, held, X_fit, X_held in blocks:
        if np.intersect1d(fit, held).size:
            raise ValueError("OOF row overlap")
        model = m1w.fit_xgb(cfg, X_fit, y[fit], device)
        p[held] = m1.predict_model(model, X_held)
        seen[held] += 1
    if not np.all(seen == 1) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid OOF coverage or probabilities")
    return {"key": key, "oof": p, "oof_auroc": float(roc_auc_score(y, p)), "seen_issue_scores": 0}


def fit_grid(D, extras, grid, device, jobs, key):
    blocks = fold_blocks(D, extras)
    rows = evaluate.parallel(oof_task, [((key, i), cfg, D["y"], blocks, device) for i, cfg in enumerate(grid)], jobs)
    rows = [{"i": i, **r} for i, r in enumerate(rows)]
    selected = rows[m1.choose(rows)]
    return selected, rows


def mean_score(disc, bundle, model):
    rows = disc.filter((pl.col("bundle") == bundle) & (pl.col("model") == model) & pl.col("N").is_in(list(SELECT_N)))
    if rows.height != len(SELECT_N) or rows["within_pair"].null_count() or not np.isfinite(rows["within_pair"].to_numpy()).all():
        raise ValueError(f"selection metric incomplete: {bundle}, {model}")
    return float(rows["within_pair"].mean())


def pick_vocab(disc, model):
    scores = {(t, k): mean_score(disc, f"lex_t{t}_k{k}", model) for t, k in SEARCH}
    best = max(scores.values())
    return min((p for p in SEARCH if scores[p] >= best - TIE), key=lambda p: (-p[0], p[1])), scores


def pick_bundle(disc, model, vocab_k):
    scores = {v: mean_score(disc, v, model) for v in BUNDLES}
    best = max(scores.values())
    return min((v for v in BUNDLES if scores[v] >= best - TIE),
               key=lambda v: (len(columns(v, max(evaluate.N_GRID), False, vocab_k)), BUNDLES.index(v))), scores


def prep_vocabulary(corpus, train, scope, texts):
    """진단·잔차 계산은 fit fold만. K=10은 같은 순위 K=20의 접두라 통계를 재사용한다."""
    alive = metrics.running(train, corpus.n).sort("traj_id")
    scope.validate(alive)
    vocabulary, diagnostic_rows, vocab_rows, overlap = {}, [], [], []
    for fold in range(5):
        fit = alive.filter(pl.col("fold") != fold)
        stats = thought.occurrences(corpus, fit.select("traj_id", "instance_id", "fold"), scope, fold, texts)
        diagnostic_rows.extend({"N": corpus.n, "fold": fold, **r} for r in thought.diagnostics(stats))
        scores = thought.association(corpus, fit, scope, fold, stats)
        for t, k in SEARCH:
            entries = thought.select_vocabulary(scores, t, k)
            vocabulary[(fold, t, k)] = entries
            # padding도 목록에 남겨 열 수/실제 어휘 수를 혼동하지 않게 한다.
            vocab_rows.extend({"N": corpus.n, "fold": fold, "t": t, "vocab_k": k,
                               "n_selected": len(entries), "padded": False, **r} for r in entries)
            vocab_rows.extend({"N": corpus.n, "fold": fold, "t": t, "vocab_k": k,
                               "n_selected": len(entries), "padded": True, "rank": j, "ngram": None,
                               "score": None, "n_trajectories": 0, "n_issues": 0, "direction": None}
                              for j in range(len(entries), k))
    for t, k in SEARCH:
        overlap.extend({"N": corpus.n, "t": t, "vocab_k": k, **r}
                       for r in thought.overlap_rows({f: vocabulary[(f, t, k)] for f in range(5)}))
    return vocabulary, diagnostic_rows, vocab_rows, overlap


def prediction_frame(D, p, n):
    return pl.DataFrame({"traj_id": D["traj_id"], "N": np.full(len(p), n, dtype=np.int16), "p_fail": p}).sort("N", "traj_id")


def intervals_complete(frame, stats):
    for name in stats:
        for suffix in ("lo", "hi"):
            col = frame[f"{name}_{suffix}"]
            if col.null_count() or not np.isfinite(col.to_numpy()).all():
                return False
    return True


def distribution_rows(X, names, *, n, fold=None, t=None, vocab_k=None, sample="train", ngrams=None):
    return [{"N": n, "fold": fold, "t": t, "vocab_k": vocab_k, "sample": sample, "feature": name,
             "ngram": None if ngrams is None else ngrams[j], "n": len(X),
             "mean": float(X[:, j].mean()), "std": float(X[:, j].std()),
             "nonzero": float((X[:, j] != 0).mean())} for j, name in enumerate(names)]


def evaluate_final(train, costs, k_star, predictions, disc, jobs, n_boot, skip_bootstrap, bundles=BUNDLES, base=BASE):
    """5b의 공용 평가 연결을 재사용. 동일 재표본, 묶음−기준(5d: S2), K* 대비 차이. 5e도 bundles·base만 바꿔 쓴다."""
    tasks = [(v, m, n, train, predictions[(v, m)].filter(pl.col("N") == n), costs, k_star)
             for v in bundles for m in MODELS for n in evaluate.N_GRID]
    ops = [r for rs in evaluate.parallel(m1w.ops_task, tasks, jobs) for r in rs]
    points = {(r["bundle"], r["model"], r["N"], r["policy"]): (r["tau"], r["cap"]) for r in ops}
    boots = {}
    if not skip_bootstrap:
        tasks = []
        for v in bundles:
            for m in MODELS:
                for n in evaluate.N_GRID:
                    pa, pb = (predictions[(b, m)].filter(pl.col("N") == n) for b in (v, base))
                    a, b = ({pol: points[(x, m, n, pol)] for pol in evaluate.POLICIES} for x in (v, base))
                    c = evaluate.compressed_costs(costs, n, [cap for _, cap in (*a.values(), *b.values())], k_star)
                    tasks.append(((v, m, n), train, evaluate.pair_preds(pa, pb), m1w.oof_stat(n, a, b, k_star, c), n_boot))
        boots = {r["key"]: r["boot"] for r in evaluate.parallel(m1w.boot_task, tasks, jobs)}

    def ci(key, stat):
        if key not in boots:
            return None, None
        row = boots[key].filter(pl.col("stat") == stat).row(0, named=True)
        return row["lo"], row["hi"]

    drows = []
    lookup = {(r["bundle"], r["model"], r["N"]): r for r in disc.iter_rows(named=True)}
    for r in disc.iter_rows(named=True):
        key = r["bundle"], r["model"], r["N"]
        ref = lookup[(base, r["model"], r["N"])]
        row = dict(r)
        for c in ("auroc", "pr_auc", "within_pair", "within_issue_mean"):
            row[f"d_{c}"] = r[c] - ref[c]
            row[f"{c}_lo"], row[f"{c}_hi"] = ci(key, c)
            row[f"d_{c}_lo"], row[f"d_{c}_hi"] = ci(key, f"do_{c}")
        drows.append(row)
    orows = []
    lookup = {(r["bundle"], r["model"], r["N"], r["policy"]): r for r in ops}
    for r in ops:
        key, pol = (r["bundle"], r["model"], r["N"]), r["policy"]
        ref, row = lookup[(base, r["model"], r["N"], pol)], dict(r)
        for c in evaluate.DELTA_COLUMNS:
            row[f"{c}_lo"], row[f"{c}_hi"] = ci(key, f"{pol}.{c}")
            row[f"d_{c}_lo"], row[f"d_{c}_hi"] = ci(key, f"{pol}.dk_{c}")
            row[f"do_{c}"] = r[c] - ref[c]
            row[f"do_{c}_lo"], row[f"do_{c}_hi"] = ci(key, f"{pol}.do_{c}")
        orows.append(row)
    ref = evaluate.cap_point(train, k_star, costs)
    # 기준선에도 같은 train 이슈 부트스트랩의 토큰 차이 구간을 제공한다.
    for pol, cap in (("cap", k_star), ("cap_proposal", evaluate.K_PROPOSAL)):
        pt = evaluate.cap_point(train, cap, costs)
        row = {"bundle": None, "model": "b0_cap", "policy": pol, "N": None, "tau": None, "cap": cap, **pt,
               **{f"d_{c}": pt[c] - ref[c] for c in evaluate.DELTA_COLUMNS},
               **evaluate.fold_spread(train, None, None, None, cap, costs)}
        if not skip_bootstrap:
            compressed = evaluate.compressed_costs(costs, None, (cap,), k_star)
            stat = evaluate.cap_stat((k_star, cap), compressed)
            # cap_stat는 같은 K를 중복받아도 dict 키가 같아 안전하다.
            boot = metrics.bootstrap(train, pl.DataFrame(schema={"traj_id": pl.String, "N": pl.Int16, "p_fail": pl.Float64}),
                                     stat, n_boot=n_boot)
            for c in evaluate.DELTA_COLUMNS:
                name = f"K{cap}.{c}"
                hit = boot.filter(pl.col("stat") == name)
                if hit.height:
                    row[f"{c}_lo"], row[f"{c}_hi"] = hit["lo"][0], hit["hi"][0]
                delta = boot.filter(pl.col("stat") == f"K{cap}.d_{c}")
                row[f"d_{c}_lo"], row[f"d_{c}_hi"] = delta["lo"][0], delta["hi"][0]
        orows.append(row)
    ops_df = pl.DataFrame(orows, infer_schema_length=None, schema_overrides={"N": pl.Int64, "cap": pl.Int64, "tau": pl.Float64})
    ops_df = ops_df.with_columns(meets_select_point=pl.col("lost_success") <= metrics.SELECT_MAX_LOSS + metrics.LOSS_EPS)
    return pl.DataFrame(drows, infer_schema_length=None), ops_df


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="XGBoost 학습 기본 cuda; 사용 불가 시 중단 (--device cpu로 명시적 변경 가능)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--gpu-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    ap.add_argument("--skip-bootstrap", action="store_true", help="개발 전용, 완료 기준 불충족")
    ap.add_argument("--smoke", action="store_true", help="train 이슈 1/10·그리드 2개·부트스트랩 20회; 임시 경로만")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.jobs, args.gpu_jobs, args.n_boot) < 1:
        raise ValueError("jobs and n_boot must be positive")
    upstream = s2_check()
    # 이 지점부터 실험이다. import/--help는 장치 점검·학습하지 않는다.
    started = time.monotonic()
    train, scope, steps, texts, b1 = load_train_inputs()
    if args.smoke:
        issues = train["instance_id"].unique().sort().gather(np.arange(0, train["instance_id"].n_unique(), 10))
        train = train.filter(pl.col("instance_id").is_in(issues.implode()))
        steps = steps.join(train.select("traj_id"), on="traj_id", how="semi")
        texts = texts.join(train.select("traj_id"), on="traj_id", how="semi")
    costs = metrics.eval_costs(train)
    k_star = metrics.select_cap(metrics.fixed_cap_curve(train, costs, range(1, 81)), metrics.SELECT_MAX_LOSS)["K"]
    feats, feat_info = base_features(train)
    bodies = issue_texts(texts)
    grid = m1.XGB_GRID[:m1w.SMOKE_GRID] if args.smoke else m1.XGB_GRID
    jobs = args.gpu_jobs if args.device == "cuda" else args.jobs
    out_dir, data_dir = (SMOKE_DIR, SMOKE_DIR / "oof") if args.smoke else (OUT_DIR, DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    io.write_summary(out_dir / "summary.json", {"mode": "smoke" if args.smoke else "train OOF only",
                                               "status": "incomplete", "criteria": {"execution_finished": False}})
    diagnostic, vocab_rows, overlap, dist, grid_rows = [], [], [], [], []
    prediction_parts = {}

    def run(bundle, model, n, D, extras, *, t=None, vocab_k=None):
        best, rs = fit_grid(D, extras, grid, args.device, jobs, (bundle, model, n))
        nf = D["X"].shape[1] + (0 if extras is None else extras[0].shape[1])
        grid_rows.extend({"bundle": bundle, "model": model, "N": n, "t": t, "vocab_k": vocab_k,
                          "n_features": nf, "i": r["i"], "config": m1.cfg_str(grid[r["i"]]),
                          "oof_auroc": r["oof_auroc"], "selected": r["i"] == best["i"]} for r in rs)
        prediction_parts.setdefault((bundle, model), []).append(prediction_frame(D, best["oof"], n))

    # N별 corpus를 해제. fold 행렬은 numpy 캐시에 두고 모든 N의 큰 어휘를 메모리에 유지하지 않는다.
    for n in evaluate.N_GRID:
        print(f"N={n}: preparing fit-only vocabulary", flush=True)
        corpus = thought.Corpus.build(steps, train, n)
        lengths = features.thought_length_features(steps, train, n, KS)
        features.check_features(lengths.with_columns(N=pl.lit(n)), train, [n], features.thought_length_columns(n, KS))
        feats[n] = feats[n].join(lengths, on="traj_id", how="left")
        vocab, dr, vr, ov = prep_vocabulary(corpus, train, scope, bodies)
        diagnostic.extend(dr)
        vocab_rows.extend(vr)
        overlap.extend(ov)
        alive = metrics.running(train, n).sort("traj_id")
        ids, folds = alive["traj_id"].to_list(), alive["fold"].to_numpy()
        for t, k in SEARCH:
            extras = {f: corpus.transform(ids, vocab[(f, t, k)], k, KS) for f in range(5)}
            for f, X in extras.items():
                # 단어의 실제 정체와 fit/held 분포를 분리해서 기록한다.
                names = [r["ngram"] for r in vocab[(f, t, k)]] + [None] * (k - len(vocab[(f, t, k)]))
                for label, mask in (("fit", folds != f), ("oof_held", folds == f)):
                    dist.extend(distribution_rows(X[mask], thought.lexical_columns(n, k, KS), n=n, fold=f, t=t, vocab_k=k,
                                                  sample=label, ngrams=names * (1 + sum(w < n for w in KS))))
                # 최종 선택 뒤 같은 어휘 행렬을 재사용; 손실 없는 원본 float64.
                np.save(data_dir / f"lex_N{n}_f{f}_t{t}_k{k}.npy", X, allow_pickle=False)
        dist.extend(distribution_rows(lengths.select(features.thought_length_columns(n, KS)).to_numpy(),
                                      features.thought_length_columns(n, KS), n=n))
        del corpus, extras, vocab
        print(f"N={n}: vocabulary preparation complete ({time.monotonic() - started:.1f}s)", flush=True)

    # 모든 N/fold에서 고정 그리드가 성립하는지 실제 모델 학습 전에 검사한다.
    for name, rows in (("vocab_diagnostics", diagnostic), ("vocabulary", vocab_rows), ("vocab_overlap", overlap)):
        pl.DataFrame(rows, infer_schema_length=None).write_csv(out_dir / f"{name}.csv", float_precision=FLOAT)
    if not args.smoke and any(r["padded"] for r in vocab_rows):
        raise ValueError(f"Not enough candidates for fixed vocabulary grid. See {out_dir / 'vocabulary.csv'}. "
                         "No model was trained; report without changing t/K. Only smoke permits zero padding.")
    m1w.check_device(args.device)
    for n in evaluate.N_GRID:
        ids = metrics.running(train, n).sort("traj_id")["traj_id"].to_list()
        for t, k in SEARCH:
            extras = {f: np.load(data_dir / f"lex_N{n}_f{f}_t{t}_k{k}.npy", allow_pickle=False) for f in range(5)}
            for model, use_b1 in MODELS.items():
                D = m1w.design(train, feats, b1 if use_b1 else None, n, s2_cols(n, use_b1))
                if D["traj_id"].to_list() != ids:
                    raise ValueError("lexical/base row ordering mismatch")
                run(f"lex_t{t}_k{k}", model, n, D, extras, t=t, vocab_k=k)
        print(f"N={n}: vocabulary search complete ({time.monotonic() - started:.1f}s)", flush=True)

    def collect():
        return {key: pl.concat(parts).sort("N", "traj_id") for key, parts in prediction_parts.items()}

    def discrimination(predictions):
        return pl.concat([evaluate.discrimination_table(m, p, {"train": train}, evaluate.N_GRID).with_columns(bundle=pl.lit(v))
                          for (v, m), p in predictions.items()])

    p = collect()
    search_disc = discrimination(p)
    vocab_pick = {m: pick_vocab(search_disc, m) for m in MODELS}
    t_star, vocab_k = vocab_pick[SELECT_MODEL][0]
    source = f"lex_t{t_star}_k{vocab_k}"
    print(f"Selected t={t_star}, vocab_k={vocab_k}; fitting four final bundles", flush=True)
    for model in MODELS:
        p[("lexical", model)] = p[(source, model)]
        prediction_parts[("lexical", model)] = [p[(source, model)]]
    for n in evaluate.N_GRID:
        extras = {f: np.load(data_dir / f"lex_N{n}_f{f}_t{t_star}_k{vocab_k}.npy", allow_pickle=False) for f in range(5)}
        for model, use_b1 in MODELS.items():
            for bundle in ("s2", "length", "length_lexical"):
                cs = columns("length" if "length" in bundle else "s2", n, use_b1)
                D = m1w.design(train, feats, b1 if use_b1 else None, n, cs)
                run(bundle, model, n, D, extras if bundle == "length_lexical" else None,
                    t=t_star if bundle == "length_lexical" else None, vocab_k=vocab_k if bundle == "length_lexical" else None)
    p = collect()
    for model in MODELS:
        grid_rows.extend({**r, "bundle": "lexical"} for r in list(grid_rows) if r["bundle"] == source and r["model"] == model)
    final_p = {k: v for k, v in p.items() if k[0] in BUNDLES}
    for (v, m), prediction in final_p.items():
        preds.validate(prediction, {"train": train}, evaluate.N_GRID)
        m1w.write_parquet(prediction, data_dir / f"{m}__{v}.parquet")
    final_disc = discrimination(final_p).with_columns(n_features=pl.struct("bundle", "N", "model").map_elements(
        lambda r: len(columns(r["bundle"], r["N"], MODELS[r["model"]], vocab_k)), return_dtype=pl.Int64))
    disc, ops = evaluate_final(train, costs, k_star, final_p, final_disc, args.jobs,
                               20 if args.smoke else args.n_boot, args.skip_bootstrap)
    selected = {m: pick_bundle(disc, m, vocab_k) for m in MODELS}
    s3 = selected[SELECT_MODEL][0]
    sel_rows = []
    for m in MODELS:
        pair, scores = vocab_pick[m]
        for (t, k), score in scores.items():
            sel_rows.append({"stage": "vocabulary", "model": m, "bundle": f"lex_t{t}_k{k}", "t": t, "vocab_k": k,
                             **{f"within_pair_N{n}": search_disc.filter((pl.col("bundle") == f"lex_t{t}_k{k}") &
                                  (pl.col("model") == m) & (pl.col("N") == n))["within_pair"][0] for n in evaluate.N_GRID},
                             "select_mean": score, "in_tie_band": score >= max(scores.values()) - TIE,
                             "picked_by_this_model": (t, k) == pair, "used": m == SELECT_MODEL and (t, k) == pair})
        bundle, scores = selected[m]
        for v, score in scores.items():
            sel_rows.append({"stage": "S3", "model": m, "bundle": v, "t": t_star if "lexical" in v else None,
                             **{f"within_pair_N{n}": disc.filter((pl.col("bundle") == v) & (pl.col("model") == m) &
                                  (pl.col("N") == n))["within_pair"][0] for n in evaluate.N_GRID},
                             "vocab_k": vocab_k if "lexical" in v else None, "select_mean": score,
                             "d_select_mean_vs_s2": score - scores[BASE], "n_features_N20": len(columns(v, 20, False, vocab_k)),
                             "in_tie_band": score >= max(scores.values()) - TIE,
                             "picked_by_this_model": v == bundle, "used": m == SELECT_MODEL and v == bundle})
    repro = reproduce_s2(final_p, upstream, args)
    hyp = disc.filter((pl.col("bundle") == s3) & pl.col("N").is_in(list(SELECT_N))).select(
        "model", "N", "d_within_pair", "d_within_pair_lo", "d_within_pair_hi").to_dicts()
    expected = len(BUNDLES) * len(MODELS) * len(evaluate.N_GRID)
    model_ops = ops.filter(pl.col("model") != "b0_cap")
    crit = {"C1_features": feat_info["official_matches_04_m1_cache"] is not False,
            "C2_fold_boundaries": True, "C3_s2_reproduced_when_available": all(r is None or r["matches_when_same_device"] for r in repro.values()),
            "C4_tables_complete": disc.height == expected and model_ops.height == expected * len(evaluate.POLICIES),
            "C5_vocabulary_complete": args.smoke or all(not r["padded"] for r in vocab_rows),
            "C6_bootstrap_ci": not args.skip_bootstrap and (args.smoke or args.n_boot == evaluate.N_BOOT)
            and intervals_complete(ops, [f"{prefix}{c}" for prefix in ("", "d_") for c in evaluate.DELTA_COLUMNS])
            and intervals_complete(model_ops, [f"do_{c}" for c in evaluate.DELTA_COLUMNS])
            and intervals_complete(disc, [f"{prefix}{c}" for prefix in ("", "d_") for c in ("auroc", "within_pair")])}
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"vocab_diagnostics": diagnostic, "vocabulary": vocab_rows, "vocab_overlap": overlap,
              "grid": grid_rows, "feature_dist": dist, "selection": sel_rows}
    for name, rows in tables.items():
        pl.DataFrame(rows, infer_schema_length=None).write_csv(out_dir / f"{name}.csv", float_precision=FLOAT)
    disc.write_csv(out_dir / "discrimination_oof.csv", float_precision=FLOAT)
    search_disc.write_csv(out_dir / "vocabulary_search_oof.csv", float_precision=FLOAT)
    ops.write_csv(out_dir / "operating_points_oof.csv", float_precision=FLOAT)
    ops.filter(pl.col("policy") == "combined").select(
        "bundle", "model", "N", "lost_success", "saved_steps", "saved_tokens", "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi",
        "d_saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi", "do_saved_steps", "do_saved_tokens",
        "do_saved_tokens_lo", "do_saved_tokens_hi").write_csv(out_dir / "n_summary.csv", float_precision=FLOAT)
    summary = {"mode": "smoke" if args.smoke else "train OOF only; full execution by user", "status": "complete" if all(crit.values()) else "incomplete", "device": args.device,
               "seed": io.SEED, "thought_version": thought.VERSION, "s2": {**upstream, "definition": f"w3 + {'+'.join(S2[1]) if S2[1] else S2[0]}", "reproduce_5c_oof": repro},
               "models": list(MODELS), "N": list(evaluate.N_GRID), "grid": list(grid), "xgb_fixed": m1.XGB_FIXED,
               "grid_tie": m1.GRID_TIE, "k_star": k_star, "features": feat_info,
               "selection": {"model": SELECT_MODEL, "N": list(SELECT_N), "tie": TIE, "t": t_star, "vocab_k": vocab_k,
                             "S3": s3, "pick_by_model": {m: {"vocabulary": list(vocab_pick[m][0]), "S3": selected[m][0]} for m in MODELS}},
               "candidates_seen": {"vocab_pairs": len(SEARCH), "final_bundles": len(BUNDLES), "xgb_configs": len(grid)},
               "unique_designs": sum(r["bundle"] != "lexical" for r in grid_rows) // len(grid),
               "grid_fits": sum(r["bundle"] != "lexical" for r in grid_rows),
               "hypothesis": {"rows": hyp, "n_positive": sum(r["d_within_pair"] > 0 for r in hyp),
                              "n_ci_lo_positive": sum(r["d_within_pair_lo"] is not None and r["d_within_pair_lo"] > 0 for r in hyp)},
               "bootstrap": {"n_boot": 20 if args.smoke else args.n_boot, "skipped": args.skip_bootstrap,
                             "unit": "train issue", "seed": io.SEED, "base": BASE, "selection_uncertainty_included": False},
               "criteria": crit, "external_validation": "pytest and byte reproducibility are separate; not implied by script criteria"}
    io.write_summary(out_dir / "summary.json", summary)
    # 최종 평가·5e 재학습에 필요한 fold별 어휘/열·설정·train 행 지문을 보존한다.
    def fingerprint(frame, sort):
        return hashlib.sha256(frame.sort(sort).write_json().encode()).hexdigest()

    metadata = {"summary": summary, "train_rows": train.select("traj_id", "instance_id", "fold").to_dicts(),
                "fingerprints": {"train": fingerprint(train, "traj_id"), "prefix_steps": fingerprint(steps, ["traj_id", "step_idx"]),
                                 "b1_train": fingerprint(b1.join(train.select("traj_id"), on="traj_id", how="semi"), ["N", "traj_id"])},
                "columns": {v: {str(n): columns(v, n, False, vocab_k) for n in evaluate.N_GRID} for v in BUNDLES},
                "vocabulary": [r for r in vocab_rows if r["t"] == t_star and r["vocab_k"] == vocab_k],
                "selected_xgb": [r for r in grid_rows if r["bundle"] in BUNDLES and r["selected"]]}
    io.write_summary(data_dir / "handoff.json", metadata)
    print(pl.DataFrame(sel_rows, infer_schema_length=None))
    print(ops.select("bundle", "model", "policy", "N", "saved_steps", "saved_tokens", "d_saved_steps",
                     "d_saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi"))
    print(f"S3={s3}; criteria={crit}; elapsed={time.monotonic() - started:.1f}s", flush=True)
    if not all(crit.values()):
        raise SystemExit("Phase 5d script criteria incomplete; do not proceed to 5e")


def reproduce_s2(predictions, upstream, args):
    out = {}
    for model in MODELS:
        path = m1t.DATA_DIR / f"{model}__{upstream['phase5c_S2']}.parquet"
        if args.smoke or not path.exists():
            out[model] = None
            continue
        old, new = pl.read_parquet(path).sort("N", "traj_id"), predictions[(BASE, model)]
        same_rows = old.select("traj_id", "N").cast({"N": pl.Int64}).equals(new.select("traj_id", "N").cast({"N": pl.Int64}))
        same_scores = bool(same_rows and np.array_equal(old["p_fail"].to_numpy(), new["p_fail"].to_numpy()))
        out[model] = {"same_rows": same_rows, "same_scores": same_scores,
                      "matches_when_same_device": same_rows and (upstream["phase5c_device"] != args.device or same_scores)}
    return out


if __name__ == "__main__":
    main()
