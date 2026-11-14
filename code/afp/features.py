"""M1 피처 (Phase 5.1). 스텝 단위 값과 궤적 단위(앞 N스텝 집계) 피처를 나눈다.

스텝 단위 (step_values): 스텝 하나와 그 앞 스텝들만 보고 정해지는 값. M2(Phase 6, LSTM)가 같은 값을 시퀀스 입력으로
쓰므로 M1과 M2의 정보가 같은 정의에서 나온다. 스텝 t의 값은 스텝 1..t만 본다 (인과적).
- command_name: 명령 종류 (parse.COMMANDS 26개 + other + none, Phase 1 사전 그대로)
- is_error, edit_failed: B0 규칙과 같은 정의 (steps.parquet에 Phase 1 parse.is_error / parse.EDIT_FAILED로 저장된 값을 그대로 읽는다)
- obs_len: 관측 문자 수 (관측 없음 = 0)
- open_file: 관측 끝 상태 줄의 열린 파일 경로 (없거나 n/a면 null)
- repeat_prev: 바로 앞 스텝과 명령 원문(앞뒤 공백 제거)이 같다
- repeat_any: 앞 스텝들 중 어딘가에 같은 명령 원문이 있다
- run_len: 이 스텝에서 끝나는 "같은 명령 연속" 길이 (반복이 아니면 1)
- opens_issue_file: open_file이 이슈 본문에 언급된 파일이다 (issue_file_mentions 규칙, 이슈 본문은 운영 중에도 보인다)

궤적 단위 (m1_features): 시점 N 진행 중 궤적(n_steps > N)마다 앞 N스텝(step_idx <= N)의 스텝 단위 값을 집계한다.
진행 중 궤적은 모두 정확히 N스텝을 봤으므로 "횟수 / N" 비율 피처는 같은 N 안에서 횟수와 정보가 같다 (N별 모델, D2).
그래서 비율 피처는 넣지 않는다. 같은 이유로 같은 N 안에서 다른 피처의 선형 변환인 값도 뺐다:
"고유 명령 수" = N - n_repeat_any, "관측 길이 합" = N × obs_len_mean.

누수: 이 모듈은 io.scan_steps / io.load_trajectories로 컬럼을 명시해 읽고 누수 컬럼(io.LEAKY_COLUMNS)을 읽지 않는다.
정답 패치에서 나온 정보(정답 파일 열람 등)는 쓰지 않는다 (사후 분석은 afp/leakage.py).
"""
import re

import polars as pl

from afp import baselines, io, metrics, parse

STEP_INPUT_COLUMNS = ["traj_id", "step_idx", "command", "command_name", "obs_len", "is_error", "edit_failed", "open_file"]
COMMAND_NAMES = (*parse.COMMANDS, "other", "none")
STEP_VALUE_COLUMNS = ["traj_id", "step_idx", "command_name", "is_error", "edit_failed", "obs_len", "open_file",
                      "repeat_prev", "repeat_any", "run_len", "opens_issue_file"]

# 이슈 본문에서 파일 경로로 보이는 토큰 (확장자 목록은 저장소 코드·설정 파일). 앞에 경로 문자가 없어야 한다
FILE_MENTION = re.compile(
    r"(?<![\w/.-])((?:[\w.-]+/)*[\w-][\w.-]*\.(?:py|pyx|pyi|cfg|toml|ini|json|ya?ml|txt|rst|md|c|h|cpp|js|ts|html|css))\b")


def _cmd(name: str) -> str:
    return f"cmd_{name}"


def _last(name: str) -> str:
    return f"last_cmd_{name}"


COUNT_FEATURES = [_cmd(c) for c in COMMAND_NAMES]
LAST_CMD_FEATURES = [_last(c) for c in COMMAND_NAMES]
FEATURES = [
    *COUNT_FEATURES,
    "n_error", "n_edit_failed", "n_edit_ok",
    "n_repeat_consec", "n_repeat_any", "max_run_len",
    "n_unique_files", "n_steps_no_file",
    "obs_len_mean", "obs_len_max", "obs_len_min", "obs_len_last",
    "issue_mentions_file", "issue_file_opened", "n_issue_file_steps",
    "last_is_error", "last_edit_failed", *LAST_CMD_FEATURES,
]
B1_FEATURE = "b1_score"


# ---- 스텝 단위 ----

def issue_file_mentions(text: str | None) -> list[str]:
    """이슈 본문에 나온 파일 경로 토큰 (중복 없이 정렬). 경로가 있으면 경로 그대로, 없으면 파일 이름만.

    이슈 13개의 본문 끝에 붙은 "고칠 파일 목록"(baselines.HINT, 정답 패치 정보)은 먼저 지운다 (B1 입력과 같은 정리).
    """
    if not text:
        return []
    return sorted(set(FILE_MENTION.findall(baselines.strip_hint(text))))


def mentions_table(trajs: pl.DataFrame) -> pl.DataFrame:
    """traj_id, instance_id, issue_text → 이슈별 언급 파일 (instance_id, mention). 언급 없는 이슈는 행이 없다."""
    issues = trajs.select("instance_id", "issue_text").unique("instance_id").sort("instance_id")
    return (issues.with_columns(mention=pl.col("issue_text").map_elements(issue_file_mentions, return_dtype=pl.List(pl.String)))
            .select("instance_id", "mention").explode("mention").drop_nulls("mention"))


def _matches(open_file: pl.Expr, mention: pl.Expr) -> pl.Expr:
    """열린 파일이 언급과 맞는지: 언급에 '/'가 있으면 경로 끝이 '/언급'(또는 같음), 없으면 파일 이름이 같다."""
    with_dir = mention.str.contains("/", literal=True)
    base = open_file.str.split("/").list.last()
    return pl.when(with_dir).then((open_file == mention) | open_file.str.ends_with(pl.lit("/") + mention)).otherwise(base == mention)


def step_values(steps: pl.DataFrame, mentions: pl.DataFrame | None = None, traj_issue: pl.DataFrame | None = None) -> pl.DataFrame:
    """스텝 단위 값 (STEP_VALUE_COLUMNS). steps: STEP_INPUT_COLUMNS를 가진 스텝 행 (몇 스텝까지 넣든 스텝 t 값은 1..t만 본다).

    mentions(instance_id, mention)와 traj_issue(traj_id, instance_id)가 있으면 opens_issue_file을 계산하고, 없으면 false.
    출력은 (traj_id, step_idx) 정렬.
    """
    s = steps.select(STEP_INPUT_COLUMNS).sort("traj_id", "step_idx")
    cmd = pl.col("command").fill_null("").str.strip_chars()
    s = s.with_columns(_cmd=cmd)
    prev = pl.col("_cmd").shift(1).over("traj_id")
    s = s.with_columns(
        repeat_prev=(pl.col("_cmd") == prev).fill_null(False),
        # 같은 명령의 몇 번째 등장인지 (1부터). 2 이상이면 앞 스텝 어딘가에 같은 명령이 있었다
        repeat_any=(pl.col("step_idx").cum_count().over("traj_id", "_cmd") > 1),
    )
    # 연속 구간 번호: repeat_prev가 아닌 스텝에서 새 구간이 시작된다
    s = s.with_columns(_run=(~pl.col("repeat_prev")).cast(pl.Int32).cum_sum().over("traj_id"))
    s = s.with_columns(run_len=pl.col("step_idx").cum_count().over("traj_id", "_run").cast(pl.Int32))
    if mentions is not None and traj_issue is not None and mentions.height:
        pairs = (s.filter(pl.col("open_file").is_not_null()).select("traj_id", "open_file").unique()
                 .join(traj_issue.select("traj_id", "instance_id"), on="traj_id", how="inner")
                 .join(mentions, on="instance_id", how="inner")
                 .filter(_matches(pl.col("open_file"), pl.col("mention")))
                 .select("traj_id", "open_file").unique().with_columns(opens_issue_file=pl.lit(True)))
        s = s.join(pairs, on=["traj_id", "open_file"], how="left").with_columns(pl.col("opens_issue_file").fill_null(False))
    else:
        s = s.with_columns(opens_issue_file=pl.lit(False))
    return s.select(STEP_VALUE_COLUMNS).sort("traj_id", "step_idx")


def load_step_values(max_n: int, trajs: pl.DataFrame | None = None) -> pl.DataFrame:
    """steps.parquet 앞 max_n스텝의 스텝 단위 값 (모든 궤적 또는 trajs 궤적). 이슈 본문은 trajectories.parquet에서 읽는다."""
    lf = io.scan_steps(STEP_INPUT_COLUMNS).filter(pl.col("step_idx") <= max_n)
    if trajs is not None:
        lf = lf.join(trajs.select("traj_id").lazy(), on="traj_id", how="semi")
    steps = lf.collect()
    t = io.load_trajectories(["traj_id", "instance_id", "issue_text"])
    if trajs is not None:
        t = t.join(trajs.select("traj_id"), on="traj_id", how="semi")
    return step_values(steps, mentions_table(t), t.select("traj_id", "instance_id"))


# ---- 궤적 단위 (M1) ----

def m1_features(values: pl.DataFrame, trajs: pl.DataFrame, n_grid, mentions: pl.DataFrame | None = None) -> pl.DataFrame:
    """스텝 단위 값 → (traj_id, N, FEATURES). 시점 N 진행 중 궤적(n_steps > N)마다 한 행, 앞 N스텝만 집계한다.

    trajs: traj_id, instance_id, n_steps. mentions가 있으면 issue_mentions_file(이슈에 파일 언급이 있는지)을 채운다.
    모든 피처는 Float64이고 결측이 없다 (진행 중 궤적은 N스텝을 모두 가진다).
    """
    has_mention = (trajs.select("traj_id", "instance_id")
                   .join(mentions.select("instance_id").unique(), on="instance_id", how="semi")
                   .select("traj_id", issue_mentions_file=pl.lit(1.0))) if mentions is not None else None
    out = []
    for n in n_grid:
        alive = metrics.running(trajs.select("traj_id", "n_steps"), n).select("traj_id")
        v = values.filter(pl.col("step_idx") <= n).join(alive, on="traj_id", how="semi")
        agg = v.group_by("traj_id").agg(
            *[(pl.col("command_name") == c).sum().cast(pl.Float64).alias(_cmd(c)) for c in COMMAND_NAMES],
            n_error=pl.col("is_error").sum().cast(pl.Float64),
            n_edit_failed=pl.col("edit_failed").sum().cast(pl.Float64),
            n_edit_ok=((pl.col("command_name") == "edit") & ~pl.col("edit_failed")).sum().cast(pl.Float64),
            n_repeat_consec=pl.col("repeat_prev").sum().cast(pl.Float64),
            n_repeat_any=pl.col("repeat_any").sum().cast(pl.Float64),
            max_run_len=pl.col("run_len").max().cast(pl.Float64),
            n_unique_files=pl.col("open_file").drop_nulls().n_unique().cast(pl.Float64),
            n_steps_no_file=pl.col("open_file").is_null().sum().cast(pl.Float64),
            obs_len_mean=pl.col("obs_len").cast(pl.Float64).mean(),
            obs_len_max=pl.col("obs_len").max().cast(pl.Float64),
            obs_len_min=pl.col("obs_len").min().cast(pl.Float64),
            obs_len_last=pl.col("obs_len").sort_by("step_idx").last().cast(pl.Float64),
            issue_file_opened=pl.col("opens_issue_file").any().cast(pl.Float64),
            n_issue_file_steps=pl.col("opens_issue_file").sum().cast(pl.Float64),
            last_is_error=pl.col("is_error").sort_by("step_idx").last().cast(pl.Float64),
            last_edit_failed=pl.col("edit_failed").sort_by("step_idx").last().cast(pl.Float64),
            _last_cmd=pl.col("command_name").sort_by("step_idx").last(),
            _n=pl.len(),
        )
        bad = agg.filter(pl.col("_n") != n).height
        if bad or agg.height != alive.height:
            raise ValueError(f"N={n}: 앞 N스텝이 모두 있지 않은 진행 중 궤적 {bad}개 / 행 {agg.height} != 진행 중 {alive.height}")
        agg = agg.with_columns([(pl.col("_last_cmd") == c).cast(pl.Float64).alias(_last(c)) for c in COMMAND_NAMES])
        if has_mention is not None:
            agg = agg.join(has_mention, on="traj_id", how="left").with_columns(pl.col("issue_mentions_file").fill_null(0.0))
        else:
            agg = agg.with_columns(issue_mentions_file=pl.lit(0.0))
        out.append(agg.with_columns(N=pl.lit(n, dtype=pl.Int16)).select("traj_id", "N", *FEATURES))
    return pl.concat(out).sort("N", "traj_id")


def check_features(feats: pl.DataFrame, trajs: pl.DataFrame, n_grid, columns=None) -> dict:
    """실데이터 점검: 모든 N에서 진행 중 궤적과 1:1, 결측·무한값 없음. 출력 {N: 행 수}.

    columns: 검사할 피처 컬럼 (기본 FEATURES). N마다 다르면 함수 columns(n) -> 컬럼 목록 (Phase 5b 창 피처)."""
    counts = {}
    for n in n_grid:
        f = feats.filter(pl.col("N") == n)
        alive = metrics.running(trajs, n)
        if f["traj_id"].n_unique() != f.height or f.height != alive.height or f.join(alive, on="traj_id", how="anti").height:
            raise ValueError(f"N={n}: 피처 행이 진행 중 궤적과 1:1이 아님 ({f.height} vs {alive.height})")
        cols = FEATURES if columns is None else (columns(n) if callable(columns) else list(columns))
        x = f.select(cols)
        nulls = x.null_count().sum_horizontal().item()
        nonfinite = x.select(pl.any_horizontal(~pl.all().is_finite())).to_series().sum()
        if nulls or nonfinite:
            raise ValueError(f"N={n}: 결측 {nulls}개, 무한값 행 {nonfinite}개")
        counts[n] = f.height
    return counts


# ---- Phase 5b: 마지막 k스텝 창·최근성 (PLAN 11절 P5b-D4~D6, docs/phase5b.md) ----
# 창 k = 스텝 N−k+1..N. k ≥ N이면 그 k의 창 피처는 만들지 않는다 (N=3은 창 없음, N=5는 k=3만, P5b-D4).
# 창 피처는 k마다 7개, 원시 횟수만 (비율·차이·감쇠·기울기는 5c, P5b-D5). 최근성 3개는 창과 무관하다 (P5b-D6).
# 모두 스텝 1..N의 스텝 단위 값(step_values)에서만 계산한다. 스텝 N 이후, 궤적 길이(n_steps)는 쓰지 않는다
# (n_steps는 "N에 진행 중인가"만 정한다).

WINDOW_KS = (3, 5)
WINDOW_BASES = ("n_error", "n_edit_failed", "n_edit_ok", "n_repeat_consec", "n_repeat_any", "n_new_files", "obs_len_mean")
RECENCY_FEATURES = ["steps_since_edit_ok", "steps_since_new_file", "cur_error_run"]


def window_name(base: str, k: int) -> str:
    return f"{base}_w{k}"


def window_columns(n: int, ks) -> list[str]:
    """시점 N에 쓸 창 피처 이름. k ≥ N인 k는 뺀다 (k 순서, k 안에서는 WINDOW_BASES 순서)."""
    return [window_name(b, k) for k in ks if k is not None and k < n for b in WINDOW_BASES]


def proportional_k(n: int) -> int | None:
    """비례 창 k = ⌈N/4⌉ (P5b-D8 확인용). k=1이거나 k ≥ N이면 창 없음 (None): 고정 k와 조건을 맞춘다."""
    k = -(-n // 4)
    return k if 2 <= k < n else None


def window_features(values: pl.DataFrame, trajs: pl.DataFrame, n: int, ks=WINDOW_KS, recency: bool = True) -> pl.DataFrame:
    """시점 N 진행 중 궤적(n_steps > N)마다 창 피처(k < N인 k만)와 최근성 피처. 출력 (traj_id, 피처…), traj_id 정렬, Float64.

    values: step_values 출력 (몇 스텝까지 들어 있든 step_idx ≤ N만 쓴다). trajs: traj_id, n_steps.
    - n_error_w{k}, n_edit_failed_w{k}, n_repeat_consec_w{k}, n_repeat_any_w{k}: 창 안 스텝 값의 합 (repeat_any는 창 이전
      명령과 같아도 센다)
    - n_edit_ok_w{k}: 창 안 성공 편집(edit 명령이고 edit_failed가 아님) 수
    - n_new_files_w{k}: 창 시작 전에 한 번도 열리지 않았고 창 안에서 처음 열린 파일 수 (open_file 첫 등장 스텝이 창 안)
    - obs_len_mean_w{k}: 창 안 관측 길이 평균
    - steps_since_edit_ok: N − 마지막 성공 편집 스텝 (없으면 N), steps_since_new_file: N − 마지막으로 새 파일을 연 스텝 (없으면 N)
    - cur_error_run: 스텝 N에서 끝나는 연속 에러 스텝 수 (스텝 N이 에러가 아니면 0)
    """
    ks = sorted({k for k in ks if k is not None and k < n})
    alive = metrics.running(trajs.select("traj_id", "n_steps"), n).select("traj_id")
    v = (values.filter(pl.col("step_idx") <= n).join(alive, on="traj_id", how="semi").sort("traj_id", "step_idx")
         .with_columns(_edit_ok=(pl.col("command_name") == "edit") & ~pl.col("edit_failed"),
                       _first=pl.col("step_idx").cum_count().over("traj_id", "open_file")))
    v = v.with_columns(_new_file=pl.col("open_file").is_not_null() & (pl.col("_first") == 1))
    aggs = []
    for k in ks:
        w = pl.col("step_idx") > n - k
        aggs += [
            (pl.col("is_error") & w).sum().cast(pl.Float64).alias(window_name("n_error", k)),
            (pl.col("edit_failed") & w).sum().cast(pl.Float64).alias(window_name("n_edit_failed", k)),
            (pl.col("_edit_ok") & w).sum().cast(pl.Float64).alias(window_name("n_edit_ok", k)),
            (pl.col("repeat_prev") & w).sum().cast(pl.Float64).alias(window_name("n_repeat_consec", k)),
            (pl.col("repeat_any") & w).sum().cast(pl.Float64).alias(window_name("n_repeat_any", k)),
            (pl.col("_new_file") & w).sum().cast(pl.Float64).alias(window_name("n_new_files", k)),
            pl.col("obs_len").filter(w).cast(pl.Float64).mean().alias(window_name("obs_len_mean", k)),
        ]
    if recency:
        aggs += [
            (n - pl.col("step_idx").filter(pl.col("_edit_ok")).max().fill_null(0)).cast(pl.Float64).alias("steps_since_edit_ok"),
            (n - pl.col("step_idx").filter(pl.col("_new_file")).max().fill_null(0)).cast(pl.Float64).alias("steps_since_new_file"),
            (n - pl.col("step_idx").filter(~pl.col("is_error")).max().fill_null(0)).cast(pl.Float64).alias("cur_error_run"),
        ]
    agg = v.group_by("traj_id").agg(*aggs, _n=pl.len(), _max=pl.col("step_idx").max())
    bad = agg.filter((pl.col("_n") != n) | (pl.col("_max") != n)).height
    if bad or agg.height != alive.height:
        raise ValueError(f"N={n}: 앞 N스텝이 모두 있지 않은 진행 중 궤적 {bad}개 / 행 {agg.height} != 진행 중 {alive.height}")
    cols = window_columns(n, ks) + (RECENCY_FEATURES if recency else [])
    return agg.select("traj_id", *cols).sort("traj_id")
