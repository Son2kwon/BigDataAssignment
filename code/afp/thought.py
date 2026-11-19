"""Phase 5d fold 안 생각 어휘. 어휘 fit은 train 명부·OOF fold 경계를 검사한다.

선택 문서는 앞 N스텝을 이어 붙인 단어열 (스텝 경계 bigram 포함), 모델 피처는
스텝 안 존재 횟수다. 따라서 경계에서만 생긴 bigram은 피처가 0일 수 있다 (P5d-D4·D12).
라벨 없는 Corpus/diagnostics와 라벨을 받는 association을 분리한다.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
import re

import numpy as np
import polars as pl

from afp import baselines, features

VERSION = 1
DIAGNOSTIC_T = (5, 10, 20, 30, 50, 100)
SEARCH_T = (10, 30, 100)
SEARCH_K = (10, 20)
# 공백 조각 단위로 경로·숫자를 먼저 거른다. 버린 토큰은 bigram 경계가 된다.
PATH = re.compile(r"/|\\|\.[a-zA-Z][a-zA-Z0-9_]*\b")
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
REPO = re.compile(r"^(.+__.+)-\d+$")


def token_segments(text: str) -> tuple[tuple[str, ...], ...]:
    """숫자 포함 조각·경로 전체 제거. 나머지는 단어/축약어, 소문자. 문장부호도 경계."""
    segments, current = [], []
    for chunk in text.lower().split():
        if PATH.search(chunk) or any(c.isdigit() for c in chunk):
            if current:
                segments.append(tuple(current))
                current = []
            continue
        matches = list(WORD.finditer(chunk))
        cursor = 0
        for match in matches:
            if match.start() > cursor and current:
                segments.append(tuple(current))
                current = []
            current.append(match.group())
            cursor = match.end()
        if cursor < len(chunk) and current:
            segments.append(tuple(current))
            current = []
    if current:
        segments.append(tuple(current))
    return tuple(segments)


def ngrams(segments) -> frozenset[str]:
    out = set()
    for segment in segments:
        out.update(segment)
        out.update(f"{a} {b}" for a, b in zip(segment, segment[1:]))
    return frozenset(out)


def repo_id(instance_id: str) -> str:
    match = REPO.fullmatch(instance_id)
    if match is None:
        raise ValueError(f"instance_id is not owner__repo-number: {instance_id!r}")
    return match.group(1)


@dataclass
class TrainScope:
    """신뢰하는 train 명부 (라벨 없음). 실제 호출자는 metrics.eval_set('train')에서 만든다."""
    registry: pl.DataFrame

    def __post_init__(self):
        if "split" in self.registry.columns and self.registry.filter(pl.col("split") != "train").height:
            raise ValueError("registry contains non-train rows")
        self.registry = self.registry.select("traj_id", "instance_id", "fold").sort("traj_id")
        if (self.registry["traj_id"].n_unique() != self.registry.height
                or self.registry.null_count().sum_horizontal().item()
                or self.registry.filter(~pl.col("fold").is_in(list(range(5)))).height
                or self.registry.group_by("instance_id").agg(pl.col("fold").n_unique().alias("n")).filter(pl.col("n") != 1).height):
            raise ValueError("invalid train registry (IDs, folds or issue groups)")

    def validate(self, rows: pl.DataFrame, held_fold: int | None = None) -> None:
        if "split" in rows.columns and (rows["split"].null_count() or rows.filter(pl.col("split") != "train").height):
            raise ValueError("test rows cannot enter vocabulary fit")
        if rows.is_empty() or rows["traj_id"].n_unique() != rows.height:
            raise ValueError("empty/duplicate fit rows")
        actual = rows.select("traj_id", "instance_id", "fold")
        if actual.null_count().sum_horizontal().item() or actual.join(self.registry, on=actual.columns, how="anti").height:
            raise ValueError("test, unknown ID, or forged issue/fold in fit input")
        if held_fold is not None:
            if held_fold not in range(5) or rows.filter(pl.col("fold") == held_fold).height:
                raise ValueError("held-out OOF fold cannot enter vocabulary fit")


@dataclass
class Corpus:
    """n-gram 문자열은 `names`/`ids`에 한 번만 두고 문서·스텝은 정수 ID 집합으로 든다 (메모리).

    ID는 문자열 사전순 순위다. 따라서 ID 정렬 = 문자열 정렬이고, 정렬·동률 처리·합산 순서가
    문자열 집합을 쓰던 때와 같다.
    """
    n: int
    names: tuple[str, ...]
    ids: dict[str, int]
    documents: dict[str, frozenset[int]]
    steps: dict[str, tuple[frozenset[int], ...]]

    @classmethod
    def build(cls, steps: pl.DataFrame, trajs: pl.DataFrame, n: int):
        prefix = features.thought_steps(steps, trajs, n)
        texts = defaultdict(list)
        for row in prefix.select("traj_id", "thought").iter_rows():
            texts[row[0]].append(row[1])
        temp = {}  # 문자열 → 임시 ID (등장 순)

        def encode(grams):
            return frozenset(temp.setdefault(g, len(temp)) for g in grams)

        documents, per_step = {}, {}
        for tid, pieces in texts.items():
            documents[tid] = encode(ngrams(token_segments(" ".join(pieces))))
            per_step[tid] = tuple(encode(ngrams(token_segments(piece))) for piece in pieces)
        names = tuple(sorted(temp))
        ids = {g: i for i, g in enumerate(names)}
        remap = [0] * len(names)
        for g, old in temp.items():
            remap[old] = ids[g]
        del temp
        for tid in documents:
            documents[tid] = frozenset(remap[i] for i in documents[tid])
            per_step[tid] = tuple(frozenset(remap[i] for i in s) for s in per_step[tid])
        return cls(n, names, ids, documents, per_step)

    def encode(self, grams) -> frozenset[int]:
        """외부 문자열 집합을 ID로. 코퍼스에 없는 n-gram은 어떤 문서에도 없으므로 버린다."""
        return frozenset(self.ids[g] for g in grams if g in self.ids)

    def document_ngrams(self, tid: str) -> frozenset[str]:
        return frozenset(self.names[i] for i in self.documents[tid])

    def transform(self, ids, vocabulary, vocab_k: int, ks=(3,)) -> np.ndarray:
        """라벨 없이 변환. 열은 rank 순 합계, 창 k 순 rank. 부족 어휘 열은 0 패딩."""
        if vocab_k < len(vocabulary):
            raise ValueError("vocab_k smaller than vocabulary")
        windows = [k for k in sorted(set(ks)) if k < self.n]
        out = np.zeros((len(ids), vocab_k * (1 + len(windows))), dtype=np.float64)
        codes = [self.ids.get(entry["ngram"], -1) for entry in vocabulary]  # -1: 어느 스텝에도 없음
        for i, tid in enumerate(ids):
            present = self.steps[tid]
            for j, g in enumerate(codes):
                out[i, j] = sum(g in s for s in present)
                for w, k in enumerate(windows, 1):
                    out[i, w * vocab_k + j] = sum(g in s for s in present[-k:])
        return out


def lexical_columns(n: int, vocab_k: int, ks=(3,)) -> list[str]:
    return [*[f"thought_g{j:02d}" for j in range(vocab_k)],
            *[f"thought_g{j:02d}_w{k}" for k in sorted(set(ks)) if k < n for j in range(vocab_k)]]


@dataclass
class Occurrences:
    trajectory_count: Counter
    issue_sets: dict
    repo_counts: dict
    overlaps_issue_text: set
    fit_ids: tuple[str, ...]
    held_fold: int


def occurrences(corpus: Corpus, fit_meta: pl.DataFrame, scope: TrainScope, held_fold: int,
                issue_text: dict[str, str]) -> Occurrences:
    """라벨 없는 fit 전용 진단 통계. 등장 횟수 = 궤적 존재 횟수."""
    scope.validate(fit_meta, held_fold)
    if "target" in fit_meta.columns:
        raise ValueError("unlabelled diagnostics must not receive target")
    count, issues, repos, overlap = Counter(), defaultdict(set), defaultdict(Counter), set()
    issue_grams = {issue: corpus.encode(ngrams(token_segments(baselines.strip_hint(issue_text[issue]))))
                   for issue in sorted(set(fit_meta["instance_id"].to_list()))}
    for tid, issue in fit_meta.select("traj_id", "instance_id").sort("traj_id").iter_rows():
        repo = repo_id(issue)
        for g in sorted(corpus.documents[tid]):
            count[g] += 1
            issues[g].add(issue)
            repos[g][repo] += 1
            if g in issue_grams[issue]:
                overlap.add(g)
    return Occurrences(count, dict(issues), dict(repos), overlap, tuple(sorted(fit_meta["traj_id"].to_list())), held_fold)


def diagnostics(stats: Occurrences, thresholds=DIAGNOSTIC_T) -> list[dict]:
    rows = []
    for t in thresholds:
        candidates = sorted(g for g, issues in stats.issue_sets.items() if len(issues) >= t)
        focused = sum(max(stats.repo_counts[g].values()) / stats.trajectory_count[g] > 0.5 for g in candidates)
        overlap = sum(g in stats.overlaps_issue_text for g in candidates)
        den = len(candidates)
        rows.append({"t": t, "n_candidates": den, "repo_focused_n": focused,
                     "repo_focused_ratio": focused / den if den else None,
                     "issue_overlap_n": overlap, "issue_overlap_ratio": overlap / den if den else None,
                     "repo_parse_failures": 0})
    return rows


def association(corpus: Corpus, fit: pl.DataFrame, scope: TrainScope, held_fold: int,
                stats: Occurrences) -> list[dict]:
    """fit 이슈 안 실패 잔차. 어휘 후보는 fit 통계만, 채점 fold 라벨은 받지 않는다."""
    scope.validate(fit, held_fold)
    if stats.fit_ids != tuple(sorted(fit["traj_id"].to_list())) or stats.held_fold != held_fold:
        raise ValueError("association received statistics from different fit rows/fold")
    if fit["target"].dtype != pl.Boolean or fit["target"].null_count():
        raise ValueError("target must be nonnull success boolean")
    residual = (fit.sort("traj_id").with_columns(_y=(~pl.col("target")).cast(pl.Float64))
                .with_columns(_r=pl.col("_y") - pl.col("_y").mean().over("instance_id")))
    sums = defaultdict(float)
    for tid, r in residual.select("traj_id", "_r").iter_rows():
        for g in sorted(corpus.documents[tid]):
            sums[g] += r
    return [{"ngram": corpus.names[g], "score": sums[g] / math.sqrt(stats.trajectory_count[g]),
             "n_trajectories": stats.trajectory_count[g], "n_issues": len(stats.issue_sets[g])}
            for g in sorted(stats.trajectory_count)]


def select_vocabulary(scores: list[dict], t: int, vocab_k: int) -> list[dict]:
    if t < 1 or vocab_k < 1:
        raise ValueError("t and vocab_k must be positive")
    candidates = [r for r in scores if r["n_issues"] >= t]
    candidates.sort(key=lambda r: (-abs(r["score"]), r["ngram"]))
    return [{**r, "rank": j, "direction": "failure" if r["score"] > 0 else "success" if r["score"] < 0 else "zero"}
            for j, r in enumerate(candidates[:vocab_k])]


def overlap_rows(vocabularies: dict[int, list[dict]]) -> list[dict]:
    out = []
    for a, b in combinations(sorted(vocabularies), 2):
        x, y = ({r["ngram"] for r in vocabularies[f]} for f in (a, b))
        union, inter = len(x | y), len(x & y)
        out.append({"fold_a": a, "fold_b": b, "intersection": inter, "union": union,
                    "jaccard": inter / union if union else None})
    return out
