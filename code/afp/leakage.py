"""사후 누수 점검 전용 (Phase 2.3). 피처·모델 코드에서 import하지 않는다.

이 저장소에서 결과 전용 컬럼(eval_logs 등)을 원본에서 읽는 곳은 이 모듈뿐이다.
읽은 값은 점검 결과(results/phase2/)로만 나가고 data/processed/의 피처 입력에는 들어가지 않는다.

eval_logs 원문 구조 [확인, 전체 80,036행 조회]:
  apply_patch 블록 2개 (prediction_patch.patch → test_patch.patch, 뒤 블록에 `git restore <기존 테스트 파일>`)
  → ___ENVIRONMENT_SET_UP_PATCH_APPLIED___ → run_tests 블록 (`pytest -rA ... <테스트 파일>`)
  → ___TESTS_SECTION___ pytest 출력 ___TESTS_SECTION_END___
테스트 식별자는 `-rA` 요약 줄(`PASSED path::name`)과 verbose 줄(`path::name PASSED`)에서 꺼낸다.
test_patch가 건드린 .py 파일 중 `git restore` 대상이 아닌 파일은 숨김 테스트 패치가 새로 만든 파일로 본다 [가정].
하네스가 되돌리는 파일은 거의 모두 .py이고, .json·.txt 같은 비 .py 파일은 기존 파일이어도 되돌리지 않아서
(에이전트가 search_dir로 본 예가 있음) 새 파일 판정에서 뺀다.
"""
import ast
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split

from afp import io

TESTS_START = "___TESTS_SECTION___"
TESTS_END = "___TESTS_SECTION_END___"

# 채점 하네스에서만 나오는 문자열. 에이전트 관측에 있으면 채점 산출물이 섞였다는 강한 증거다.
HARNESS_MARKERS = (
    TESTS_START,
    TESTS_END,
    "___ENVIRONMENT_SET_UP_PATCH_APPLIED___",
    "test_patch.patch",
    "prediction_patch.patch",
    "Will execute following command for run_tests",
    "Will execute following command for apply_patch",
)
# 하네스 디렉터리 이름. 컨테이너의 `ls /`에 보이는 환경 정보라 그 자체로는 채점 결과 누수가 아니다.
HARNESS_DIR = "swebench_matterhorn"

RUN_TESTS_BLOCK = re.compile(r"Will execute following command for run_tests:\n```\n(.*?)```", re.S)
APPLY_BLOCK = re.compile(r"Will execute following command for apply_patch:\n```\n(.*?)```\n(.*?)(?=Will execute following command|___ENVIRONMENT_SET_UP_PATCH_APPLIED___|\Z)", re.S)
CHECKING_PATCH = re.compile(r"^Checking patch (\S+?)\.\.\.$", re.M)
GIT_RESTORE = re.compile(r"git restore (\S+)")
# 값을 받는 pytest 옵션 (전체 데이터에서 나온 것: -p, -W)
PYTEST_VALUE_OPTS = {"-p", "-W", "-k", "-m", "-c", "-o", "--rootdir"}

OUTCOMES = ("PASSED", "FAILED", "ERROR", "XFAIL", "XPASS", "SKIPPED")
# path::Class::name[param]. param에는 공백이 들어갈 수 있다 (예: test_x[str_check-  s  -s0])
NODE = r"[^\s:]+(?:::[^\s\[:]+)+(?:\[[^\n]*?\])?"
SUMMARY_LINE = re.compile(rf"^(?P<outcome>PASSED|FAILED|ERROR|XFAIL|XPASS) (?P<node>{NODE})(?: - .*)?$", re.M)
VERBOSE_LINE = re.compile(rf"^(?P<node>{NODE}) (?P<outcome>{'|'.join(OUTCOMES)})\b", re.M)

# 에이전트 명령이 테스트 실행인지 (운영 중에도 보이는 정상 정보)
TEST_RUN = re.compile(
    r"\b(?:pytest|py\.test|unittest|nosetests|nose2|tox|runtests(?:\.py)?)\b"
    r"|\bmanage\.py\s+test\b|\bsetup\.py\s+test\b|\bpython[\d.]*\s+(?:-m\s+)?\S*test\S*"
)
# 저장소 내용을 보여 주는 명령 (기존 테스트 코드가 보이는 정상 경로)
REPO_VIEW = {
    "open", "goto", "scroll_down", "scroll_up", "search_dir", "search_file", "find_file",
    "grep", "cat", "find", "ls", "edit", "create", "head", "tail", "sed", "less", "more", "nl", "awk",
}
CATEGORIES = ("harness_artifact", "issue_text", "agent_typed", "agent_test_run", "repo_view", "other")


@dataclass
class EvalLog:
    status: str  # missing | no_tests_section | truncated | ok
    test_files: list[str] = field(default_factory=list)  # run_tests 명령의 대상 경로
    test_patch_files: list[str] = field(default_factory=list)
    new_test_files: list[str] = field(default_factory=list)  # 숨김 테스트 패치가 새로 만든 파일
    tests: dict[str, str] = field(default_factory=dict)  # node id → outcome


def test_name(node: str) -> str:
    """'tests/a.py::TestX::test_y[p1]' → 'test_y'."""
    return node.split("::")[-1].split("[", 1)[0]


def node_base(node: str) -> str:
    """파라미터를 뺀 node id. 'tests/a.py::test_y[p1]' → 'tests/a.py::test_y'."""
    return node.split("[", 1)[0]


def _pytest_targets(command: str) -> list[str]:
    lines = [ln for ln in command.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    try:
        toks = shlex.split(lines[-1])
    except ValueError:
        toks = lines[-1].split()
    if not toks or "pytest" not in toks[0]:
        return []
    out, skip = [], False
    for tok in toks[1:]:
        if skip:
            skip = False
        elif tok in PYTEST_VALUE_OPTS:
            skip = True
        elif not tok.startswith("-"):
            out.append(tok)
    return out


def parse_eval_log(text: str | None) -> EvalLog:
    """eval_logs 원문 → 테스트 식별자. 형식이 깨진 로그는 status로 구분한다."""
    if not text:
        return EvalLog("missing")
    log = EvalLog("ok")
    for cmd, out in APPLY_BLOCK.findall(text):
        if "test_patch.patch" in cmd:
            restored = set(GIT_RESTORE.findall(cmd))
            files = CHECKING_PATCH.findall(out)
            log.test_patch_files = sorted(set(files))
            log.new_test_files = sorted(f for f in set(files) - restored if f.endswith(".py"))
    runs = RUN_TESTS_BLOCK.findall(text)
    if runs:
        log.test_files = _pytest_targets(runs[-1])
    start = text.find(TESTS_START)
    if start < 0:
        log.status = "no_tests_section"
        return log
    end = text.find(TESTS_END, start)
    if end < 0:
        log.status = "truncated"
    section = text[start:] if end < 0 else text[start:end]
    for m in VERBOSE_LINE.finditer(section):
        log.tests.setdefault(m["node"], m["outcome"])
    for m in SUMMARY_LINE.finditer(section):  # 요약 줄이 최종 결과라 덮어쓴다
        log.tests[m["node"]] = m["outcome"]
    return log


def _word_regex(words) -> re.Pattern | None:
    words = sorted({w for w in words if w}, key=len, reverse=True)
    if not words:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, words)) + r")(?!\w)")


def _result_line(name: str) -> re.Pattern:
    """name에 대한 pytest 결과 줄 (요약형 또는 verbose형)."""
    n = re.escape(name)
    return re.compile(
        rf"(?:{'|'.join(OUTCOMES)})\s+\S*::{n}(?:\[|\s|$)|::{n}(?:\[[^\n]*?\])?\s+(?:{'|'.join(OUTCOMES)})\b",
        re.M,
    )


def classify(step: dict, earlier_actions: str, issue_text: str, word: re.Pattern) -> str:
    """식별자가 처음 보인 스텝의 맥락. 앞에 있는 범주일수록 우선한다."""
    obs = step["obs_text"] or ""
    if any(m in obs for m in HARNESS_MARKERS):
        return "harness_artifact"
    if word.search(issue_text or ""):
        return "issue_text"
    if word.search(earlier_actions):
        return "agent_typed"
    command = step["command"] or ""
    if TEST_RUN.search(command):
        return "agent_test_run"
    if step["command_head"] in REPO_VIEW:
        return "repo_view"
    return "other"


def generic_names(logs: dict[str, EvalLog], repo_of: dict[str, str], min_repos: int = 3) -> set[str]:
    """여러 저장소(min_repos개 이상)의 채점 로그에 나오는 테스트 이름 ('test', 'test_repr' 등).

    이런 이름은 우연히 겹치기 쉬워서 겹침 수를 따로 센다. logs·repo_of의 키는 traj_id.
    """
    repos: dict[str, set[str]] = {}
    for tid, log in logs.items():
        for name in {test_name(n) for n in log.tests}:
            repos.setdefault(name, set()).add(repo_of[tid])
    return {n for n, r in repos.items() if len(r) >= min_repos or len(n) < 5}


def audit_trajectory(log: EvalLog, issue_text: str, steps: list[dict], generic: frozenset | set = frozenset()) -> tuple[dict, list[dict]]:
    """eval_logs 식별자와 앞 스텝 관측의 겹침.

    steps: step_idx 순으로 정렬된 앞 N스텝 (step_idx, action_text, command, command_head, obs_text).
    generic: 우연히 겹치기 쉬운 테스트 이름 (generic_names). 겹침 행에 표시만 하고 빼지는 않는다.
    돌려주는 것: (궤적 요약, 겹친 식별자별 행)
    """
    names = sorted({test_name(n) for n in log.tests})
    nodes = sorted({node_base(n) for n in log.tests})
    failed_names = {test_name(n) for n, o in log.tests.items() if o in ("FAILED", "ERROR")}
    new_files = set(log.new_test_files)
    new_names = {test_name(n) for n in log.tests if n.split("::")[0] in new_files}
    files = sorted(set(log.test_files) | set(log.test_patch_files))

    obs = [(s, s["obs_text"] or "") for s in steps]
    name_re = _word_regex(names)
    first_step = {}
    if name_re is not None:
        for s, text in obs:
            for m in name_re.finditer(text):
                first_step.setdefault(m.group(0), s)

    rows = []
    for name, s in sorted(first_step.items(), key=lambda kv: (kv[1]["step_idx"], kv[0])):
        word = _word_regex([name])
        earlier = "\n".join(t["action_text"] or "" for t in steps if t["step_idx"] <= s["step_idx"])
        result_step = next((t for t, text in obs if _result_line(name).search(text)), None)
        rows.append({
            "kind": "test_name",
            "identifier": name,
            "first_step": s["step_idx"],
            "category": classify(s, earlier, issue_text, word),
            "command": (s["command"] or "")[:200],
            "generic_name": name in generic,
            "failed_in_eval": name in failed_names,
            "in_new_test_file": name in new_names,
            "result_line_step": None if result_step is None else result_step["step_idx"],
            "result_line_command_is_test_run": None if result_step is None else bool(TEST_RUN.search(result_step["command"] or "")),
        })
    for path in sorted(new_files):
        s = next((t for t, text in obs if path in text), None)
        if s is None:
            continue
        word = re.compile(re.escape(path))
        earlier = "\n".join(t["action_text"] or "" for t in steps if t["step_idx"] <= s["step_idx"])
        rows.append({
            "kind": "new_test_file",
            "identifier": path,
            "first_step": s["step_idx"],
            "category": classify(s, earlier, issue_text, word),
            "command": (s["command"] or "")[:200],
            "generic_name": False,
            "failed_in_eval": None,
            "in_new_test_file": True,
            "result_line_step": None,
            "result_line_command_is_test_run": None,
        })

    node_hits = sum(any(n in text for _, text in obs) for n in nodes)
    file_hits = sum(any(f in text for _, text in obs) for f in files)
    name_rows = [r for r in rows if r["kind"] == "test_name"]
    summary = {
        "eval_status": log.status,
        "n_tests": len(log.tests),
        "n_test_names": len(names),
        "n_test_files": len(files),
        "n_new_test_files": len(new_files),
        "n_new_test_names": len(new_names),
        "name_hits": len(name_rows),
        "specific_name_hits": sum(not r["generic_name"] for r in name_rows),
        "specific_name_hits_unexplained": sum(not r["generic_name"] and r["category"] in ("harness_artifact", "other") for r in name_rows),
        "node_hits": node_hits,
        "file_hits": file_hits,
        "failed_name_hits": sum(r["failed_in_eval"] for r in name_rows),
        "new_name_hits": sum(r["in_new_test_file"] for r in name_rows),
        "new_file_hits": sum(r["kind"] == "new_test_file" for r in rows),
        "result_line_hits": sum(r["result_line_step"] is not None for r in name_rows),
        "result_line_without_test_run": sum(r["result_line_command_is_test_run"] is False for r in name_rows),
        "harness_marker_steps": sum(any(m in text for m in HARNESS_MARKERS) for _, text in obs),
        "harness_dir_steps": sum(HARNESS_DIR in text for _, text in obs),
        **{f"cat_{c}": sum(r["category"] == c for r in rows) for c in CATEGORIES},
    }
    return summary, rows


# ---------------------------------------------------------------------------
# 표본, 원본 로드, 대조군

# 표본·로드에 쓰는 궤적 컬럼 (피처 입력이 아니라 점검용 메타데이터)
SAMPLE_COLUMNS = ["traj_id", "file_idx", "row_idx", "instance_id", "model_name", "target", "issue_text"]
STEP_COLUMNS = ["traj_id", "step_idx", "action_text", "command", "command_head", "obs_text"]


def sample_trajectories(traj: pl.DataFrame, n: int, seed: int = io.SEED) -> pl.DataFrame:
    """(model_name, target) 층화 무작위 표본. 같은 입력·n·seed면 항상 같은 표본이다."""
    traj = traj.sort("traj_id")
    if n >= traj.height:
        return traj
    strata = (traj["model_name"] + "|" + traj["target"].cast(pl.String)).to_numpy()
    idx, _ = train_test_split(range(traj.height), train_size=n, stratify=strata, random_state=seed)
    return traj[sorted(idx)]


def load_eval_logs(sample: pl.DataFrame) -> dict[str, str | None]:
    """표본 궤적의 eval_logs 원문. 원본 행의 instance_id·model_name이 궤적 표와 다르면 멈춘다."""
    out = {}
    for k, part in sample.group_by("file_idx", maintain_order=True):
        k = k[0]
        raw = io.read_raw(k, ["instance_id", "model_name", "eval_logs"], allow_leaky=True)
        rows = part["row_idx"].to_list()
        got = raw.take(rows).to_pylist()
        for r, meta, tid, iid, model in zip(rows, got, part["traj_id"], part["instance_id"], part["model_name"]):
            if (meta["instance_id"], meta["model_name"]) != (iid, model):
                raise ValueError(f"{tid}: 원본 {k}번 파일 {r}행과 궤적 표가 다르다")
            out[tid] = meta["eval_logs"]
    return out


def control_donors(meta: pl.DataFrame, same_repo: bool) -> dict[str, str]:
    """대조군: 각 궤적에 다른 이슈의 eval_logs 식별자를 붙여 우연히 겹치는 비율을 잰다.

    same_repo=True면 같은 저장소의 다른 이슈, False면 다른 저장소에서 고른다 (결정적).
    meta: 식별자가 있는 표본 궤적의 traj_id, instance_id.
    """
    df = meta.select("traj_id", "instance_id").with_columns(repo=pl.col("instance_id").str.replace(r"-\d+$", ""))
    df = df.sort("repo", "instance_id", "traj_id")
    tids, iids, repos = df["traj_id"].to_list(), df["instance_id"].to_list(), df["repo"].to_list()
    n, out = len(tids), {}
    for i in range(n):
        for step in range(1, n):
            j = (i + (step if same_repo else n // 2 + step - 1)) % n
            if iids[j] == iids[i] or (repos[j] == repos[i]) != same_repo:
                continue
            out[tids[i]] = tids[j]
            break
    return out


# ---------------------------------------------------------------------------
# (a) 정적 점검: 누수 컬럼 이름을 코드에서 쓰는 곳

LEAKY_WORD = re.compile(r"\b(?:" + "|".join(io.LEAKY_COLUMNS) + r")\b")
# 누수 컬럼을 이름으로 다뤄도 되는 파일 (code/ 기준 상대 경로)
#   afp/io.py      LEAKY_COLUMNS 정의와 차단 함수
#   afp/leakage.py 이 사후 점검 모듈
#   01_parse.py    exit_status를 분석 전용으로 trajectories.parquet에 보관 (Phase 1 스키마)
LEAKY_ALLOWLIST = ("afp/io.py", "afp/leakage.py", "01_parse.py")


def leaky_references(path: Path) -> list[tuple[int, str]]:
    """파이썬 파일에서 누수 컬럼 이름이 나오는 (줄, 이름). 주석·docstring은 무시한다."""
    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            hits += [(node.lineno, m) for m in LEAKY_WORD.findall(node.value)]
        elif isinstance(node, ast.Name) and LEAKY_WORD.fullmatch(node.id):
            hits.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and LEAKY_WORD.fullmatch(node.attr):
            hits.append((node.lineno, node.attr))
        elif isinstance(node, ast.keyword) and node.arg and LEAKY_WORD.fullmatch(node.arg):
            hits.append((node.value.lineno, node.arg))
    return sorted(set(hits))


def scan_code(code_dir: Path = io.ROOT / "code") -> dict[str, list[tuple[int, str]]]:
    """code/ 아래 (tests 제외) 파이썬 파일 중 누수 컬럼 이름을 쓰는 파일."""
    out = {}
    for path in sorted(code_dir.rglob("*.py")):
        rel = path.relative_to(code_dir).as_posix()
        if rel.startswith("tests/"):
            continue
        hits = leaky_references(path)
        if hits:
            out[rel] = hits
    return out
