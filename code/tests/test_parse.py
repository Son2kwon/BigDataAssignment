"""afp.parse 단위 테스트. 손으로 만든 작은 궤적으로 경계 사례를 확인한다."""
from afp import parse

TRAILER = "\n(Open file: {f})\n(Current directory: /repo)\nbash-$"
ISSUE = (
    "We're currently solving the following issue within our repository. Here's the issue text:\n"
    "ISSUE:\nCrash on empty input\n\nINSTRUCTIONS:\nNow, you're going to solve this issue on your own."
)


def msg(role, text, **kw):
    return {"role": role, "text": text, "mask": role == "ai", "system_prompt": None, "cutoff_date": None, **kw}


def head(*extra):
    return [msg("system", None, system_prompt="SETTING: ..."), msg("user", ISSUE), *extra]


def test_normal_trajectory_last_step_without_obs():
    traj = head(
        msg("ai", "Let's look.\n```\nfind_file \"a.py\"\n```"),
        msg("user", "Found 1 matches" + TRAILER.format(f="n/a")),
        msg("ai", "Open it.\n```\nopen a.py 10\n```"),
        msg("user", "[File: /repo/a.py]" + TRAILER.format(f="/repo/a.py")),
        msg("ai", "Done.\n```\nsubmit\n```"),
    )
    info, steps = parse.parse_trajectory(traj)

    assert info["n_steps"] == 3
    assert info["role_pattern_ok"]
    assert info["issue_text"] == "Crash on empty input"
    assert info["system_len"] == len("SETTING: ...")
    assert [s["step_idx"] for s in steps] == [1, 2, 3]
    assert [s["command_name"] for s in steps] == ["find_file", "open", "submit"]
    assert steps[1]["open_file"] == "/repo/a.py" and steps[1]["cwd"] == "/repo"
    assert steps[0]["open_file"] is None
    assert steps[-1]["obs_text"] is None and steps[-1]["obs_len"] == 0
    assert steps[-1]["open_file"] is None and not steps[-1]["is_error"]


def test_ai_without_code_block_and_empty_text():
    traj = head(
        msg("ai", "I am thinking without a command."),
        msg("user", "Your output was not formatted correctly."),
        msg("ai", None),
        msg("user", ""),
    )
    info, steps = parse.parse_trajectory(traj)

    assert info["n_steps"] == 2
    for s in steps:
        assert s["command"] is None and s["command_head"] is None and s["command_name"] == "none"
    assert steps[1]["action_text"] == "" and steps[1]["ai_len"] == 0
    assert steps[1]["obs_text"] == "" and steps[1]["obs_len"] == 0


def test_only_issue_has_zero_steps():
    info, steps = parse.parse_trajectory(head())
    assert info["n_steps"] == 0 and steps == []
    assert info["role_pattern_ok"]


def test_abnormal_roles_join_observations():
    traj = head(
        msg("ai", "```\nls\n```"),
        msg("user", "a.py"),
        msg("user", "b.py"),
        msg("ai", "```\nsubmit\n```"),
    )
    info, steps = parse.parse_trajectory(traj)
    assert not info["role_pattern_ok"]
    assert info["n_steps"] == 2
    assert steps[0]["obs_text"] == "a.py\nb.py"


def test_last_code_block_takes_last_and_handles_nested_fences():
    text = "first\n```\nls\n```\nthen\n```\nedit 1:2\n```python\nx = 1\n```\nend_of_edit\n```"
    # SWE-agent 규칙: ```python 은 여는 펜스이므로 edit 블록 안에 중첩된 블록이 된다
    assert parse.last_code_block(text) == "edit 1:2\n```python\nx = 1\n```\nend_of_edit"
    assert parse.last_code_block("no fence here") is None
    assert parse.last_code_block("```\nunclosed") is None


def test_command_head_normalizes_missing_space():
    assert parse.command_head('find. -name "*.py"') == "find"
    assert parse.command_head("cd..") == "cd"
    assert parse.command_head("ls.github") == "ls"
    assert parse.command_head("./reproduce.sh") == "./reproduce.sh"
    assert parse.command_head("edit 141:143\nfoo\nend_of_edit") == "edit"
    assert parse.command_head("") is None
    assert parse.command_name("find") == "find"
    assert parse.command_name("./reproduce.sh") == "other"
    assert parse.command_name(None) == "none"


def test_error_detection():
    errors = [
        "Traceback (most recent call last):\n  File \"x.py\", line 1",
        "NameError: name 'x' is not defined",
        "requests.exceptions.HTTPError: 403 Client Error",
        "Your proposed edit has introduced new syntax error(s). Please read this error message carefully",
        "/bin/bash: line 1: cd..: command not found",
        "ls: cannot access 'x': No such file or directory",
        "/bin/bash: line 1: syntax error near unexpected token `('",
        "Directory src not found",
        "usage: open <file> [<line_number>]",
    ]
    for obs in errors:
        assert parse.is_error(obs), obs
    not_errors = [
        None,
        "",
        "No matches found for \"foo\" in /repo",
        "120:            raise FileNotFoundError(errno.ENOENT)",  # 파일 보기 출력의 코드 줄
        "File updated. Please review the changes.",
    ]
    for obs in not_errors:
        assert not parse.is_error(obs), obs


def test_extract_issue_falls_back_to_full_text():
    assert parse.extract_issue("no template") == "no template"
