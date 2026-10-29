"""궤적 → 스텝 파싱.

스텝 = ai 메시지 1개 + 바로 뒤 user 메시지(환경 관측, 없을 수 있음).
시스템 메시지와 첫 user 메시지(이슈 본문)는 스텝이 아니다.
step_idx는 1부터 센다. "앞 N스텝" = step_idx <= N.
"""
import hashlib
import re

# 정상 역할 순서: system, user(이슈), (ai, user)*, [ai]. S/U/A로 줄여 비교한다.
ROLE_CODE = {"system": "S", "user": "U", "ai": "A"}
NORMAL_ROLES = re.compile(r"SU(AU)*A?")

# SWE-agent ThoughtActionParser와 같은 규칙: 중첩되지 않은 마지막 ``` 블록이 명령이다.
# 언어 표시가 있는 ```python 은 여는 펜스, 맨 ``` 은 열린 블록이 있으면 닫는 펜스다.
FENCE = re.compile(r"^```(\S*)\s*\n|^```(?!\S)", re.MULTILINE)

# 명령 이름은 첫 토큰의 앞쪽 식별자다. 모델이 "find. -name", "cd..", "ls.github"처럼
# 공백을 빼먹은 명령을 자주 내서, 첫 토큰을 그대로 쓰면 같은 명령이 여러 이름으로 갈라진다.
# 원문은 command에 남으므로 잘못된 명령인지는 관측(에러 여부)으로 볼 수 있다.
COMMAND_HEAD = re.compile(r"[A-Za-z_][\w\-]*|\./[^\s]*|\S+")

# 이름 사전. 전체 2,115,623스텝에서 command_head 빈도가 0.1% 이상인 26개 (누적 96.7%).
# 빈도표는 01_parse.py가 results/phase1/command_counts.csv로 저장한다.
# 나머지는 other, 코드 블록이 없으면 none.
COMMANDS = (
    "edit", "python", "open", "search_dir", "search_file", "ls", "goto", "create",
    "find_file", "scroll_down", "submit", "rm", "find", "grep", "cd", "scroll_up",
    "dvc", "echo", "pip", "cat", "mkdir", "pytest", "mv", "curl", "wget", "bash",
)

# 관측 끝에 SWE-agent가 붙이는 상태 줄.
TRAILER = re.compile(r"\(Open file: (?P<file>.*)\)\n\(Current directory: (?P<cwd>.*)\)\n?bash-\$\s*\Z")

EDIT_FAILED = "Your proposed edit has introduced new syntax error"
ERROR = re.compile(
    "|".join([
        r"Traceback \(most recent call last\)",
        r"^\s*[\w.]*(?:Error|Exception)\b:",  # NameError: ..., requests.exceptions.HTTPError: ...
        re.escape(EDIT_FAILED),
        r"command not found",
        r"No such file or directory",
        r"syntax error near unexpected token",  # bash
        r"^(?:File|Directory) .* not found$",  # SWE-agent open/search 명령
        r"^usage:",
    ]),
    re.MULTILINE | re.IGNORECASE,
)


def last_code_block(text: str) -> str | None:
    """마지막 최상위 코드 블록의 내용. 없으면 None."""
    stack, last = [], None
    for m in FENCE.finditer(text):
        if stack and not m.group(1):
            start = stack.pop()
            if not stack:
                last = (start, m)
        elif m.group(1) is not None:
            stack.append(m)
    if last is None:
        return None
    return text[last[0].end():last[1].start()].strip()


def command_head(command: str | None) -> str | None:
    """명령 첫 토큰의 식별자 부분. 'find. -name x' → 'find'."""
    if not command:
        return None
    m = COMMAND_HEAD.match(command.lstrip())
    return m.group(0) if m else None


def command_name(head: str | None) -> str:
    if head is None:
        return "none"
    return head if head in COMMANDS else "other"


def is_error(obs: str | None) -> bool:
    return obs is not None and ERROR.search(obs) is not None


def parse_trailer(obs: str | None) -> tuple[str | None, str | None]:
    """관측 끝 상태 줄의 (열린 파일, 현재 디렉터리). 열린 파일이 n/a면 None."""
    if obs is None:
        return None, None
    m = TRAILER.search(obs)
    if m is None:
        return None, None
    f = m.group("file")
    return (None if f == "n/a" else f), m.group("cwd")


def extract_issue(first_user: str) -> str:
    """첫 user 메시지에서 이슈 본문만 꺼낸다 (B1 입력)."""
    start = first_user.find("ISSUE:\n")
    end = first_user.find("\nINSTRUCTIONS:")
    if start < 0 or end < 0:
        return first_user
    return first_user[start + len("ISSUE:\n"):end].strip()


def parse_trajectory(messages: list[dict]) -> tuple[dict, list[dict]]:
    """메시지 리스트 → (궤적 정보, 스텝 리스트).

    역할 순서가 정상이 아니어도 멈추지 않는다. ai 메시지마다 스텝 하나를 만들고,
    그 뒤 다음 ai 전까지의 user 메시지를 이어 붙여 관측으로 쓴다.
    role_pattern_ok로 비정상 궤적을 표시한다.
    """
    roles = "".join(ROLE_CODE.get(m["role"], "?") for m in messages)
    system = next((m for m in messages if m["role"] == "system"), None)
    system_text = (system.get("system_prompt") or system.get("text") or "") if system else ""
    first_ai = roles.find("A")
    before_ai = messages if first_ai < 0 else messages[:first_ai]
    first_user = next((m["text"] or "" for m in before_ai if m["role"] == "user"), "")

    steps = []
    for i, m in enumerate(messages):
        if m["role"] != "ai":
            continue
        obs_parts = []
        for nxt in messages[i + 1:]:
            if nxt["role"] == "ai":
                break
            if nxt["role"] == "user":
                obs_parts.append(nxt["text"] or "")
        obs = "\n".join(obs_parts) if obs_parts else None
        action = m["text"] or ""
        command = last_code_block(action)
        head = command_head(command)
        open_file, cwd = parse_trailer(obs)
        steps.append({
            "step_idx": len(steps) + 1,
            "action_text": action,
            "command": command,
            "command_head": head,
            "command_name": command_name(head),
            "obs_text": obs,
            "ai_len": len(action),
            "obs_len": 0 if obs is None else len(obs),
            "is_error": is_error(obs),
            "edit_failed": obs is not None and EDIT_FAILED in obs,
            "open_file": open_file,
            "cwd": cwd,
        })

    info = {
        "n_steps": len(steps),
        "issue_text": extract_issue(first_user),
        "issue_msg_len": len(first_user),
        "system_len": len(system_text),
        "system_hash": hashlib.sha1(system_text.encode()).hexdigest()[:12],
        "role_pattern_ok": NORMAL_ROLES.fullmatch(roles) is not None,
        "roles_head": roles[:12],
    }
    return info, steps
