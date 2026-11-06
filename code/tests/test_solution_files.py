"""5.5 사후 분석 (정답 파일 열람) 함수. afp/leakage.py 전용 경로."""
import polars as pl

from afp import leakage

PATCH = """diff --git a/pkg/a.py b/pkg/a.py
index 1..2 100644
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1 +1 @@
-x
+y
diff --git a/reproduce.py b/reproduce.py
new file mode 100644
index 0000000..1
diff --git a/tests/test_a.py b/tests/test_a.py
index 1..2 100644
diff --git a/pkg/old.py b/pkg/old.py
deleted file mode 100644
diff --git a/pkg/conftest.py b/pkg/conftest.py
index 1..2 100644
"""


def test_patch_files_keeps_existing_non_test_files():
    assert leakage.patch_files(PATCH) == ["pkg/a.py"]
    assert leakage.patch_files(None) == [] and leakage.patch_files("") == []


def test_hint_files():
    text = "bug\n\n\nHere is the list of files that is recommended to be edited in order to fix the issue.\nFILES TO MODIFY:\n- a/b.py\n- c.py\n"
    assert leakage.hint_files(text) == ["a/b.py", "c.py"]
    assert leakage.hint_files("no hint - x.py") == []


def test_solution_proxy_is_leave_one_out():
    trajs = pl.DataFrame({"traj_id": ["s1", "s2", "f1", "g1"], "instance_id": ["i", "i", "i", "j"], "target": [True, True, False, False]})
    files = pl.DataFrame({"traj_id": ["s1", "s2", "f1", "g1"], "files": [["a.py"], ["b.py"], ["c.py"], ["d.py"]]})
    p = {r["traj_id"]: r["proxy"] for r in leakage.solution_proxy(trajs, files).iter_rows(named=True)}
    assert p == {"s1": ["b.py"], "s2": ["a.py"], "f1": ["a.py", "b.py"]}  # 자기 패치는 빼고, 실패 궤적 패치는 안 쓴다. j는 성공 없음


def test_opened_files_first_step_and_suffix_match():
    steps = pl.DataFrame({"traj_id": ["t", "t", "t", "u"], "step_idx": [1, 2, 3, 1],
                          "open_file": ["/repo/reproduce.py", "/repo/xpkg/a.py", "/repo/pkg/a.py", None]})
    targets = pl.DataFrame({"traj_id": ["t", "u"], "proxy": [["pkg/a.py"], ["pkg/a.py"]]})
    got = leakage.opened_files(steps, targets, "proxy")
    assert got.to_dicts() == [{"traj_id": "t", "first_step": 3}]  # "/repo/xpkg/a.py"는 '/pkg/a.py'로 끝나지 않는다
