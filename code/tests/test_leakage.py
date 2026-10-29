"""afp.leakage 단위 테스트. 실제 eval_logs 형식을 본뜬 로그와 손으로 만든 스텝으로 확인한다."""
import textwrap

import polars as pl
import pytest

from afp import io, leakage

FENCE = "```"
LOG = textwrap.dedent(f"""\
    Will execute following command for apply_patch:
    {FENCE}

    cd //proj &&
    source activate base &&
    conda activate proj &&
    git apply -v /swebench_matterhorn/prediction_patch.patch --allow-empty
    {FENCE}

    Checking patch proj/core.py...
    Applied patch proj/core.py cleanly.
    Will execute following command for apply_patch:
    {FENCE}

    cd //proj &&
    source activate base &&
    conda activate proj &&
    git restore tests/test_core.py &&
    git apply -v /swebench_matterhorn/test_patch.patch --allow-empty
    {FENCE}

    Checking patch tests/test_core.py...
    Checking patch tests/test_new_feature.py...
    Checking patch tests/data/fixture.json...
    Applied patch tests/test_core.py cleanly.
    Applied patch tests/test_new_feature.py cleanly.
    Applied patch tests/data/fixture.json cleanly.
    ___ENVIRONMENT_SET_UP_PATCH_APPLIED___
    2024-09-26 02:11:42,079 - root - INFO - Will run the following test command
    Will execute following command for run_tests:
    {FENCE}

    cd //proj &&
    source activate base &&
    conda activate proj &&
    pytest --no-header -rA --tb=no  -p no:cacheprovider -W ignore::DeprecationWarning tests/test_core.py tests/test_new_feature.py
    {FENCE}

    ___TESTS_SECTION___
    ============================= test session starts ==============================
    collected 6 items

    tests/test_core.py::TestCore::test_alpha PASSED                          [ 16%]
    tests/test_core.py .F.                                                   [ 50%]

    =========================== short test summary info ============================
    PASSED tests/test_core.py::TestCore::test_alpha
    PASSED tests/test_core.py::test_beta[str_check-  s  -s0]
    FAILED tests/test_core.py::test_gamma - AssertionError: assert [1] == [2]
    ERROR tests/test_new_feature.py::test_brand_new_behaviour - ImportError
    XFAIL tests/test_core.py::test_delta - reason
    SKIPPED [1] tests/test_core.py:12: needs network
    ERROR tests/test_core.py
    ============================== 1 failed, 3 passed in 0.54s =====================
    ___TESTS_SECTION_END___
    Some tests failed.
    """)
TRAILER = "\n(Open file: n/a)\n(Current directory: /proj)\nbash-$"


def test_parse_eval_log_extracts_identifiers():
    log = leakage.parse_eval_log(LOG)
    assert log.status == "ok"
    assert log.test_files == ["tests/test_core.py", "tests/test_new_feature.py"]  # -p, -W 값은 빠진다
    assert log.test_patch_files == ["tests/data/fixture.json", "tests/test_core.py", "tests/test_new_feature.py"]
    assert log.new_test_files == ["tests/test_new_feature.py"]  # 되돌리지 않은 .py만 새 파일로 본다
    assert log.tests == {
        "tests/test_core.py::TestCore::test_alpha": "PASSED",
        "tests/test_core.py::test_beta[str_check-  s  -s0]": "PASSED",
        "tests/test_core.py::test_gamma": "FAILED",
        "tests/test_new_feature.py::test_brand_new_behaviour": "ERROR",
        "tests/test_core.py::test_delta": "XFAIL",
    }
    assert leakage.test_name("tests/test_core.py::test_beta[str_check-  s  -s0]") == "test_beta"
    assert leakage.node_base("tests/test_core.py::test_beta[p]") == "tests/test_core.py::test_beta"


def test_parse_eval_log_broken_formats():
    assert leakage.parse_eval_log(None).status == "missing"
    assert leakage.parse_eval_log("").status == "missing"
    no_tests = LOG.split("___ENVIRONMENT_SET_UP_PATCH_APPLIED___")[0] + "error: patch failed\n"
    log = leakage.parse_eval_log(no_tests)
    assert log.status == "no_tests_section" and log.tests == {}
    truncated = leakage.parse_eval_log(LOG.split("XFAIL")[0])
    assert truncated.status == "truncated"
    assert "tests/test_core.py::test_gamma" in truncated.tests


def step(i, command, obs, action=None, head=None):
    return {
        "step_idx": i,
        "action_text": action if action is not None else f"Do it.\n```\n{command}\n```",
        "command": command,
        "command_head": head or command.split()[0],
        "obs_text": obs,
    }


def test_audit_trajectory_classifies_contexts():
    log = leakage.parse_eval_log(LOG)
    steps = [
        step(1, "open tests/test_core.py", "[File: /proj/tests/test_core.py]\n10:    def test_alpha(self):\n11: test_alpha_extra" + TRAILER),
        step(2, "pytest tests/test_core.py", "PASSED tests/test_core.py::test_beta[x]\n1 passed" + TRAILER),
        step(3, "python reproduce.py", "___TESTS_SECTION___\nFAILED tests/test_core.py::test_gamma" + TRAILER),
        step(4, "search_dir test_delta", "Found test_delta in docs" + TRAILER),
        step(5, "ls", "tests/test_new_feature.py" + TRAILER),
        step(6, "python reproduce.py", None),
    ]
    summary, rows = leakage.audit_trajectory(log, "The issue mentions test_delta.", steps)
    by_name = {r["identifier"]: r for r in rows}
    assert by_name["test_alpha"]["category"] == "repo_view"
    assert by_name["test_beta"]["category"] == "agent_test_run"
    assert by_name["test_beta"]["result_line_command_is_test_run"] is True
    assert by_name["test_gamma"]["category"] == "harness_artifact"
    assert by_name["test_gamma"]["result_line_command_is_test_run"] is False and by_name["test_gamma"]["failed_in_eval"]
    assert by_name["test_delta"]["category"] == "issue_text"
    assert by_name["tests/test_new_feature.py"]["kind"] == "new_test_file"
    assert "test_brand_new_behaviour" not in by_name
    assert summary["eval_status"] == "ok" and summary["n_test_names"] == 5
    assert summary["name_hits"] == 4 and summary["node_hits"] == 2  # test_beta (파라미터 제외), test_gamma
    assert summary["result_line_hits"] == 2 and summary["result_line_without_test_run"] == 1
    assert summary["harness_marker_steps"] == 1 and summary["new_file_hits"] == 1
    assert summary["specific_name_hits_unexplained"] == 1  # harness_artifact 하나
    assert sum(summary[f"cat_{c}"] for c in leakage.CATEGORIES) == len(rows)


def test_audit_word_boundary_agent_typed_and_generic():
    log = leakage.parse_eval_log(LOG)
    steps = [
        step(1, "open a.py", "x = my_test_alpha_helper + test_alphabet" + TRAILER),  # 단어 경계 밖
        step(2, "create test_gamma.py", "[File: /proj/test_gamma.py (1 lines total)]" + TRAILER),
        step(3, "goto 5", "def test_beta(): pass" + TRAILER, action="Look.\n```\ngoto 5\n```"),
    ]
    summary, rows = leakage.audit_trajectory(log, "", steps, generic={"test_beta"})
    by_name = {r["identifier"]: r for r in rows}
    assert set(by_name) == {"test_gamma", "test_beta"}
    assert by_name["test_gamma"]["category"] == "agent_typed"
    assert by_name["test_beta"]["category"] == "repo_view" and by_name["test_beta"]["generic_name"]
    assert summary["specific_name_hits"] == 1


def test_audit_empty_log_and_no_steps():
    summary, rows = leakage.audit_trajectory(leakage.parse_eval_log(None), "issue", [step(1, "ls", "a.py")])
    assert rows == [] and summary["n_test_names"] == 0 and summary["name_hits"] == 0
    summary, rows = leakage.audit_trajectory(leakage.parse_eval_log(LOG), "issue", [])
    assert rows == [] and summary["n_test_names"] == 5


def test_generic_names():
    log = leakage.parse_eval_log(LOG)
    logs = {"a": log, "b": log, "c": log, "d": leakage.EvalLog("ok", tests={"x.py::test_unique_name": "PASSED"})}
    generic = leakage.generic_names(logs, {"a": "r1", "b": "r2", "c": "r3", "d": "r1"})
    assert "test_alpha" in generic and "test_unique_name" not in generic


def test_control_donors_pick_other_issues():
    meta = pl.DataFrame({
        "traj_id": ["t1", "t2", "t3", "t4", "t5"],
        "instance_id": ["a__x-1", "a__x-1", "a__x-2", "b__y-1", "c__z-1"],
    })
    iid = dict(zip(meta["traj_id"], meta["instance_id"]))
    same = leakage.control_donors(meta, same_repo=True)
    assert set(same) == {"t1", "t2", "t3"}  # b, c 저장소는 다른 이슈가 없다
    for t, d in same.items():
        assert iid[t] != iid[d] and iid[t].split("-")[0] == iid[d].split("-")[0]
    other = leakage.control_donors(meta, same_repo=False)
    assert set(other) == set(iid)
    for t, d in other.items():
        assert iid[t].split("-")[0] != iid[d].split("-")[0]


def test_sample_trajectories_deterministic_and_stratified():
    traj = pl.DataFrame({
        "traj_id": [f"00-{i:05d}" for i in range(1000)],
        "model_name": ["m70"] * 900 + ["m8"] * 100,
        "target": [i % 5 == 0 for i in range(1000)],
    })
    a = leakage.sample_trajectories(traj, 200)
    assert a.height == 200 and a["traj_id"].is_sorted()
    assert leakage.sample_trajectories(traj.reverse(), 200).equals(a)
    assert a.filter(pl.col("model_name") == "m8").height == 20
    assert a["target"].mean() == pytest.approx(0.2, abs=0.01)
    assert leakage.sample_trajectories(traj, 5000).height == 1000


def test_leaky_references_static_scan(tmp_path):
    src = tmp_path / "feature.py"
    src.write_text(textwrap.dedent('''\
        """docstring mentioning eval_logs is fine."""
        # comment mentioning generated_patch is fine
        import polars as pl
        df = pl.read_parquet("x.parquet", columns=["traj_id", "exit_status"])
        y = df.generated_patch
        z = dict(eval_logs=1)
        ok = "exit_status_like"
    '''))
    assert leakage.leaky_references(src) == [(4, "exit_status"), (5, "generated_patch"), (6, "eval_logs")]


def test_repository_code_uses_leaky_columns_only_in_allowlist():
    """(a) 강제: 피처·분할 코드에서 누수 컬럼 이름을 쓰면 실패한다."""
    refs = leakage.scan_code()
    assert set(refs) <= set(leakage.LEAKY_ALLOWLIST), {f: refs[f] for f in set(refs) - set(leakage.LEAKY_ALLOWLIST)}
    assert "afp/leakage.py" in refs  # 스캐너가 실제로 찾는지 (빈 결과로 통과하지 않게)


def test_io_guards_reject_leaky_columns():
    for col in io.LEAKY_COLUMNS:
        with pytest.raises(ValueError, match="누수"):
            io.check_safe_columns(["traj_id", col])
        with pytest.raises(ValueError, match="누수"):
            io.load_trajectories(["traj_id", col])
        with pytest.raises(ValueError, match="누수"):
            io.scan_steps(["traj_id", col])
        with pytest.raises(ValueError, match="누수"):
            io.read_raw(0, ["instance_id", col])
    with pytest.raises(ValueError, match="명시"):
        io.check_safe_columns(None)
    with pytest.raises(ValueError, match="명시"):
        io.check_safe_columns("traj_id")
    assert io.check_safe_columns(("traj_id", "target")) == ["traj_id", "target"]


def test_expected_raw_checksums_parse_download_script():
    sums = io.expected_raw_checksums()
    assert sorted(sums) == [f"train-{i:05d}-of-00012.parquet" for i in range(io.N_RAW_FILES)]
    assert all(size > 0 and len(sha) == 64 for size, sha in sums.values())
