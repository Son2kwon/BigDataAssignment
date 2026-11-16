"""Phase 5d 순수 추출·길이 손 계산/누수 테스트. 실제 학습·GPU를 사용하지 않는다."""
import polars as pl
import pytest

from afp import features


def example(texts=("ab", "", "c", "def", "ghij")):
    steps = pl.DataFrame({"traj_id": ["a"] * len(texts), "step_idx": list(range(1, len(texts) + 1)), "action_text": list(texts)})
    trajs = pl.DataFrame({"traj_id": ["a"], "n_steps": [20]})
    return steps, trajs


@pytest.mark.parametrize("text,expected", [
    (None, ""), ("  \n  ", ""), (" Plain TEXT ", "Plain TEXT"),
    ("before```python\nsecret\n```after", "before after"),
    ("a```x```b```y```c", "a b c"), ("a```unclosed secret", "a"),
    ("```x```after```unfinished", "after"), ("```only code", ""),
    ("a  b", "a  b"), ("a``````b", "a b"),
])
def test_extract_thought_all_fences(text, expected):
    assert features.extract_thought(text) == expected


def test_lengths_by_hand_and_no_window_at_n_equals_k():
    steps, trajs = example()
    out = features.thought_length_features(steps, trajs, 5)
    r = out.row(0, named=True)
    assert r["thought_len_mean"] == 2
    assert r["thought_len_max"] == 4
    assert r["thought_len_last"] == 4
    assert r["n_empty_thought"] == 1
    assert r["thought_len_mean_w3"] == pytest.approx(8 / 3)
    assert "thought_len_mean_w3" not in features.thought_length_features(steps, trajs, 3).columns
    features.check_features(out.with_columns(N=pl.lit(5)), trajs, [5], features.thought_length_columns(5))


def test_future_action_text_mutations_do_not_change_lengths():
    steps, trajs = example()
    expected = features.thought_length_features(steps, trajs, 3)
    variants = [steps.filter(pl.col("step_idx") <= 3), steps.reverse(),
                steps.with_columns(action_text=pl.when(pl.col("step_idx") > 3).then(pl.lit("future```secret")).otherwise(pl.col("action_text"))),
                pl.concat([steps, pl.DataFrame({"traj_id": ["a"], "step_idx": [19], "action_text": ["changed future"]})])]
    for variant in variants:
        assert features.thought_length_features(variant, trajs, 3).equals(expected)
    assert features.thought_length_features(steps, trajs.with_columns(n_steps=pl.lit(999)), 3).equals(expected)


def test_window_boundary_by_hand():
    for pos, expected in ((2, 0), (3, 1), (5, 1)):
        steps, trajs = example(tuple("abc" if i == pos else "" for i in range(1, 6)))
        assert features.thought_length_features(steps, trajs, 5)["thought_len_mean_w3"][0] == expected


def test_null_is_empty_but_missing_and_duplicate_steps_are_errors():
    steps, trajs = example((None, "", "```x```"))
    assert features.thought_length_features(steps, trajs, 3)["n_empty_thought"][0] == 3
    for bad in (steps.head(2), pl.concat([steps, steps.head(1)]), steps.with_columns(step_idx=pl.col("step_idx") + 1)):
        with pytest.raises(ValueError, match="prefix steps"):
            features.thought_length_features(bad, trajs, 3)


def test_ended_trajectory_is_excluded():
    steps, trajs = example()
    other = steps.with_columns(traj_id=pl.lit("ended"))
    both = pl.concat([trajs, pl.DataFrame({"traj_id": ["ended"], "n_steps": [3]})])
    assert features.thought_length_features(pl.concat([steps, other]), both, 3)["traj_id"].to_list() == ["a"]
