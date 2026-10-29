"""pytest 설정 확인용 스모크 테스트. 실제 테스트가 생기면 지운다."""


def test_pythonpath_includes_code():
    import sys
    from pathlib import Path

    code_dir = Path(__file__).resolve().parents[1]
    assert any(Path(p).resolve() == code_dir for p in sys.path)
