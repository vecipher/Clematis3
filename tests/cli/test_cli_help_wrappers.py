import pytest
from clematis.cli.main import build_parser

@pytest.mark.parametrize("sub", ["rotate-logs","inspect-snapshot","bench-t4","seed-lance-demo"])
def test_help_exits_zero(sub):
    p = build_parser()
    with pytest.raises(SystemExit) as e:
        p.parse_args([sub, "--help"])
    assert e.value.code == 0