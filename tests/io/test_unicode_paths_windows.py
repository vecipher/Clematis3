# tests/io/test_unicode_paths_windows.py
import os
import sys
import subprocess
import pytest

is_win = sys.platform.startswith("win")

@pytest.mark.skipif(not is_win, reason="Windows-specific path handling")
def test_unicode_logs_dir_windows(tmp_path):
    # e.g. Japanese/Greek/checkmark chars
    uni = tmp_path / "テスト✓Ω"
    uni.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "CI": "true",
        "CLEMATIS_NETWORK_BAN": "1",
        "CLEMATIS_T3__ALLOW_REFLECTION": "false",
        "CLEMATIS_SCHEDULER__ENABLED": "false",
        "CLEMATIS_PERF__PARALLEL__ENABLED": "false",
        "CLEMATIS_T2__QUALITY__ENABLED": "false",
        "CLEMATIS_T2__QUALITY__SHADOW": "false",
        "CLEMATIS_T2__HYBRID__ENABLED": "false",
        "CLEMATIS_GRAPH__ENABLED": "false",
        # Direct logs into the Unicode dir
        "CLEMATIS_LOG_DIR": str(uni),
    })

    subprocess.run(
        [sys.executable, "-m", "clematis.scripts.demo", "--steps", "2", "--text", "identity", "--fixed-now-ms", "0"],
        check=True, env=env
    )

    for name in ("t1.jsonl","t2.jsonl","t3_plan.jsonl","t3_dialogue.jsonl","t4.jsonl","apply.jsonl","health.jsonl"):
        p = uni / name
        assert p.exists(), f"Missing expected log file: {p}"
        # Ensure we can read with utf-8 explicitly
        _ = p.read_text(encoding="utf-8")
