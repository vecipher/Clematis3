

from __future__ import annotations

from pathlib import Path
from typing import Dict

from clematis.cli._config import discover_config_path, maybe_log_selected


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# yaml\n", encoding="utf-8")
    return p


def test_explicit_wins_even_if_others_exist(tmp_path):
    # Prepare cwd and XDG files, but explicit path should win
    cwd_cfg = _touch(tmp_path / "configs" / "config.yaml")
    xdg = tmp_path / "xdg"
    xdg_cfg = _touch(xdg / "clematis" / "config.yaml")

    other_env: Dict[str, str] = {"XDG_CONFIG_HOME": str(xdg)}
    p, src = discover_config_path(str(cwd_cfg), tmp_path, other_env)
    assert p == cwd_cfg
    assert src == "explicit"


def test_explicit_missing_is_reported(tmp_path):
    missing = tmp_path / "nope.yaml"
    p, src = discover_config_path(str(missing), tmp_path, {})
    # path is echoed back (expanded), with explicit-missing tag
    assert Path(p) == missing
    assert src == "explicit-missing"


def test_env_clematis_config_points_to_file(tmp_path):
    cfg = _touch(tmp_path / "e.yaml")
    env = {"CLEMATIS_CONFIG": str(cfg)}
    p, src = discover_config_path(None, tmp_path, env)
    assert p == cfg
    assert src == "env:CLEMATIS_CONFIG"


def test_env_clematis_config_points_to_dir(tmp_path):
    d = tmp_path / "dir"
    cfg = _touch(d / "config.yaml")
    env = {"CLEMATIS_CONFIG": str(d)}
    p, src = discover_config_path(None, tmp_path, env)
    assert p == cfg
    assert src == "env:CLEMATIS_CONFIG"


def test_cwd_configs_config_yaml(tmp_path):
    cfg = _touch(tmp_path / "configs" / "config.yaml")
    p, src = discover_config_path(None, tmp_path, {})
    assert p == cfg
    assert src == "cwd:configs/config.yaml"


def test_xdg_lookup(tmp_path):
    xdg = tmp_path / "xdg"
    cfg = _touch(xdg / "clematis" / "config.yaml")
    env = {"XDG_CONFIG_HOME": str(xdg)}
    p, src = discover_config_path(None, tmp_path, env)
    assert p == cfg
    assert src == "xdg"


def test_precedence_env_over_cwd_over_xdg(tmp_path):
    # Create all three locations
    envdir = tmp_path / "envdir"
    envcfg = _touch(envdir / "config.yaml")
    cwdcfg = _touch(tmp_path / "configs" / "config.yaml")
    xdg = tmp_path / "xdg"
    xdgc = _touch(xdg / "clematis" / "config.yaml")

    # env wins
    p, src = discover_config_path(None, tmp_path, {"CLEMATIS_CONFIG": str(envdir), "XDG_CONFIG_HOME": str(xdg)})
    assert p == envcfg and src == "env:CLEMATIS_CONFIG"

    # remove env, cwd wins
    p, src = discover_config_path(None, tmp_path, {"XDG_CONFIG_HOME": str(xdg)})
    assert p == cwdcfg and src == "cwd:configs/config.yaml"

    # remove cwd, xdg wins
    (tmp_path / "configs" / "config.yaml").unlink()
    p, src = discover_config_path(None, tmp_path, {"XDG_CONFIG_HOME": str(xdg)})
    assert p == xdgc and src == "xdg"


def test_none_when_no_candidates(tmp_path):
    p, src = discover_config_path(None, tmp_path, {})
    assert p is None and src == "none"


def test_verbose_logging_includes_source_and_path(tmp_path, capsys):
    cfg = _touch(tmp_path / "configs" / "config.yaml")
    p, src = discover_config_path(None, tmp_path, {})
    assert src == "cwd:configs/config.yaml"
    maybe_log_selected(p, src, verbose=True)
    out = capsys.readouterr().err
    assert "selected=" in out and "cwd:configs/config.yaml" in out