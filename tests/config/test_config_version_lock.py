

from __future__ import annotations
import pytest

from configs.validate import validate_config, CONFIG_VERSION
from clematis.errors import ConfigError


def _base_cfg() -> dict:
  # Minimal skeleton accepted by the validator; extend as the validator evolves
  return {"t1": {}, "t2": {}, "t3": {}, "t4": {}}


def test_injects_version_when_missing():
  cfg = _base_cfg()
  out = validate_config(dict(cfg))  # validate on a shallow copy
  assert out.get("version") == CONFIG_VERSION


def test_accepts_v1_version():
  cfg = _base_cfg()
  cfg["version"] = CONFIG_VERSION
  out = validate_config(cfg)
  assert out["version"] == CONFIG_VERSION


def test_rejects_wrong_version():
  cfg = _base_cfg()
  cfg["version"] = "v999"
  with pytest.raises(ConfigError) as e:
    validate_config(cfg)
  # Message should be explicit but avoid overfitting exact text
  assert "must be" in str(e.value)


def test_rejects_unknown_top_level_keys():
  cfg = _base_cfg()
  cfg["whaaat"] = {}
  with pytest.raises(ConfigError):
    validate_config(cfg)
