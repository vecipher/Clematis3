

# -*- coding: utf-8 -*-
import importlib


def test_reflection_stage_ord_constant():
  L = importlib.import_module("clematis.engine.util.io_logging")
  assert L.STAGE_ORD.get("t3_reflection.jsonl") == 10, "t3_reflection.jsonl must have stage ordinal 10"
  assert isinstance(L.STAGE_ORD["t3_reflection.jsonl"], int)

def test_reflection_not_in_identity_logs():
  L = importlib.import_module("clematis.engine.util.io_logging")
  identity_set = getattr(L, "_IDENTITY_LOGS", set())
  assert "t3_reflection.jsonl" not in identity_set, "reflection log must not participate in identity checks"
