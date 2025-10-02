from __future__ import annotations

from typing import Dict, List, Tuple
from types import SimpleNamespace as SNS
import string

__all__ = [
    "agent_ids",
    "make_state_disjoint",
    "make_state_overlapping",
    "make_tasks",
    "fixed_ctx",
]


def agent_ids(n: int) -> List[str]:
    """Return the first *n* uppercase agent IDs (A, B, C, ...)."""
    assert 1 <= n <= 26, "n must be between 1 and 26"
    return list(string.ascii_uppercase[:n])


def make_state_disjoint(n_agents: int = 2) -> Dict[str, Dict[str, List[str]]]:
    """A minimal state with pairwise-disjoint graph sets per agent.

    Returns both shapes recognized by orchestrator helpers:
      * state["graphs_by_agent"][agent_id] -> List[str]
      * state["agents"][agent_id]["graphs"] -> List[str]
    """
    gids = agent_ids(n_agents)
    gba: Dict[str, List[str]] = {}
    agents: Dict[str, Dict[str, List[str]]] = {}
    for aid in gids:
        graphs = [f"G{aid}"]
        gba[aid] = graphs
        agents[aid] = {"graphs": list(graphs)}
    return {"graphs_by_agent": gba, "agents": agents}


def make_state_overlapping(a: str = "A", b: str = "B") -> Dict[str, Dict[str, List[str]]]:
    """A minimal state where two agents share the same graph set (overlap)."""
    shared = ["G1"]
    gba = {a: list(shared), b: list(shared)}
    agents = {a: {"graphs": list(shared)}, b: {"graphs": list(shared)}}
    return {"graphs_by_agent": gba, "agents": agents}


def make_tasks(n_agents: int = 2) -> List[Tuple[str, str]]:
    """Deterministic (agent, text) pairs for tests.

    Texts cycle through a small fixed list so ordering is easy to assert.
    """
    texts = ["hi", "yo", "ok", "go"]
    out: List[Tuple[str, str]] = []
    for i, aid in enumerate(agent_ids(n_agents)):
        out.append((aid, texts[i % len(texts)]))
    return out


def fixed_ctx(*, turn_id: int, cfg: dict) -> SNS:
    """Create a minimal TurnCtx stub with fixed turn_id and provided cfg."""
    return SNS(cfg=cfg, turn_id=int(turn_id))
