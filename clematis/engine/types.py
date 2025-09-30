from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple
import numpy as np
from numpy import ndarray

PlanVersion = Literal["t3-plan-v1"]
BundleVersion = Literal["t3-bundle-v1"]

# ---- Core datatypes ----


@dataclass
class TurnCtx:
    turn_id: str
    agent_id: str
    scene_tags: List[str]
    now: str  # ISO8601 UTC
    cfg: "Config"


@dataclass
class Node:
    id: str
    label: str
    vec_full: ndarray[np.float32] | None = None
    vec_surface: ndarray[np.float32] | None = None
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    id: str
    src: str
    dst: str
    weight: float
    rel: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptGraph:
    graph_id: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, Edge] = field(default_factory=dict)
    views_surface_k: int = 32
    flags: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    version_etag: str = "v0"


@dataclass
class EpisodeRef:
    id: str
    owner: str
    score: float
    text: str = ""


@dataclass
class T1Result:
    graph_deltas: List[Dict[str, Any]]
    metrics: Dict[str, Any]


@dataclass
class T2Result:
    retrieved: List[EpisodeRef]
    graph_deltas_residual: List[Dict[str, Any]]
    metrics: Dict[str, Any]


@dataclass(frozen=True)
class SpeakOp:
    kind: Literal["Speak"]
    intent: Literal["ack", "question", "assertion", "summary"]
    topic_labels: List[str]
    max_tokens: int


@dataclass(frozen=True)
class RequestRetrieveOp:
    kind: Literal["RequestRetrieve"]
    query: str
    owner: Literal["agent", "world", "any"]
    k: int
    tier_pref: Optional[Literal["exact_semantic", "cluster_semantic", "archive"]] = None
    hints: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EditGraphOp:
    kind: Literal["EditGraph"]
    edits: List[Dict[str, Any]]  # {op:"upsert_node"|"upsert_edge", id|src|dst, weight?, rel?}
    cap: int


@dataclass(frozen=True)
class CreateGraphOp:
    kind: Literal["CreateGraph"]
    title: str
    tags: List[str]


@dataclass(frozen=True)
class SetMetaFilterOp:
    kind: Literal["SetMetaFilter"]
    params: Dict[str, Any]  # {delta_norm_cap?, novelty_cap?, churn_cap?, cooldown_s?}


@dataclass(frozen=True)
class ProposedDelta:
    # Canonical, additive proposal for a target attribute (usually "weight")
    target_kind: Literal["node", "edge"]
    target_id: str  # e.g., "n:node_id" or "e:src|rel|dst"
    attr: str  # e.g., "weight"
    delta: float  # additive change to apply
    op_idx: Optional[int] = None  # index of originating plan.op, if known
    idx: Optional[int] = None  # original index for deterministic replay


@dataclass(frozen=True)
class OpRef:
    kind: str
    idx: int


Op = SpeakOp | RequestRetrieveOp | EditGraphOp | CreateGraphOp | SetMetaFilterOp


@dataclass
class Plan:
    version: PlanVersion
    reflection: bool = False
    ops: List[Op] = field(default_factory=list)
    deltas: List[ProposedDelta] = field(default_factory=list)
    request_retrieve: Optional[Dict[str, Any]] = None


@dataclass
class T4Result:
    approved_deltas: List[ProposedDelta]
    rejected_ops: List[OpRef]
    reasons: List[str]
    metrics: Dict[str, Any]


@dataclass
class ApplyResult:
    applied: int  # number of deltas actually applied
    clamps: int  # number of values clamped to [weight_min, weight_max]
    version_etag: Optional[str]  # new store version after apply
    snapshot_path: Optional[str]  # path if a snapshot was written this turn, else None
    metrics: Dict[str, Any]  # additional counters/timings


@dataclass
class TurnResult:
    line: str
    events: List[Dict[str, Any]] = field(default_factory=list)


# ---- Protocols (APIs) ----


class ConceptGraphStore(Protocol):
    def get_graph(self, gid: str) -> ConceptGraph: ...
    def upsert_nodes(self, gid: str, nodes: List[Node]) -> None: ...
    def upsert_edges(self, gid: str, edges: List[Edge]) -> None: ...
    def apply_deltas(self, gid: str, deltas: List[Dict[str, Any]]) -> Dict[str, Any]: ...
    def surface_view(
        self, gid: str, node_ids: List[str], k: int, method: Literal["PCA", "TopK"]
    ) -> Dict[str, ndarray[np.float32]]: ...
    def csr(self, gid: str) -> Any: ...
    def csc(self, gid: str) -> Any: ...
    def version_etag(self, gid: str) -> str: ...


class MemoryIndex(Protocol):
    def add(self, ep: Dict[str, Any]) -> None: ...
    def search_tiered(
        self,
        owner: Optional[str],
        q_vec: ndarray[np.float32],
        k: int,
        tier: Literal["exact_semantic", "cluster_semantic", "archive"],
        hints: Dict[str, Any],
    ) -> List[EpisodeRef]: ...
    def index_version(self) -> int: ...


class EmbeddingAdapter(Protocol):
    def encode(self, texts: List[str]) -> List[ndarray[np.float32]]: ...


class LLMAdapter(Protocol):
    def deliberate(self, plan_bundle: Dict[str, Any]) -> Plan: ...
    def speak(self, dialog_bundle: Dict[str, Any]) -> str: ...


class MetaFilter(Protocol):
    def gate(self, ctx: TurnCtx, state: Dict[str, Any], proposal: Dict[str, Any]) -> T4Result: ...


class Scheduler(Protocol):
    def choose_agent(self, world_state: Dict[str, Any]) -> str: ...
    def on_timeout(self, agent_id: str) -> None: ...


@dataclass
class Config:
    k_surface: int = 32
    surface_method: Literal["PCA", "TopK"] = "PCA"
    t1: Dict[str, Any] = field(
        default_factory=lambda: {
            "decay": {"mode": "exp_floor", "rate": 0.6, "floor": 0.05},
            "edge_type_mult": {"supports": 1.0, "associates": 0.6, "contradicts": 0.8},
            "iter_cap": 50,  # legacy name; see iter_cap_layers
            "iter_cap_layers": 50,  # layers beyond seeds (depth cap)
            "node_budget": 1.5,
            "queue_budget": 10_000,
            "radius_cap": 4,
            "relax_cap": None,  # optional max total relaxations (edge traversals)
            "cache": {"enabled": True, "max_entries": 512, "ttl_s": 300},
        }
    )
    t2: Dict[str, Any] = field(
        default_factory=lambda: {
            "backend": "inmemory",  # or "lancedb"
            "k_retrieval": 64,
            "sim_threshold": 0.3,
            "tiers": ["exact_semantic", "cluster_semantic", "archive"],
            "exact_recent_days": 30,
            "ranking": {"alpha_sim": 0.75, "beta_recency": 0.2, "gamma_importance": 0.05},
            "clusters_top_m": 3,
            "owner_scope": "any",
            "residual_cap_per_turn": 32,
            "cache": {"enabled": True, "max_entries": 512, "ttl_s": 300},
            "lancedb": {
                "uri": "./.data/lancedb",
                "table": "episodes",
                "meta_table": "meta",
                "index": {"metric": "cosine", "ef_search": 64, "m": 16},
            },
        }
    )
    t3: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_rag_loops": 1,
            "tokens": 256,
            "temp": 0.7,
            "max_ops_per_turn": 3,
            "allow_reflection": False,
            "backend": "rulebased",
            "dialogue": {
                "template": "style_prefix| summary: {labels}. next: {intent}",
                "include_top_k_snippets": 2,
            },
        }
    )
    t4: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "delta_norm_cap_l2": 1.5,
            "novelty_cap_per_node": 0.3,
            "churn_cap_edges": 64,
            "cooldowns": {"EditGraph": 2, "CreateGraph": 10},
            # Forward-looking defaults (used in later PRs, harmless here)
            "weight_min": -1.0,
            "weight_max": 1.0,
            "snapshot_every_n_turns": 1,
            "snapshot_dir": "./.data/snapshots",
            "cache_bust_mode": "on-apply",
        }
    )
    budgets: Dict[str, Any] = field(
        default_factory=lambda: {
            "time_ms": 1000,
            "ops": 1000,
            "tokens": 1024,
            "time_ms_reflection": 6000,
        }
    )
    flags: Dict[str, Any] = field(
        default_factory=lambda: {"enable_world_memory": True, "allow_reflection": True}
    )

    # M5: scheduler config (feature-flagged; defaults merged by validate.py)
    scheduler: Dict[str, Any] = field(default_factory=dict)

# --- M9: deterministic parallelism config typing --------------------------------

try:  # Python 3.11+
    from typing import TypedDict, NotRequired  # type: ignore[attr-defined]
except Exception:  # fallback for older typing support
    from typing_extensions import TypedDict  # type: ignore[assignment]
    try:
        from typing_extensions import NotRequired  # type: ignore[assignment]
    except Exception:
        # For very old environments, NotRequired may not exist; keep types permissive.
        NotRequired = None  # type: ignore

class PerfParallelConfig(TypedDict, total=False):
    """Config surface for deterministic parallel execution (PR63/M9).

    All fields are optional from the type system's POV; validation enforces shape.
    """
    enabled: bool
    max_workers: int  # 0 or 1 => sequential; negative normalized to 0
    t1: bool
    t2: bool
    agents: bool

class PerfMetricsConfig(TypedDict, total=False):
    """Minimal metrics config surface referenced by PR63.

    Only keys surfaced by validate.py today; extended in later PRs.
    """
    report_memory: bool


class PerfConfig(TypedDict, total=False):
    """Top-level perf config container.

    Validation in configs/validate.py is the single source of truth. This
    type is intentionally permissive (total=False) so adding future fields
    does not break imports.
    """
    enabled: bool
    metrics: PerfMetricsConfig
    parallel: PerfParallelConfig


def is_parallel_enabled(perf: Dict[str, Any] | "PerfConfig" | None) -> bool:
    """Return True only when parallel execution should actually run.

    Conditions (post-normalization by validate.py):
    - perf.enabled is True
    - perf.parallel.enabled is True
    - at least one of {t1,t2,agents} is True
    - max_workers >= 2  (0/1 are sequential)
    """
    if not perf:
        return False
    # TypedDicts are dicts at runtime.
    p = perf if isinstance(perf, dict) else dict(perf)  # type: ignore[arg-type]
    if not p.get("enabled"):
        return False
    par = p.get("parallel") or {}
    if not isinstance(par, dict):
        return False
    if not par.get("enabled"):
        return False
    if not any(bool(par.get(k)) for k in ("t1", "t2", "agents")):
        return False
    maxw = int(par.get("max_workers", 0))
    return maxw >= 2


def parallel_max_workers(perf: Dict[str, Any] | "PerfConfig" | None) -> int:
    """Helper that mirrors validate.py normalization for max_workers.

    Values <= 0 are treated as 0 (sequential). This does *not* clamp high values;
    scheduling code should apply its own upper bounds if needed.
    """
    if not perf:
        return 0
    p = perf if isinstance(perf, dict) else dict(perf)  # type: ignore[arg-type]
    par = p.get("parallel") or {}
    if not isinstance(par, dict):
        return 0
    try:
        mw = int(par.get("max_workers", 0))
    except Exception:
        mw = 0
    return mw if mw > 0 else 0
