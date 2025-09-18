from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple
from numpy.typing import NDArray
import numpy as np

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
    vec_full: NDArray[np.float32] | None = None
    vec_surface: NDArray[np.float32] | None = None
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

@dataclass
class Plan:
    version: str
    ops: List[Dict[str, Any]]
    request_retrieve: Optional[Dict[str, Any]] = None
    reflection: bool = False

@dataclass
class T4Result:
    approved_deltas: List[Dict[str, Any]]
    rejected_ops: List[Dict[str, Any]]
    reasons: List[str]
    metrics: Dict[str, Any]

@dataclass
class ApplyResult:
    applied: Dict[str, Any]
    clamps: List[str]
    line: str
    metrics: Dict[str, Any]

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
    def surface_view(self, gid: str, node_ids: List[str], k: int, method: Literal["PCA","TopK"]) -> Dict[str, NDArray[np.float32]]: ...
    def csr(self, gid: str) -> Any: ...
    def csc(self, gid: str) -> Any: ...
    def version_etag(self, gid: str) -> str: ...

class MemoryIndex(Protocol):
    def add(self, ep: Dict[str, Any]) -> None: ...
    def search_tiered(self, owner: Optional[str], q_vec: NDArray[np.float32], k: int, tier: Literal["exact_semantic","cluster_semantic","archive"], hints: Dict[str, Any]) -> List[EpisodeRef]: ...

class EmbeddingAdapter(Protocol):
    def encode(self, texts: List[str]) -> List[NDArray[np.float32]]: ...

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
    surface_method: Literal["PCA","TopK"] = "PCA"
    t1: Dict[str, Any] = field(default_factory=lambda: {"decay":{"mode":"exp_floor","rate":0.6,"floor":0.05},
                                                        "edge_type_mult":{"supports":1.0,"associates":0.6,"contradicts":0.8},
                                                        "iter_cap":50,"node_budget":1.5,"queue_budget":10_000,"radius_cap":4})
    t2: Dict[str, Any] = field(default_factory=lambda: {"k_retrieval":64,"sim_threshold":0.3,
                                                        "tiers":["exact_semantic","cluster_semantic","archive"],
                                                        "ranking":{"alpha_sim":0.75,"beta_recency":0.2,"gamma_importance":0.05},
                                                        "clusters_top_m":3})
    t3: Dict[str, Any] = field(default_factory=lambda: {"max_rag_loops":1,"tokens":512,"temp":0.7})
    t4: Dict[str, Any] = field(default_factory=lambda: {"delta_norm_cap":2.0,"novelty_cap":1.0,"churn_cap":64,"cooldown_s":60})
    budgets: Dict[str, Any] = field(default_factory=lambda: {"time_ms":1000,"ops":1000,"tokens":1024,"time_ms_reflection":6000})
    flags: Dict[str, Any] = field(default_factory=lambda: {"enable_world_memory":True,"allow_reflection":True})
