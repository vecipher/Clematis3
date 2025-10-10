#!/usr/bin/env python3
"""Interactive Clematis chat loop."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, List

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.validate import validate_config_api
from scripts.console import adapter_reset, warn_nondeterminism
from clematis.cli._config import discover_config_path
from clematis.engine.stages.t2.state import _init_index_from_cfg
from clematis.graph.store import InMemoryGraphStore
from clematis.engine.types import Node, Edge
from clematis.memory.index import InMemoryIndex

DEFAULT_LOG_DIR = Path(".logs") / "chat"
DEFAULT_FIXTURE = Path("fixtures") / "llm" / "qwen_small.jsonl"
DEFAULT_TEMPLATE_PATH = (Path(REPO_ROOT) / "configs" / "prompts" / "clematis_dialogue_template.txt").resolve()
DEFAULT_IDENTITY = (
    "You are Clematis, the knowledge gardener. Maintain this persona, remember user-provided facts accurately, "
    "respond concisely, and never claim to be Qwen or mention Alibaba Cloud."
)
DEFAULT_GRAPH_ID = "g:surface"
MAX_HISTORY = 12
_STOPWORDS = {
    "the",
    "and",
    "that",
    "have",
    "with",
    "this",
    "your",
    "from",
    "into",
    "about",
    "just",
    "like",
    "been",
    "will",
    "what",
    "when",
    "where",
    "there",
    "them",
    "then",
    "over",
    "only",
    "into",
    "upon",
    "onto",
    "such",
    "some",
    "more",
    "many",
}
DEFAULT_LANCEDB_URI = Path(os.getenv("CLEMATIS_LANCEDB_URI", ".data/lancedb_chat"))
DEFAULT_LANCEDB_TABLE = "episodes"
DEFAULT_LANCEDB_META = "meta"

SEED_MEMORIES = [
    {
        "id": "seed_coach",
        "owner": "demo",
        "text": "Clematis coached a user about maintaining a concept graph with strong labels.",
        "importance": 0.75,
        "tags": ["seed", "demo"],
    },
    {
        "id": "seed_story",
        "owner": "bot",
        "text": "An earlier conversation explored stories about botanical gardens and their symbolism.",
        "importance": 0.6,
        "tags": ["seed", "story"],
    },
    {
        "id": "seed_task",
        "owner": "demo",
        "text": "We evaluated retrieval quality by asking for summaries of recent tasks and reflections.",
        "importance": 0.8,
        "tags": ["seed", "task"],
    },
    {
        "id": "seed_graph",
        "owner": "bot",
        "text": "The assistant mapped related concepts like lattice, vine, and bloom inside the surface graph.",
        "importance": 0.7,
        "tags": ["seed", "graph"],
    },
    {
        "id": "seed_journal",
        "owner": "demo",
        "text": "Clematis guided a traveler through dreams and memory, awakening long-forgotten ideas.",
        "importance": 0.65,
        "tags": ["seed", "story", "forest", "memories"],
    },
    {
        "id": "seed_lattice",
        "owner": "bot",
        "text": "A lattice of thoughts once converged into a bloom called Ambrose—an echo of an older age.",
        "importance": 0.7,
        "tags": ["seed", "bot", "lattice", "ambrose"],
    },
    {
        "id": "seed_well",
        "owner": "demo",
        "text": "Beside a moonlit well, Clematis listened to Vecipher describe a manor hidden by ivy.",
        "importance": 0.75,
        "tags": ["seed", "manor", "vecipher", "moonlight"],
    },
    {
        "id": "seed_dell",
        "owner": "bot",
        "text": "The garden once became a moonlit dell; footsteps and laughter still linger there.",
        "importance": 0.6,
        "tags": ["seed", "garden", "dell", "moonlight"],
    },
    {
        "id": "seed_wine",
        "owner": "demo",
        "text": "Clematis never brewed wine but remembers the scent of cracked casks along the manor road.",
        "importance": 0.7,
        "tags": ["seed", "wine", "manor", "memories"],
    },
    {
        "id": "seed_solstice",
        "owner": "bot",
        "text": "During the solstice, the breeze carried petals and riddles across the canopy.",
        "importance": 0.65,
        "tags": ["seed", "solstice", "forest", "breeze"],
    },
    {
        "id": "seed_library",
        "owner": "demo",
        "text": "Clematis cataloged memories into a living library of roots and vines.",
        "importance": 0.72,
        "tags": ["seed", "library", "roots", "vines"],
    },
    {
        "id": "seed_watch",
        "owner": "bot",
        "text": "Water watches quietly as stars fall into the reflection pool.",
        "importance": 0.6,
        "tags": ["seed", "water", "stars", "reflection"],
    },
]


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or yaml is None:
        raw: Dict[str, Any] = {}
    else:
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            raw = {}
    ok, errs, cfg = validate_config_api(dict(raw))
    if not ok or cfg is None:
        msg = "\n".join(errs or ["invalid configuration"])
        print(msg, file=sys.stderr)
        raise SystemExit(2)
    return cfg


class _AttrDict(dict):
    """Dict that also supports attribute access recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return _AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_attrdict(v) for v in obj)
    return obj


def _ensure_store(state: Dict[str, Any]) -> InMemoryGraphStore:
    store = state.get("store")
    if not isinstance(store, InMemoryGraphStore):
        store = InMemoryGraphStore()
        state["store"] = store
    graph = store.ensure(DEFAULT_GRAPH_ID)
    state.setdefault("active_graphs", [DEFAULT_GRAPH_ID])
    if not state.get("_chat_seeded_graph"):
        root = Node(id="n:root", label="memory-root", attrs={"notes": "chat demo root"})
        store.upsert_nodes(DEFAULT_GRAPH_ID, [root])
        state["_chat_seeded_graph"] = True
    return store


def _empty_state() -> Dict[str, Any]:
    empty_meta = {
        "schema": "v1.1",
        "merges": [],
        "splits": [],
        "promotions": [],
        "concept_nodes_count": 0,
        "edges_count": 0,
    }
    graph = {"nodes": {}, "edges": {}, "meta": dict(empty_meta)}
    state = {"graph": dict(graph), "gel": dict(graph), "version_etag": "0", "logs": [], "store": {}}
    return state


def _load_state(snapshot_path: Optional[str]) -> Dict[str, Any]:
    try:
        state = adapter_reset(snapshot_path)
    except SystemExit as exc:
        code = getattr(exc, "code", None)
        if snapshot_path or code not in (None, 0, 2):
            raise
        print("[chat] WARNING: no snapshot available; starting from empty state", file=sys.stderr)
        state = _empty_state()
    if isinstance(state, dict):
        state.setdefault("logs", [])
        store = state.get("store")
        if not isinstance(store, InMemoryGraphStore):
            state["store"] = {}
        state.setdefault("_chat_history", [])
        state.setdefault("_chat_memory_usage", {})
        state.setdefault("_chat_mem_node_map", {})
        state.setdefault("_chat_memories", [])
    return state


def _get_embedding_adapter(dim: int):
    try:
        from clematis.adapters.embeddings import BGEAdapter  # local import for optional dep
    except ModuleNotFoundError as exc:
        missing = exc.name or "numpy"
        print(f"Missing dependency for embeddings: {missing}. Install project requirements.", file=sys.stderr)
        return None
    return BGEAdapter(dim=dim)


def _tune_t2(cfg_root: Dict[str, Any]) -> None:
    t2 = cfg_root.setdefault("t2", {})
    t2.setdefault("backend", "lancedb")
    t2.setdefault("sim_threshold", 0.05)
    t2.setdefault("k_retrieval", 6)
    t2.setdefault("exact_recent_days", 365)
    ranking = t2.setdefault("ranking", {})
    ranking.setdefault("alpha_sim", 0.6)
    ranking.setdefault("beta_recency", 0.35)
    ranking.setdefault("gamma_importance", 0.05)
    ldb = t2.setdefault("lancedb", {})
    ldb.setdefault("uri", str(DEFAULT_LANCEDB_URI))
    ldb.setdefault("table", DEFAULT_LANCEDB_TABLE)
    ldb.setdefault("meta_table", DEFAULT_LANCEDB_META)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _owner_quarter(owner: str, iso_ts: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        quarter = (dt.month - 1) // 3 + 1
        return f"{owner}_{dt.year}Q{quarter}"
    except Exception:
        return f"{owner}_unknown"


def _record_memory_id(state: Dict[str, Any], mem_id: str) -> None:
    ids = state.setdefault("_chat_memory_ids", [])
    ids.append(mem_id)


def _memory_exists(state: Dict[str, Any], mem_id: str) -> bool:
    ids = state.get("_chat_memory_ids", [])
    return mem_id in ids


def _extract_keywords(text: str, limit: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    keywords: List[str] = []
    for tok in tokens:
        if len(tok) <= 3:
            continue
        if tok in _STOPWORDS:
            continue
        if tok not in keywords:
            keywords.append(tok)
        if len(keywords) >= limit:
            break
    return keywords


def _speaker_label(owner: str, kind: str) -> str:
    own = (owner or "").lower()
    if kind == "dialogue":
        if own in {"agent", "user"}:
            return "user"
        if own in {"assistant", "bot"}:
            return "assistant"
    if own in {"demo", "seed"}:
        return "seed"
    return owner


def _append_history(state: Dict[str, Any], role: str, text: str, turn: int) -> None:
    if not text:
        return
    history = state.setdefault("_chat_history", [])
    history.append({"role": role, "text": text, "turn": turn})
    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]


def _mark_memory_usage(state: Dict[str, Any], mem_ids: List[str]) -> None:
    if not mem_ids:
        return
    usage = state.setdefault("_chat_memory_usage", {})
    idx = state.get("mem_index")
    store = state.get("store")
    node_map = state.setdefault("_chat_mem_node_map", {})
    mem_records = state.setdefault("_chat_memories", [])
    for mem_id in dict.fromkeys(mem_ids):
        if not mem_id:
            continue
        usage[mem_id] = usage.get(mem_id, 0) + 1
        usage_tag = f"usage:{usage[mem_id]}"
        if hasattr(idx, "_eps"):
            try:
                for ep in getattr(idx, "_eps", []):
                    if ep.get("id") == mem_id:
                        tags = list(ep.get("tags") or [])
                        tags = [t for t in tags if not str(t).startswith("usage:")]
                        tags.append(usage_tag)
                        ep["tags"] = tags
                        ep["usage_count"] = usage[mem_id]
                        break
            except Exception:
                pass
        if isinstance(store, InMemoryGraphStore):
            try:
                node_id = node_map.get(mem_id)
                if node_id:
                    graph = store.get_graph(DEFAULT_GRAPH_ID)
                    node = graph.nodes.get(node_id)
                    if node:
                        tags = list(node.attrs.get("tags", []))
                        tags = [t for t in tags if not str(t).startswith("usage:")]
                        tags.append(usage_tag)
                        node.attrs["tags"] = tags
                        node.attrs["usage_count"] = usage[mem_id]
            except Exception:
                pass
        for rec in mem_records:
            if rec.get("id") == mem_id:
                rec["usage_count"] = usage[mem_id]
                tags = [t for t in rec.get("tags", []) if not str(t).startswith("usage:")]
                tags.append(usage_tag)
                rec["tags"] = tags
                break


def _print_memories(state: Dict[str, Any], limit: int = 10, *, verbose: bool = False) -> None:
    mem_records = list(state.get("_chat_memories", []))
    if not mem_records:
        print("[mem] no memories stored yet.")
        return
    records = list(mem_records)[-limit:][::-1]
    if verbose:
        print(f"[mem-v] showing up to {limit} memories (most recent first):")
        for rec in records:
            ident = rec.get("id")
            owner = rec.get("owner")
            usage = rec.get("usage_count", 0)
            turn = rec.get("turn")
            text = rec.get("text", "")
            kind = rec.get("kind") or ("seed" if (turn is None or turn < 0) else "dialogue")
            speaker = rec.get("speaker") or _speaker_label(owner, kind)
            tags = ", ".join(str(t) for t in rec.get("tags", []))
            print(
                f"  {ident} (kind={kind}, speaker={speaker}, owner={owner}, usage={usage}, turn={turn}) -> {text}"
            )
            if tags:
                print(f"    tags: {tags}")
    else:
        print(f"[mem] showing up to {limit} memories (most recent first):")
        for rec in records:
            ident = rec.get("id")
            owner = rec.get("owner")
            text = rec.get("text", "")
            used = rec.get("usage_count", 0)
            turn = rec.get("turn")
            kind = rec.get("kind") or ("seed" if (turn is None or turn < 0) else "dialogue")
            speaker = rec.get("speaker") or _speaker_label(owner, kind)
            if kind == "dialogue":
                label = speaker
            else:
                label = "seed"
            prefix = f"{ident} [{label}]"
            if used:
                prefix += f" (used x{used})"
            print(f"  {prefix}: {text}")


def _ensure_index(state: Dict[str, Any], cfg_t2: Dict[str, Any]):
    idx, backend_selected, fallback_reason = _init_index_from_cfg(state, cfg_t2)
    if fallback_reason and not state.get("_chat_index_warning"):
        print(
            f"[chat] WARNING: using fallback in-memory index (reason: {fallback_reason}). "
            "Install LanceDB support with `pip install \"clematis[lancedb]\"` to enable the configured backend.",
            file=sys.stderr,
        )
        state["_chat_index_warning"] = True
    state["_chat_index_backend"] = backend_selected
    return idx


def _add_memory_entry(
    state: Dict[str, Any],
    cfg_t2: Dict[str, Any],
    adapter,
    *,
    text: str,
    owner: str,
    tags: Optional[list[str]] = None,
    importance: float = 0.5,
    mem_id: Optional[str] = None,
    turn: Optional[int] = None,
) -> None:
    if adapter is None or not text:
        return
    idx = _ensure_index(state, cfg_t2)
    seq = state.get("_chat_memory_seq", 0)
    if mem_id is None:
        mem_id = f"chat_{seq:04d}"
    if _memory_exists(state, mem_id):
        return
    vec = adapter.encode([text])[0]
    ts = _now_iso()
    role_tag = f"role:{owner}"
    base_tags = list(tags or []) if tags else []
    if role_tag not in base_tags:
        base_tags.append(role_tag)
    base_tags.append("source:chat")
    for kw in _extract_keywords(text):
        if kw not in base_tags:
            base_tags.append(kw)
        tag_kw = f"kw:{kw}"
        if tag_kw not in base_tags:
            base_tags.append(tag_kw)
    base_tags = list(dict.fromkeys(base_tags))
    rec_kind = "seed" if (turn is None or turn < 0) else "dialogue"
    speaker = _speaker_label(owner, rec_kind)
    speaker_tag = f"speaker:{speaker}"
    if speaker_tag not in base_tags:
        base_tags.append(speaker_tag)
    base_tags = list(dict.fromkeys(base_tags))
    episode = {
        "id": mem_id,
        "owner": owner,
        "role": owner,
        "text": text,
        "ts": ts,
        "tags": base_tags,
        "importance": float(importance),
        "quarter": _owner_quarter(owner, ts),
        "turn": turn,
        "usage_count": 0,
        "vec_full": vec.astype("float32").tolist(),
    }
    idx.add(episode)
    _record_memory_id(state, mem_id)
    state["_chat_memory_seq"] = seq + 1

    # Also mirror into the concept graph for T1/T4 visibility
    store = _ensure_store(state)
    node_seq = state.get("_chat_node_seq", 0)
    node_id = f"n:mem_{node_seq:04d}"
    label = text.split(".")[0][:48] or f"memory-{node_seq}"
    store.upsert_nodes(
        DEFAULT_GRAPH_ID,
        [
            Node(
                id=node_id,
                label=label,
                attrs={
                    "text": text,
                    "owner": owner,
                    "role": owner,
                    "speaker": speaker,
                    "tags": base_tags,
                    "mem_id": mem_id,
                    "turn": turn,
                    "usage_count": 0,
                },
            )
        ],
    )
    # Connect memory node to root (both directions) for propagation.
    root_id = "n:root"
    edge_fw = Edge(
        id=f"e:{root_id}->{node_id}",
        src=root_id,
        dst=node_id,
        weight=0.6,
        rel="associates",
    )
    edge_bw = Edge(
        id=f"e:{node_id}->{root_id}",
        src=node_id,
        dst=root_id,
        weight=0.4,
        rel="associates",
    )
    store.upsert_edges(DEFAULT_GRAPH_ID, [edge_fw, edge_bw])
    state.setdefault("_chat_mem_node_map", {})[mem_id] = node_id
    state["_chat_node_seq"] = node_seq + 1
    mem_records = state.setdefault("_chat_memories", [])
    for rec in mem_records:
        if rec.get("id") == mem_id:
            break
    else:
        mem_records.append(
            {
                "id": mem_id,
                "owner": owner,
                "speaker": speaker,
                "text": text,
                "tags": list(base_tags),
                "turn": turn,
                "usage_count": 0,
                "kind": rec_kind,
            }
        )


def _seed_memories(
    state: Dict[str, Any],
    cfg_t2: Dict[str, Any],
    adapter,
    *,
    force: bool = False,
) -> None:
    if adapter is None:
        return
    _ensure_index(state, cfg_t2)
    seeded = bool(state.get("_chat_seeded_memories"))
    if seeded and not force:
        return
    for row in SEED_MEMORIES:
        _add_memory_entry(
            state,
            cfg_t2,
            adapter,
            text=row["text"],
            owner=row["owner"],
            tags=row.get("tags"),
            importance=row.get("importance", 0.5),
            mem_id=row["id"],
            turn=-1,
        )
    state["_chat_seeded_memories"] = True


def _wipe_memories(state: Dict[str, Any], cfg_t2: Dict[str, Any]) -> None:
    idx = state.get("mem_index")
    if idx is not None:
        clear = getattr(idx, "clear", None)
        if callable(clear):
            try:
                clear()
            except Exception as exc:  # pragma: no cover - defensive logging path
                print(f"[chat] WARNING: failed to clear memory index: {exc}", file=sys.stderr)
    state.pop("mem_index", None)
    state.pop("mem_backend", None)
    state.pop("mem_backend_fallback_reason", None)
    state["_chat_seeded_memories"] = False
    state["_chat_memory_ids"] = []
    state["_chat_memory_seq"] = 0
    state["_chat_node_seq"] = 0
    state["_chat_history"] = []
    state["_chat_memory_usage"] = {}
    state["_chat_mem_node_map"] = {}
    state["_chat_memories"] = []
    state["_chat_index_warning"] = False
    state["_chat_index_backend"] = None
    # Reset store/graph snapshots
    state["store"] = InMemoryGraphStore()
    state["graph"] = {"nodes": {}, "edges": {}, "meta": {"schema": "v1.1"}}
    state["gel"] = {"nodes": {}, "edges": {}, "meta": {"schema": "v1.1"}}
    state["active_graphs"] = [DEFAULT_GRAPH_ID]
    state["_chat_seeded_graph"] = False
    _ensure_store(state)
    # Recreate the configured index so subsequent seeds write to the persistent store.
    try:
        _ensure_index(state, cfg_t2)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[chat] WARNING: failed to reinitialise memory index after wipe: {exc}", file=sys.stderr)
        state["mem_index"] = InMemoryIndex()
        state["mem_backend"] = "inmemory"


def _apply_llm_mode(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    t3 = cfg.setdefault("t3", {})
    t3.setdefault("enabled", True)
    t3["apply_ops"] = not bool(getattr(args, "no_apply_ops", False))
    llm_cfg = t3.setdefault("llm", {})
    if args.llm_mode == "rulebased":
        t3["backend"] = "rulebased"
        return
    t3["backend"] = "llm"
    if args.llm_mode == "fixture":
        llm_cfg["provider"] = "fixture"
        fx = llm_cfg.setdefault("fixtures", {})
        fx["enabled"] = True
        fx["path"] = str(args.fixture_path or DEFAULT_FIXTURE)
    else:  # live
        llm_cfg["provider"] = "ollama"
        llm_cfg["endpoint"] = args.endpoint
        llm_cfg["model"] = args.model
        llm_cfg["temp"] = float(args.temp)
        llm_cfg["timeout_ms"] = int(args.timeout_ms)
        fx = llm_cfg.setdefault("fixtures", {})
        fx["enabled"] = False


def _tail_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None


def _iso_from_ms(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.isoformat()


def _prepare_ctx(cfg: Any, agent_id: str, turn: int, now_ms: int, text: str):
    ctx = SimpleNamespace()
    ctx.turn_id = str(turn)
    ctx.agent_id = agent_id
    ctx.now_ms = now_ms
    ctx.now = _iso_from_ms(now_ms)
    ctx.cfg = cfg
    ctx.config = cfg
    ctx.input_text = text
    try:
        dialogue_cfg = cfg.t3.dialogue  # type: ignore[attr-defined]
        ctx.style_prefix = getattr(dialogue_cfg, "style_prefix", "")
        ctx.identity = getattr(dialogue_cfg, "identity", DEFAULT_IDENTITY)
    except Exception:
        ctx.style_prefix = ""
    return ctx


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Clematis chat demo.")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml (auto-discovered if omitted).")
    p.add_argument("--snapshot", type=str, default=None, help="Seed state from snapshot JSON.")
    p.add_argument("--log-dir", type=str, default=None, help="Directory for CLEMATIS_LOG_DIR (default: ./.logs/chat).")
    p.add_argument("--agent", type=str, default="clematis", help="Agent id for the turn context.")
    p.add_argument("--llm-mode", choices=["rulebased", "fixture", "live"], default="live", help="Planner/dialogue backend.")
    p.add_argument("--model", type=str, default="qwen3:4b-instruct", help="Model name for --llm-mode=live.")
    p.add_argument("--endpoint", type=str, default="http://localhost:11434/api/generate", help="Endpoint for --llm-mode=live.")
    p.add_argument("--temp", type=float, default=0.2, help="Temperature passed to the LLM.")
    p.add_argument("--timeout-ms", type=int, default=10000, help="Timeout for the LLM call in milliseconds.")
    p.add_argument("--fixture-path", type=str, default=str(DEFAULT_FIXTURE), help="Fixture JSONL for --llm-mode=fixture.")
    p.add_argument("--no-apply-ops", action="store_true", help="Disable T3 apply_ops (graph edits).")
    p.add_argument("--now-ms", type=int, default=None, help="Initial logical timestamp (ms since epoch).")
    p.add_argument("--step-ms", type=int, default=1000, help="Delta applied to now_ms each turn.")
    p.add_argument("--show-plan", action="store_true", help="Print a compact plan/metrics summary each turn.")
    p.add_argument("--no-network-ban", action="store_true", help="Unset CLEMATIS_NETWORK_BAN for live LLM smoke.")
    p.add_argument("--no-seed", action="store_true", help="Skip seeding baseline retrieval memories.")
    p.add_argument("--no-auto-memories", action="store_true", help="Skip writing chat turns back into memory index.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    cfg_path, _ = discover_config_path(args.config)
    if cfg_path is not None and not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg_dict = _load_config(cfg_path)
    t3_cfg = cfg_dict.setdefault("t3", {})
    dialogue_cfg = t3_cfg.setdefault("dialogue", {})
    dialogue_cfg.setdefault("identity", DEFAULT_IDENTITY)
    dialogue_cfg.setdefault("style_prefix", "clematis")
    if DEFAULT_TEMPLATE_PATH.exists():
        dialogue_cfg.setdefault("template_file", str(DEFAULT_TEMPLATE_PATH))
    _apply_llm_mode(cfg_dict, args)
    _tune_t2(cfg_dict)
    try:
        DEFAULT_LANCEDB_URI.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    cfg = _to_attrdict(cfg_dict)

    log_dir = Path(args.log_dir or DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CLEMATIS_LOG_DIR"] = str(log_dir)
    os.environ["CLEMATIS_T3_ALLOW"] = "1"
    os.environ["CLEMATIS_T3_APPLY_OPS"] = "0" if args.no_apply_ops else "1"
    os.environ["CLEMATIS_LLM_MODE"] = args.llm_mode
    if args.no_network_ban:
        os.environ["CLEMATIS_NETWORK_BAN"] = "0"
    else:
        os.environ.setdefault("CLEMATIS_NETWORK_BAN", "1")

    warn_nondeterminism()

    snapshot = args.snapshot
    state = _load_state(snapshot)
    store = _ensure_store(state)

    adapter = _get_embedding_adapter(int(cfg_dict.get("k_surface", 32)))
    if adapter is None:
        return 2
    cfg_t2 = cfg_dict.setdefault("t2", {})
    if not args.no_seed:
        _seed_memories(state, cfg_t2, adapter, force=False)

    try:
        from clematis.engine.orchestrator.core import run_turn
    except ModuleNotFoundError as exc:
        mod = exc.name or "dependency"
        print(f"Missing dependency: {mod}. Install project requirements to run chat.", file=sys.stderr)
        return 2

    now_ms = args.now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    turn = 1

    print("Interactive Clematis chat (type /exit to quit, /reset to reload snapshot).")
    print(f"Config: {cfg_path if cfg_path else 'defaults'}")
    print(f"Logs:   {log_dir}")
    if args.llm_mode != "rulebased":
        print("LLM backend active. Ensure Ollama is running for --llm-mode=live.")
    print("Commands: [wipe] clears memories, [seed] restores demo memories.")

    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not text:
            continue
        if text.lower() in {"/exit", ":exit", "/quit", ":quit"}:
            break
        lowered = text.lower()
        if lowered in {"/reset", ":reset"}:
            state = _load_state(snapshot)
            _ensure_store(state)
            if not args.no_seed:
                _seed_memories(state, cfg_t2, adapter, force=False)
            turn = 1
            now_ms = args.now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
            print("[reset] state reloaded.")
            continue
        if lowered in {"[wipe]", "/wipe"}:
            _wipe_memories(state, cfg_t2)
            print("[wipe] memories cleared.")
            if not args.no_seed:
                print("  (use [seed] to restore demo memories.)")
            continue
        if lowered in {"[seed]", "/seed"}:
            _seed_memories(state, cfg_t2, adapter, force=True)
            print("[seed] baseline memories restored.")
            continue
        if lowered in {"[mem]", "/mem", ":mem"}:
            _print_memories(state, verbose=False)
            continue
        if lowered in {"[mem-v]", "/mem-v", ":mem-v"}:
            _print_memories(state, verbose=True)
            continue

        _append_history(state, "user", text, turn)
        ctx = _prepare_ctx(cfg, args.agent, turn, now_ms, text)
        result = run_turn(ctx, state, text)
        print(result.line)
        if not args.no_auto_memories:
            _add_memory_entry(
                state,
                cfg_t2,
                adapter,
                text=text,
                owner="agent",
                tags=["chat", "user"],
                turn=turn,
            )
            if result.line:
                _add_memory_entry(
                    state,
                    cfg_t2,
                    adapter,
                    text=result.line,
                    owner="assistant",
                    tags=["chat", "assistant"],
                    turn=turn,
                )
        _append_history(state, "assistant", result.line, turn)
        retrieved_ids = list(state.pop("_chat_last_retrieved", []))
        if retrieved_ids:
            _mark_memory_usage(state, retrieved_ids)

        if args.show_plan:
            plan_entry = _tail_jsonl(log_dir / "t3_plan.jsonl")
            dialogue_entry = _tail_jsonl(log_dir / "t3_dialogue.jsonl")
            if plan_entry:
                ops = plan_entry.get("ops_counts", {})
                backend = plan_entry.get("backend")
                print(f"  plan: backend={backend} ops={ops} rag={plan_entry.get('rag_used')}")
                if plan_entry.get("retrieved_ids"):
                    print(f"    retrieved_ids={plan_entry.get('retrieved_ids')}")
            if dialogue_entry:
                print(
                    f"  speak: tokens={dialogue_entry.get('tokens')} truncated={dialogue_entry.get('truncated')} backend={dialogue_entry.get('backend')}"
                )
            usage = state.get("_chat_memory_usage", {})
            if usage:
                recent_usage = {k: usage[k] for k in list(usage.keys())[-4:]}
                print(f"  memory_usage={recent_usage}")

        turn += 1
        now_ms += int(args.step_ms)

    print("bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
