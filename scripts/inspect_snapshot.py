

#!/usr/bin/env python3
"""
Inspect the latest snapshot.

Usage:
  python3 scripts/inspect_snapshot.py [--dir DIR] [--format pretty|json]

Exit codes:
  0 => snapshot found and printed
  2 => no snapshot found or unreadable
"""
import argparse
import json
import os
import sys

try:
    from clematis.engine.snapshot import get_latest_snapshot_info
except Exception:
    # Fallback: add repo root to sys.path and retry
    HERE = os.path.abspath(os.path.dirname(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from clematis.engine.snapshot import get_latest_snapshot_info  # type: ignore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect the latest snapshot")
    ap.add_argument("--dir", default="./.data/snapshots", help="Snapshot directory")
    ap.add_argument("--format", choices=["pretty", "json"], default="pretty")
    args = ap.parse_args(argv)

    info = get_latest_snapshot_info(args.dir)
    if info is None:
        print(f"No snapshot found in {args.dir}", file=sys.stderr)
        return 2

    # Enrich from the snapshot file: nodes/edges, graph schema, and GEL meta counts (PR24)
    try:
        with open(info["path"], "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        gel = payload.get("gel") or {}
        graph_summary = payload.get("graph") or {}

        # Graph/GEL schema version
        gsv = payload.get("graph_schema_version")
        if gsv:
            info["graph_schema_version"] = gsv

        # Prefer full GEL structure for counts
        gel_nodes_cnt = None
        gel_edges_cnt = None
        if isinstance(gel, dict):
            nmap = gel.get("nodes")
            emap = gel.get("edges")
            if isinstance(nmap, dict):
                gel_nodes_cnt = len(nmap)
            if isinstance(emap, dict):
                gel_edges_cnt = len(emap)
            meta = gel.get("meta") or {}
            if isinstance(meta, dict):
                merges = meta.get("merges")
                splits = meta.get("splits")
                promos = meta.get("promotions")
                concepts = meta.get("concept_nodes_count")
                mschema = meta.get("schema")
                if isinstance(merges, list):
                    info["gel_merges"] = len(merges)
                if isinstance(splits, list):
                    info["gel_splits"] = len(splits)
                if isinstance(promos, list):
                    info["gel_promotions"] = len(promos)
                if isinstance(concepts, int):
                    info["gel_concepts"] = concepts
                if isinstance(mschema, str):
                    info["gel_meta_schema"] = mschema
                # If edges_count present in meta, prefer it for gel_edges when we don't have a dict
                if gel_edges_cnt is None:
                    try:
                        ec_meta = int(meta.get("edges_count"))
                        gel_edges_cnt = ec_meta
                    except Exception:
                        pass

        # Fallback to compact graph summary
        if gel_nodes_cnt is None:
            nodes_cnt = graph_summary.get("nodes_count")
            if nodes_cnt is not None:
                gel_nodes_cnt = nodes_cnt
        if gel_edges_cnt is None:
            edges_cnt = graph_summary.get("edges_count")
            if edges_cnt is not None:
                gel_edges_cnt = edges_cnt

        # Write back if we found anything, without clobbering existing explicit values
        if info.get("nodes") is None and gel_nodes_cnt is not None:
            info["nodes"] = gel_nodes_cnt
        if info.get("edges") is None and gel_edges_cnt is not None:
            info["edges"] = gel_edges_cnt
        if gel_nodes_cnt is not None:
            info["gel_nodes"] = gel_nodes_cnt
        if gel_edges_cnt is not None:
            info["gel_edges"] = gel_edges_cnt
    except Exception:
        pass

    if args.format == "json":
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    caps = info.get("caps") or {}
    print(f"Snapshot:      {info['path']}")
    print(f"schema_version:{info['schema_version']}")
    if info.get("graph_schema_version"):
        print(f"graph schema :{info['graph_schema_version']}")
    print(f"version_etag  :{info['version_etag']}")
    print(f"nodes/edges   :{info['nodes']} / {info['edges']}")
    if info.get("gel_nodes") is not None or info.get("gel_edges") is not None:
        print(
            f"gel nodes/edges:{info.get('gel_nodes')} / {info.get('gel_edges')}"
        )
    if (
        info.get("gel_merges") is not None
        or info.get("gel_splits") is not None
        or info.get("gel_promotions") is not None
        or info.get("gel_concepts") is not None
    ):
        print(
            "gel meta      : "
            f"merges={info.get('gel_merges')} "
            f"splits={info.get('gel_splits')} "
            f"promotions={info.get('gel_promotions')} "
            f"concepts={info.get('gel_concepts')}"
        )
    print(f"last_update   :{info['last_update']}")
    print(
        "caps          : "
        f"delta_l2={caps.get('delta_norm_cap_l2')} "
        f"novelty={caps.get('novelty_cap_per_node')} "
        f"churn={caps.get('churn_cap_edges')} "
        f"weights=[{caps.get('weight_min')}, {caps.get('weight_max')}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())