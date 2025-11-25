from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ...utils.logging import get_logger


def dedupe_entities(in_path: str, out_path: str) -> str:
    """Merge entities with the same name, aggregating ent_ipc values into a unique list."""
    logger = get_logger("dedupe")
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    merged: Dict[str, Dict[str, Any]] = {}
    for item in data:
        name = item.get("name") or item.get("Entity_Title")
        if not name:
            continue
        entry = merged.setdefault(name, {**item, "ent_ipc": []})
        ent_ipc_val = item.get("ent_ipc") or item.get("IPC_Classification")
        if ent_ipc_val is None:
            continue
        if isinstance(ent_ipc_val, list):
            entry["ent_ipc"].extend(ent_ipc_val)
        else:
            entry["ent_ipc"].append(ent_ipc_val)

    for v in merged.values():
        v["ent_ipc"] = sorted(list(set(v.get("ent_ipc", []))))

    out_list: List[Dict[str, Any]] = list(merged.values())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    logger.info("Deduped %d -> %d (%s)", len(data), len(out_list), out_path)
    return out_path


def _encode(text: str, dim: int = 128) -> List[float]:
    vec = [0.0] * dim
    if not text:
        return vec
    for ch in text.lower():
        idx = (ord(ch) * 131) % dim
        vec[idx] += 1.0
    # L2 normalize
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return float(num / (da * db))


def run(
    in_json: str,
    out_json: str,
    *,
    name_key: str = "name",
    parent_key: str = "parent",
    sim_key: str = "sim",
    threshold: float = 0.6,
) -> Dict[str, int]:
    """Dedupe L3 with parent-aware key and near-duplicate merge.

    - Group by (name.lower(), parent.ent_ipc).
    - Within same parent group, greedily merge entries with cosine(name, other_name) >= threshold.
    - Keep the highest score/sim as representative; aggregate aliases.
    """
    logger = get_logger("dedupe")
    with open(in_json, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    # Step 1: exact key dedupe
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for e in items:
        name = (e.get(name_key) or "").strip().lower()
        parent = e.get(parent_key) or {}
        parent_ipc = (parent.get("ent_ipc") if isinstance(parent, dict) else "") or ""
        groups.setdefault((name, parent_ipc), []).append(e)

    merged_exact: List[Dict[str, Any]] = []
    for (_name, _ipc), arr in groups.items():
        if not arr:
            continue
        # select best by score then sim
        arr_sorted = sorted(arr, key=lambda x: (float(x.get("score", 0.0)), float(x.get(sim_key, 0.0))), reverse=True)
        rep = dict(arr_sorted[0])
        aliases = [a.get(name_key) for a in arr_sorted[1:] if a.get(name_key) and a.get(name_key) != rep.get(name_key)]
        if aliases:
            rep["aliases"] = sorted(set([rep.get(name_key)] + aliases))
        merged_exact.append(rep)

    # Step 2: near-duplicate merge within same parent_ipc
    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for e in merged_exact:
        parent = e.get(parent_key) or {}
        parent_ipc = (parent.get("ent_ipc") if isinstance(parent, dict) else "") or ""
        by_parent.setdefault(parent_ipc, []).append(e)

    final_list: List[Dict[str, Any]] = []
    for parent_ipc, arr in by_parent.items():
        # sort by score then sim
        arr_sorted = sorted(arr, key=lambda x: (float(x.get("score", 0.0)), float(x.get(sim_key, 0.0))), reverse=True)
        used = [False] * len(arr_sorted)
        enc_cache = [_encode((e.get(name_key) or "")) for e in arr_sorted]
        for i, e in enumerate(arr_sorted):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            for j in range(i + 1, len(arr_sorted)):
                if used[j]:
                    continue
                sim = _cosine(enc_cache[i], enc_cache[j])
                if sim >= threshold:
                    used[j] = True
                    cluster.append(j)
            if len(cluster) == 1:
                final_list.append(e)
            else:
                # choose best rep (already sorted), aggregate aliases
                rep = dict(arr_sorted[cluster[0]])
                alias_names = []
                for k in cluster[1:]:
                    name = arr_sorted[k].get(name_key)
                    if name and name != rep.get(name_key):
                        alias_names.append(name)
                if alias_names:
                    rep.setdefault("aliases", [])
                    rep["aliases"] = sorted(set((rep.get("aliases") or []) + alias_names))
                final_list.append(rep)

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    logger.info("Dedupe (parent-aware) %d -> %d (%s)", len(items), len(final_list), out_json)
    return {"input": len(items), "output": len(final_list)}
