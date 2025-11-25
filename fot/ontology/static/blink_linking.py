from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

from ...utils.logging import get_logger
from ...utils import paths as pathutil
from ...blink_adapter.blink_adapter import run_blink
import os


DEFAULT_OUTPUTS = {
    "blink_all_levels": "blink_candidates_l1l2l3.json",
    "blink_l3_only": "blink_candidates_l3.json",
}


def _resolve_output(dir_key: str, name_or_path: str) -> str:
    # If caller passed a path (contains os.sep), use it as-is; otherwise expand under dir_key
    if os.sep in name_or_path or name_or_path.startswith("./"):
        p = Path(name_or_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    return pathutil.expand(dir_key, name_or_path)


def link_entities(
    floor1: Optional[str],
    floor2: Optional[str],
    floor3: str,
    out_all_levels: Optional[str] = None,
    out_l3_only: Optional[str] = None,
    *,
    entity_jsonl: Optional[str] = None,  # Now optional, not used by original scripts
    mode: str = "both",
    cfg: Optional[Dict] = None,
    dry_run: bool = False,
    fast: bool = False,
) -> Dict[str, Optional[str]]:
    """Unified wrapper for BLINK linking over IPC floors.

    Returns dict with keys 'all_levels' and 'l3_only' pointing to produced files.
    """
    logger = get_logger("blink_linking")
    outputs_cfg = (cfg or {}).get("outputs", {})

    all_levels_name = out_all_levels or outputs_cfg.get("blink_all_levels") or DEFAULT_OUTPUTS["blink_all_levels"]
    l3_only_name = out_l3_only or outputs_cfg.get("blink_l3_only") or DEFAULT_OUTPUTS["blink_l3_only"]

    all_dir_key = "data.interim"
    l3_dir_key = "data.interim"

    out_all = _resolve_output(all_dir_key, all_levels_name)
    out_l3 = _resolve_output(l3_dir_key, l3_only_name)

    floors = {k: v for k, v in {"l1": floor1, "l2": floor2, "l3": floor3}.items() if v}

    results: Dict[str, Optional[str]] = {"all_levels": None, "l3_only": None}

    if mode in ("123", "both") and floor1 and floor2:
        logger.info("Running BLINK (mode=123)")
        run_blink(floors, out_all, mode="123", dry_run=dry_run, fast=fast)
        results["all_levels"] = out_all
        # Assert file exists
        if not Path(out_all).exists():
            raise RuntimeError(
                f"BLINK 123 expected output missing: {out_all}. floors={floors}"
            )

    if mode in ("3", "both"):
        logger.info("Running BLINK (mode=3)")
        run_blink({"l3": floor3}, out_l3, mode="3", dry_run=dry_run, fast=fast)
        results["l3_only"] = out_l3
        if not Path(out_l3).exists():
            raise RuntimeError(
                f"BLINK 3 expected output missing: {out_l3}. floor3={floor3}"
            )

    # Collect minimal stats for observability
    def _count_entity_jsonl(path: str) -> int:
        try:
            return sum(1 for _ in Path(path).read_text(encoding="utf-8").splitlines() if _.strip())
        except Exception:
            return 0

    def _collect_stats(json_path: Optional[str]) -> Dict[str, Any]:
        if not json_path:
            return {}
        try:
            data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        except Exception:
            return {"file_kb": max(1, int((Path(json_path).stat().st_size + 1023) / 1024)) if Path(json_path).exists() else 0}
        # data can be list of blocks each with Texts_Entities
        te_counts: List[int] = []
        for block in data if isinstance(data, list) else []:
            texts = block.get("Texts_Entities", []) if isinstance(block, dict) else []
            for te in texts:
                ce = 0
                ce += len(te.get("Biencoder_Recommended_Entities", []) or [])
                ce += len(te.get("Crossencoder_Recommended_Entities", []) or [])
                te_counts.append(ce)
        avg = (sum(te_counts) / len(te_counts)) if te_counts else 0.0
        med = 0.0
        if te_counts:
            s = sorted(te_counts)
            mid = len(s) // 2
            med = s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2
        size_kb = max(1, int((Path(json_path).stat().st_size + 1023) / 1024)) if Path(json_path).exists() else 0
        return {"texts": len(te_counts), "avg_candidates_per_text": avg, "median_candidates_per_text": med, "file_kb": size_kb}

    stats = {
        "input_entities": _count_entity_jsonl(entity_jsonl) if entity_jsonl else 0,
        "mode": mode,
        "floors_provided": list(floors.keys()),
        "l123": _collect_stats(results.get("all_levels")),
        "l3": _collect_stats(results.get("l3_only")),
    }
    results["stats"] = stats  # non-breaking: tests only assert files exist
    return results
