from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from ...utils import paths as pathutil
from ...utils.logging import get_logger
from . import blink_linking
from ...data.cleaning import filter_and_augment
from . import gat_classifier
from . import semantic_parent_linker
from ...data.cleaning import dedupe as dedupe_mod


RENAME_DEFAULTS = [
    ("ipc_entities_4.json", "blink_candidates_l1l2l3.json", "Unified naming for all-level BLINK candidates"),
    ("new_ipc_entities_level_3.json", "blink_candidates_l3.json", "Unified naming for L3 BLINK candidates"),
    ("try_ipc_entity_level2.json", "level2_seeds.json", "Filtered L2 seeds after augment"),
    ("predicted_entities.json", "gat_l3_predictions.json", "GAT predictions for L3"),
    ("new_predicted_entities.json", "l3_with_parents.json", "L3 entities with assigned parents"),
    ("merged_entities.json", "l3_dedup.json", "Deduplicated L3 entities"),
    ("best_model_0.pth", "gat_best.pth", "Best GAT model weights"),
]


def _write_rename_map(doc_path: str) -> str:
    Path(doc_path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["old_name,new_name,reason"]
    for old, new, reason in RENAME_DEFAULTS:
        lines.append(f"{old},{new},{reason}")
    Path(doc_path).write_text("\n".join(lines), encoding="utf-8")
    return doc_path


from ...utils.runinfo import new_run_id


def run(
    cfg: Dict[str, Any], *, dry_run: bool = False, fast: bool = False, skip_blink: bool = False, generate_rename_map: bool = False, run_id: str | None = None
) -> Dict[str, str]:
    logger = get_logger("build_static")
    run_id = run_id or new_run_id()

    outputs = cfg.get("outputs", {})

    # Resolve output paths
    interim = "data.interim"
    processed = "data.processed"
    models_dir = "artifacts.models"

    blink_all = pathutil.expand(interim, outputs.get("blink_all_levels", "blink_candidates_l1l2l3.json"))
    blink_l3 = pathutil.expand(interim, outputs.get("blink_l3_only", "blink_candidates_l3.json"))
    level2_seeds = pathutil.expand(interim, outputs.get("level2_seeds", "level2_seeds.json"))
    gat_pred = pathutil.expand(processed, outputs.get("gat_pred", "gat_l3_predictions.json"))
    l3_with_parents = pathutil.expand(processed, outputs.get("l3_with_parents", "l3_with_parents.json"))
    l3_dedup = pathutil.expand(processed, outputs.get("l3_dedup", "l3_dedup.json"))
    gat_model = pathutil.expand(models_dir, outputs.get("gat_model", "gat_best.json"))

    # Step 1: BLINK linking
    blink_mode = cfg.get("blink", {}).get("mode", "both")
    floors = cfg.get("floors", {}) or {}
    entity_jsonl = cfg.get("entity_jsonl")
    # Fallback to fixtures when dry_run and not provided
    if dry_run:
        pr = Path(__file__).resolve().parents[3]  # project root
        fixtures = pr / "tests" / "fixtures" / "stage1"
        if not entity_jsonl:
            entity_jsonl = str((fixtures / "entity.jsonl").resolve())
        floors.setdefault("l1", str((fixtures / "floor_1.csv").resolve()))
        floors.setdefault("l2", str((fixtures / "floor_2.csv").resolve()))
        floors.setdefault("l3", str((fixtures / "floor_3.csv").resolve()))
    # If mode=both but missing L1/L2, warn and run only 3
    if blink_mode == "both" and not (floors.get("l1") and floors.get("l2")):
        logger.warning("blink.mode=both, but L1/L2 floors missing. Running mode='3' only.")
        local_mode = "3"
    else:
        local_mode = blink_mode

    # 检查是否可以跳过BLINK
    should_skip_blink = False
    if skip_blink or cfg.get("blink", {}).get("skip_if_exists", False):
        # 检查输出文件是否存在
        if Path(blink_all).exists() and Path(blink_l3).exists():
            try:
                # 验证文件有效性
                with open(blink_all, 'r', encoding='utf-8') as f:
                    data_all = json.load(f)
                with open(blink_l3, 'r', encoding='utf-8') as f:
                    data_l3 = json.load(f)

                # 检查是否有数据
                if data_all and data_l3:
                    should_skip_blink = True
                    logger.info("BLINK output files found and valid, skipping BLINK entity linking")
                    logger.info("  - %s: %d records", blink_all, len(data_all))
                    logger.info("  - %s: %d records", blink_l3, len(data_l3))
            except Exception as e:
                logger.warning("BLINK output files exist but invalid: %s", e)
                logger.info("Will regenerate BLINK outputs")

    if should_skip_blink:
        # 创建虚拟的blink_res以保持后续流程兼容
        blink_res = {
            "all_levels": blink_all,
            "l3_only": blink_l3,
            "stats": {
                "skipped": True,
                "reason": "using existing output files",
                "l123_count": len(data_all) if 'data_all' in locals() else 0,
                "l3_count": len(data_l3) if 'data_l3' in locals() else 0
            }
        }
    else:
        # 原有的BLINK执行逻辑（保持不变）
        blink_res = blink_linking.link_entities(
            floors.get("l1"),
            floors.get("l2"),
            floors.get("l3"),
            out_all_levels=blink_all,
            entity_jsonl=entity_jsonl,
            out_l3_only=blink_l3,
            mode=local_mode,
            cfg=cfg,
            dry_run=dry_run,
            fast=fast,
        )

    # Step 2: Filter & augment (L2 seeds)
    filter_cfg = cfg.get("filter_augment", {})
    src_for_filter = blink_all if blink_res.get("all_levels") else blink_l3
    fa_stats = filter_and_augment.filter_entities(
        src_for_filter,
        entity_jsonl,
        level2_seeds,
        rules={
            "include_rules": filter_cfg.get("include_rules", {}),
            "exclude_rules": filter_cfg.get("exclude_rules", {}),
            "manual_whitelist": filter_cfg.get("manual_whitelist", []),
            "manual_blacklist": filter_cfg.get("manual_blacklist", []),
        },
        mode=filter_cfg.get("mode", "index"),
        indexes_file=filter_cfg.get("indexes_file"),
    )
    logger.info("Filter/Augment stats: %s", fa_stats)

    # Step 3: GAT classifier
    gat_cfg = cfg.get("gat", {})
    model_path, pred_path = gat_classifier.run(
        level2_seeds,
        out_model_path=gat_model,
        out_pred_path=gat_pred,
        mode=gat_cfg.get("mode", "mlp"),
        entity_catalog_path=entity_jsonl,
        mag_file_path=gat_cfg.get("mag_file_path"),
        level3_candidates_path=gat_cfg.get("level3_candidates_path"),
        encoder=gat_cfg.get("encoder", "random"),
        dry_run=dry_run,
        fast=fast,
        run_id=run_id,
        gat_config=gat_cfg.get("gat_config"),
        network_config=gat_cfg.get("network"),
    )
    # Summaries
    def _count_json(path: str) -> int:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return len(data) if isinstance(data, list) else 0
        except Exception:
            return 0

    logger.info(
        "GAT artifacts: model=%s pred=%s count=%d",
        model_path,
        pred_path,
        _count_json(pred_path),
    )

    # Step 4: Parent assignment
    parent_cfg = cfg.get("parent_linker", {})
    parent_stats = semantic_parent_linker.run(gat_pred, level2_seeds, l3_with_parents, config=parent_cfg)
    parent_out = parent_stats.get("out_path", l3_with_parents)
    logger.info("Parent assigned count=%d", _count_json(parent_out))

    # Step 5: Deduplicate
    dedupe_cfg = cfg.get("dedupe", {})
    dedupe_mode = dedupe_cfg.get("mode", "simple")
    dedupe_threshold = dedupe_cfg.get("threshold", 0.6)

    if dry_run:
        # Always use simple dedupe in dry-run mode
        from ...data.cleaning.dedupe import dedupe_entities as _ded
        _ded(l3_with_parents, l3_dedup)
        dedupe_stats = {"input": _count_json(l3_with_parents), "output": _count_json(l3_dedup)}
        logger.info("Using simple deduplication (dry-run mode)")
    elif dedupe_mode == "simple":
        # Use original simple dedupe (only merge exact name matches)
        from ...data.cleaning.dedupe import dedupe_entities
        dedupe_entities(l3_with_parents, l3_dedup)
        dedupe_stats = {"input": _count_json(l3_with_parents), "output": _count_json(l3_dedup)}
        logger.info("Using simple deduplication (original mode)")
    else:
        # Use advanced parent-aware dedupe with similarity matching
        dedupe_stats = dedupe_mod.run(l3_with_parents, l3_dedup, threshold=dedupe_threshold)
        logger.info("Using advanced deduplication (threshold=%.2f)", dedupe_threshold)

    logger.info("Deduped count=%d", _count_json(l3_dedup))

    # Merge L2 and L3 into single JSON
    try:
        level2_list = json.loads(Path(level2_seeds).read_text(encoding="utf-8"))
        level3_list = json.loads(Path(l3_dedup).read_text(encoding="utf-8"))
    except Exception:
        level2_list, level3_list = [], []
    merged = {"level2": level2_list, "level3": level3_list}
    merged_path = pathutil.expand(processed, outputs.get("merged_entities", "merged_entities.json"))
    Path(merged_path).write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    if generate_rename_map:
        doc = pathutil.expand("reports", "ARTIFACTS_RENAME_MAP.md")
        _write_rename_map(doc)

    def _size_kb(path: str) -> int:
        try:
            return max(1, int((Path(path).stat().st_size + 1023) / 1024))
        except Exception:
            return 0

    result = {
        "blink_all_levels": blink_all,
        "blink_l3_only": blink_l3,
        "level2_seeds": level2_seeds,
        "gat_model": gat_model,
        "gat_pred": gat_pred,
        "l3_with_parents": l3_with_parents,
        "l3_dedup": l3_dedup,
        "merged_entities": merged_path,
    }
    # Attach sizes
    result_with_sizes = {k: v for k, v in result.items()}
    for k, v in result.items():
        result_with_sizes[f"{k}_kb"] = _size_kb(v)
    result_with_sizes["run_id"] = run_id

    # Write Stage 1 summary for observability
    try:
        summary = {
            "run_id": run_id,
            "blink": blink_res.get("stats", {}),
            "filter_augment": fa_stats,
            "gat": {
                "model_path": model_path,
                "pred_path": pred_path,
                "metrics_path": str(Path("reports") / f"stage1_gat_metrics_{run_id}.json"),
                "predicted_count": _count_json(gat_pred),
            },
            "parent_link": {
                "groups": {},
                "assigned": _count_json(parent_out),
                "linked_count": int(parent_stats.get("linked_count", 0)),
            },
            "dedupe": {
                "input_count": int(dedupe_stats.get("input", _count_json(l3_with_parents))),
                "output_count": int(dedupe_stats.get("output", _count_json(l3_dedup))),
            },
        }
        # parent groups distribution from parent_stats
        if isinstance(parent_stats.get("groups"), dict):
            summary["parent_link"]["groups"] = parent_stats["groups"]
        s_path = Path(pathutil.expand("reports", f"stage1_summary_{run_id}.json"))
        s_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to write Stage1 summary")

    return result_with_sizes
