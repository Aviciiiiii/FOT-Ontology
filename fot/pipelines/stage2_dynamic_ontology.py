from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from ..utils.logging import get_logger
from ..utils.runinfo import new_run_id
from ..utils import paths as pathutil
from ..ontology.dynamic.dataset_builder import run as build_dataset
from ..models.ner.train import run_pretrain
from ..models.ner.finetune import run_finetune


def run(cfg: Dict[str, Any], *, dry_run: bool = True, fast: bool = True, run_id: str | None = None) -> Dict[str, str]:
    logger = get_logger("stage2_dynamic")
    run_id = run_id or new_run_id()
    outputs = cfg.get("outputs", {})

    # CRITICAL: Force balanced sampling in real mode for proper F1 performance
    if not dry_run:
        original_fast = fast
        fast = False  # Disable fast mode to ensure balanced sampling
        if original_fast:
            logger.info("Real training mode - disabled fast mode to ensure balanced sampling (critical for F1 > 0.6)")
        else:
            logger.info("Real training mode - balanced sampling enabled (critical for F1 > 0.6)")

    # Build datasets
    ds = build_dataset(cfg, dry_run=dry_run, fast=fast)

    models_dir = "artifacts.models"
    pretrain_out = pathutil.expand(models_dir, outputs.get("ner_pretrain_model", "NER_pretrain_stub.json"))
    finetune_out = pathutil.expand(models_dir, outputs.get("ner_finetune_model", "NER_finetune_stub.json"))

    # Check if model files exist
    # CRITICAL FIX: Handle multiple extension cases properly
    # Examples:
    #   "NER_pretrain.json" → "NER_pretrain_best.pth"
    #   "NER_pretrain.pth"  → "NER_pretrain_best.pth"
    #   "NER_pretrain_best.pth" → "NER_pretrain_best.pth" (no change)
    #   "NER_pretrain"      → "NER_pretrain_best.pth"

    def get_best_model_path(base_path: str) -> Path:
        """Convert config output path to actual best model path."""
        if base_path.endswith('_best.pth'):
            # Already in correct format
            return Path(base_path)
        elif base_path.endswith('.json'):
            # Replace .json with _best.pth
            return Path(base_path.replace('.json', '_best.pth'))
        elif base_path.endswith('.pth'):
            # Insert _best before .pth
            return Path(base_path.replace('.pth', '_best.pth'))
        else:
            # No extension, append _best.pth
            return Path(base_path + '_best.pth')

    pretrain_model_path = get_best_model_path(pretrain_out)
    finetune_model_path = get_best_model_path(finetune_out)

    # Run NER pretrain (skip if model already exists)
    if pretrain_model_path.exists():
        logger.info(f"✓ Pretrain model already exists: {pretrain_model_path}")
        logger.info(f"  Skipping pretraining step...")
        pre_model = str(pretrain_model_path)
    else:
        logger.info(f"✗ Pretrain model not found: {pretrain_model_path}")
        logger.info(f"  Running pretraining...")
        pre_model = run_pretrain(ds["mag1_clean"], ds["mag2_clean"], ds["mag_entities"], pretrain_out, dry_run=dry_run, fast=fast, run_id=run_id)

    # Run NER finetune (skip if model already exists)
    if finetune_model_path.exists():
        logger.info(f"✓ Finetune model already exists: {finetune_model_path}")
        logger.info(f"  Skipping fine-tuning step...")
        fin_model = str(finetune_model_path)
    else:
        logger.info(f"✗ Finetune model not found: {finetune_model_path}")
        logger.info(f"  Running fine-tuning...")
        fin_model = run_finetune(ds["fot1_clean"], ds["fot2_clean"], ds["mag_entities"], ds["third_entities"], pre_model, finetune_out, dry_run=dry_run, fast=fast, run_id=run_id)

    # Summaries
    def _count(path: str) -> int:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return len(data) if isinstance(data, list) else 0
        except Exception:
            return 0

    logger.info(
        "NER stats | MAG cleaned: %d,%d | FOT cleaned: %d,%d",
        _count(ds["mag1_clean"]),
        _count(ds["mag2_clean"]),
        _count(ds["fot1_clean"]),
        _count(ds["fot2_clean"]),
    )

    # Write stage2 summary (non-intrusive)
    try:
        import json as _json
        from pathlib import Path as _Path
        def _len_json(p: str) -> int:
            try:
                return len(_json.loads(_Path(p).read_text(encoding="utf-8")))
            except Exception:
                return 0
        summary = {
            "run_id": run_id,
            "search": {
                "mag": {"input": _len_json(ds["mag_entities"]), "kept": _len_json(ds["mag1_tagged"]) + _len_json(ds["mag2_tagged"])},
                "fot": {"input": _len_json(ds["third_entities"]), "kept": _len_json(ds["fot1_tagged"]) + _len_json(ds["fot2_tagged"])},
            },
            "clean": {
                "mag": {"input": _len_json(ds["mag1_tagged"]) + _len_json(ds["mag2_tagged"]), "output": _len_json(ds["mag1_clean"]) + _len_json(ds["mag2_clean"])},
                "fot": {"input": _len_json(ds["fot1_tagged"]) + _len_json(ds["fot2_tagged"]), "output": _len_json(ds["fot1_clean"]) + _len_json(ds["fot2_clean"])},
            },
            "ner": {
                "pretrain": {"model": pre_model},
                "finetune": {"model": fin_model},
            },
        }
        s_path = _Path(pathutil.expand("reports", f"stage2_summary_{run_id}.json"))
        s_path.write_text(_json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to write Stage2 summary")

    result = {**ds, "ner_pretrain_model": pre_model, "ner_finetune_model": fin_model, "run_id": run_id}
    return result
