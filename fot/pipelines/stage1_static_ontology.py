from __future__ import annotations

from typing import Dict, Any

from ..utils.logging import get_logger
from ..utils.runinfo import new_run_id
from ..ontology.static.build_static import run as build_static_run


def run(cfg: Dict[str, Any], *, dry_run: bool = False, fast: bool = False, skip_blink: bool = False, generate_rename_map: bool = False, run_id: str | None = None) -> Dict[str, str]:
    """
    Stage 1: Static Ontology Construction Pipeline

    Args:
        cfg: Configuration dictionary from ontology_static.yaml
        dry_run: Use fixtures instead of real data
        fast: Reduce work for smoke testing
        skip_blink: Skip BLINK entity linking if output files exist
        generate_rename_map: Generate docs/ARTIFACTS_RENAME_MAP.md
        run_id: Optional run identifier for tracking

    Returns:
        Dictionary of output file paths and metadata
    """
    logger = get_logger("stage1_static")
    run_id = run_id or new_run_id()

    logger.info("Running Stage 1 (static ontology) with dry_run=%s fast=%s skip_blink=%s", dry_run, fast, skip_blink)

    # Delegate to build_static.run()
    result = build_static_run(
        cfg,
        dry_run=dry_run,
        fast=fast,
        skip_blink=skip_blink,
        generate_rename_map=generate_rename_map,
        run_id=run_id
    )

    logger.info("Stage 1 complete. Artifacts: %s", result)
    return result
