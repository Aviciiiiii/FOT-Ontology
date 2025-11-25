from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from ...utils.logging import get_logger
from ...utils import paths as pathutil
from ...data.prepare.mag_entities import build_mag_entities
from ...data.prepare.third_entities import build_third_entities
from ...search.mag_search import run_mag_search
from ...search.fot_search import run_fot_search
from ...data.cleaning.mag_cleaner import clean_mag
from ...data.cleaning.fot_cleaner import clean_fot


def run(cfg: Dict[str, Any], *, dry_run: bool = False, fast: bool = True) -> Dict[str, str]:
    logger = get_logger("dataset_builder")
    outputs = cfg.get("outputs", {})

    interim = "data.interim"
    processed = "data.processed"

    # Resolve paths
    mag_entities = pathutil.expand(interim, outputs.get("mag_entities", "mag_entities.json"))
    third_entities = pathutil.expand(interim, outputs.get("third_entities", "third_entities.json"))

    mag1_tagged = pathutil.expand(interim, outputs.get("mag1_tagged", "mag1_tagged_searched_sentences_with_entity.json"))
    mag2_tagged = pathutil.expand(interim, outputs.get("mag2_tagged", "mag2_tagged_searched_sentences_with_entity.json"))
    fot1_tagged = pathutil.expand(interim, outputs.get("fot1_tagged", "FOT1_tagged_searched_sentences_with_entity.json"))
    fot2_tagged = pathutil.expand(interim, outputs.get("fot2_tagged", "FOT2_tagged_searched_sentences_with_entity.json"))

    mag1_clean = pathutil.expand(processed, outputs.get("mag1_clean", "cleaned_mag1_tagged_searched_sentences.json"))
    mag2_clean = pathutil.expand(processed, outputs.get("mag2_clean", "cleaned_mag2_tagged_searched_sentences.json"))
    fot1_clean = pathutil.expand(processed, outputs.get("fot1_clean", "cleaned_FOT1_tagged_searched_sentences.json"))
    fot2_clean = pathutil.expand(processed, outputs.get("fot2_clean", "cleaned_FOT2_tagged_searched_sentences.json"))

    # Prepare entity lists
    build_mag_entities(cfg.get("mag_nt_path", ""), mag_entities, dry_run=dry_run)
    build_third_entities(cfg.get("merged_entities_path", ""), third_entities, dry_run=dry_run)

    # Search backends - skip if files already exist
    search_cfg = cfg.get("search", {}) or {}
    backend = search_cfg.get("backend", "dryrun") if not dry_run else "dryrun"

    # Check if search results already exist
    mag_search_exists = all(Path(f).exists() for f in [mag1_tagged, mag2_tagged])
    fot_search_exists = all(Path(f).exists() for f in [fot1_tagged, fot2_tagged])

    if mag_search_exists:
        logger.info("MAG search files already exist, skipping MAG search: %s, %s", mag1_tagged, mag2_tagged)
    else:
        logger.info("Running MAG search to generate: %s, %s", mag1_tagged, mag2_tagged)
        run_mag_search(mag_entities, mag1_tagged, mag2_tagged, dry_run=dry_run, fast=fast, backend=backend, cfg=search_cfg.get("mag", {}))

    if fot_search_exists:
        logger.info("FOT search files already exist, skipping FOT search: %s, %s", fot1_tagged, fot2_tagged)
    else:
        logger.info("Running FOT search to generate: %s, %s", fot1_tagged, fot2_tagged)
        run_fot_search("", third_entities, fot1_tagged, fot2_tagged, dry_run=dry_run, fast=fast, backend=backend, cfg=search_cfg.get("fot", {}))

    # Clean - skip if cleaned files already exist
    mag_clean_exists = all(Path(f).exists() for f in [mag1_clean, mag2_clean])
    fot_clean_exists = all(Path(f).exists() for f in [fot1_clean, fot2_clean])

    if mag_clean_exists:
        logger.info("MAG cleaned files already exist, skipping MAG cleaning: %s, %s", mag1_clean, mag2_clean)
    else:
        logger.info("Running MAG cleaning to generate: %s, %s", mag1_clean, mag2_clean)
        clean_mag(mag1_tagged, mag2_tagged, mag_entities, mag1_clean, mag2_clean)

    if fot_clean_exists:
        logger.info("FOT cleaned files already exist, skipping FOT cleaning: %s, %s", fot1_clean, fot2_clean)
    else:
        logger.info("Running FOT cleaning to generate: %s, %s", fot1_clean, fot2_clean)
        clean_fot(fot1_tagged, fot2_tagged, mag_entities, third_entities, fot1_clean, fot2_clean)

    # Log counts
    def _count(path: str) -> int:
        """Count samples in JSON or JSONL format files.

        Supports both:
        - JSON array format: [{"tokens":[...],"tags":[...]}, ...]
        - JSONL format: {"tokens":[...],"tags":[...]}\n{"tokens":[...],"tags":[...]}\n...
        """
        try:
            # Try JSON array format first
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return len(data) if isinstance(data, list) else 0
        except json.JSONDecodeError:
            # Try JSONL format (one JSON object per line)
            try:
                count = 0
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            count += 1
                return count
            except Exception:
                return 0
        except Exception:
            return 0

    logger.info(
        "Counts | MAG tagged: %d,%d cleaned: %d,%d | FOT tagged: %d,%d cleaned: %d,%d",
        _count(mag1_tagged),
        _count(mag2_tagged),
        _count(mag1_clean),
        _count(mag2_clean),
        _count(fot1_tagged),
        _count(fot2_tagged),
        _count(fot1_clean),
        _count(fot2_clean),
    )

    return {
        "mag_entities": mag_entities,
        "third_entities": third_entities,
        "mag1_tagged": mag1_tagged,
        "mag2_tagged": mag2_tagged,
        "fot1_tagged": fot1_tagged,
        "fot2_tagged": fot2_tagged,
        "mag1_clean": mag1_clean,
        "mag2_clean": mag2_clean,
        "fot1_clean": fot1_clean,
        "fot2_clean": fot2_clean,
    }
