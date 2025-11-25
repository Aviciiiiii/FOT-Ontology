from __future__ import annotations

import json
import re
from typing import Dict, List, Any, Optional

from ...utils.logging import get_logger
from pathlib import Path
from ...utils.paths import _CONFIG_DIR


def _matches(entity: Dict[str, Any], patterns: Dict[str, str]) -> bool:
    for key, pat in patterns.items():
        val = entity.get(key)
        if val is None:
            return False
        if not re.search(pat, str(val)):
            return False
    return True


def filter_entities(
    ipc_candidates_path: str,
    entity_jsonl_path: str,
    out_level2_path: str,
    *,
    rules: Dict[str, Any],
    out_level1_path: Optional[str] = None,
    mode: str = "index",
    indexes_file: Optional[str] = None,
) -> Dict[str, int]:
    """Filter and optionally augment IPC candidate entities.

    Supports two modes:
    - "rules": Use include/exclude rules and manual whitelist/blacklist
    - "index": Use hardcoded indexes from original script logic
    """
    logger = get_logger("filter_and_augment")
    logger.info("Filter mode: %s", mode)

    with open(ipc_candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    raw_count = len(candidates) if isinstance(candidates, list) else 0

    # Load entity catalog for lookups (needed for both modes)
    title2id, id2title, id2text = _load_entity_catalog(entity_jsonl_path) if entity_jsonl_path else ({}, {}, {})

    if mode == "index":
        out = _filter_entities_by_index(candidates, indexes_file, title2id, id2title, id2text, logger)
    else:
        out = _filter_entities_by_rules(candidates, rules, logger)

    # Add additional entities from manual_whitelist (supporting both modes)
    whitelist: List[str] = rules.get("manual_whitelist", []) or []
    if whitelist:
        out = _add_additional_entities(out, whitelist, title2id, id2text, logger)

    # Write outputs
    out_level2_path = str(out_level2_path)
    Path(out_level2_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_level2_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(
        "Wrote level2 seeds to %s (%d items) [raw_candidates=%d, flattened=%d]",
        out_level2_path,
        len(out),
        raw_count,
        len(candidates),
    )

    # Optionally write level1 file (not populated in scaffold)
    if out_level1_path:
        Path(out_level1_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_level1_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    return {"raw_candidates": raw_count, "flattened": len(candidates), "output_level2": len(out)}


def _load_entity_catalog(entity_jsonl_path: str) -> tuple[Dict[str, int], Dict[int, str], Dict[int, str]]:
    """Load Wikipedia entity catalog for entity lookups."""
    title2id = {}
    id2title = {}
    id2text = {}

    if not entity_jsonl_path or not Path(entity_jsonl_path).exists():
        return title2id, id2title, id2text

    with open(entity_jsonl_path, "r", encoding="utf-8") as f:
        local_idx = 0
        for line in f:
            if not line.strip():
                continue
            entity = json.loads(line)

            title = entity.get("title", "")
            text = entity.get("text", "")

            title2id[title] = local_idx
            id2title[local_idx] = title
            id2text[local_idx] = text
            local_idx += 1

    return title2id, id2title, id2text


def _filter_entities_by_index(
    candidates: List[Dict[str, Any]],
    indexes_file: Optional[str],
    title2id: Dict[str, int],
    id2title: Dict[int, str],
    id2text: Dict[int, str],
    logger
) -> List[Dict[str, Any]]:
    """Filter entities using hardcoded indexes from original script."""

    # Load indexes_to_keep
    if indexes_file:
        indexes_path = _CONFIG_DIR / indexes_file
    else:
        indexes_path = _CONFIG_DIR / "indexes_to_keep.json"

    with open(indexes_path, "r", encoding="utf-8") as f:
        indexes_to_keep = json.load(f)

    logger.info("Loaded %d index groups from %s", len(indexes_to_keep), indexes_path)

    ipc_entities_final = []
    id_selected = {}  # Track selected entities for deduplication

    for ipc_index, text_indexes in enumerate(indexes_to_keep):
        if ipc_index >= len(candidates):
            break

        ipc_entry = candidates[ipc_index]
        if not isinstance(ipc_entry, dict) or "Texts_Entities" not in ipc_entry:
            continue

        texts_entities = ipc_entry["Texts_Entities"]

        for text_index, index_info in enumerate(text_indexes):
            if text_index >= len(texts_entities):
                continue

            # Parse index_info: [biencoder_indexes, crossencoder_indexes, parent_ent_id]
            biencoder_indexes = [i - 1 for i in index_info[0] if i > 0]  # Convert to 0-based
            crossencoder_indexes = [i - 1 for i in index_info[1] if i > 0]  # Convert to 0-based
            parent_ent_id = index_info[2] if len(index_info) > 2 else None

            # Get entity lists
            biencoder_entities = texts_entities[text_index].get("Biencoder_Recommended_Entities", [])
            crossencoder_entities = texts_entities[text_index].get("Crossencoder_Recommended_Entities", [])

            # Process Biencoder entities
            for i in biencoder_indexes:
                if i < len(biencoder_entities):
                    entity = biencoder_entities[i]
                    ent_id = entity.get("Entity_ID")
                    if ent_id is not None and ent_id not in id_selected:
                        id_selected[ent_id] = True
                        ipc_entities_final.append(_create_entity_dict(entity, ipc_entry, parent_ent_id, id2title, id2text))

            # Process Crossencoder entities
            for i in crossencoder_indexes:
                if i < len(crossencoder_entities):
                    entity = crossencoder_entities[i]
                    ent_id = entity.get("Entity_ID")
                    if ent_id is not None and ent_id not in id_selected:
                        id_selected[ent_id] = True
                        ipc_entities_final.append(_create_entity_dict(entity, ipc_entry, parent_ent_id, id2title, id2text))

    logger.info("Selected %d entities using index mode", len(ipc_entities_final))
    return ipc_entities_final


def _filter_entities_by_rules(candidates: List[Dict[str, Any]], rules: Dict[str, Any], logger) -> List[Dict[str, Any]]:
    """Filter entities using include/exclude rules (original logic)."""

    # Flatten BLINK-style nested output if present
    flat: List[Dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, dict) and "Texts_Entities" in item:
            ipc = item.get("IPC_Classification", "")
            lvl = item.get("level", 2)
            for te in item.get("Texts_Entities", []):
                for ent in te.get("Biencoder_Recommended_Entities", []) + te.get(
                    "Crossencoder_Recommended_Entities", []
                ):
                    flat.append(
                        {
                            "id": ent.get("Entity_ID", -1),
                            "name": ent.get("Entity_Title", ent.get("name", "")),
                            "text": ent.get("Entity_Text", ent.get("text", "")),
                            "url": ent.get("Entity_URL", ent.get("url", "")),
                            "level": lvl,
                            "ent_ipc": ipc,
                            "parent_ent": item.get("parent_ent"),
                        }
                    )
        else:
            flat.append(item)

    include_rules = rules.get("include_rules", {}) or {}
    exclude_rules = rules.get("exclude_rules", {}) or {}
    blacklist: List[str] = rules.get("manual_blacklist", []) or []

    out: List[Dict[str, Any]] = []
    for item in flat:
        name = item.get("name") or item.get("Entity_Title") or ""
        if name in blacklist:
            continue
        if include_rules and not _matches(item, include_rules):
            continue
        if exclude_rules and _matches(item, exclude_rules):
            continue
        out.append(
            {
                "id": item.get("id", item.get("Entity_ID", -1)),
                "name": item.get("name", item.get("Entity_Title", "")),
                "text": item.get("text", item.get("Entity_Text", "")),
                "url": item.get("url", item.get("Entity_URL", "")),
                "level": item.get("level", 2),
                "ent_ipc": item.get("ent_ipc", item.get("IPC_Classification", "")),
                "parent_ent": item.get("parent_ent"),
            }
        )

    logger.info("Selected %d entities using rules mode (from %d flattened)", len(out), len(flat))
    return out


def _create_entity_dict(entity: Dict[str, Any], ipc_entry: Dict[str, Any], parent_ent_id: Optional[int], id2title: Dict[int, str], id2text: Dict[int, str]) -> Dict[str, Any]:
    """Create entity dictionary in the expected output format."""
    ent_id = entity.get("Entity_ID", -1)
    return {
        "id": ent_id,
        "name": entity.get("Entity_Title", entity.get("Entity_Name", id2title.get(ent_id, ""))),
        "text": entity.get("Entity_Text", id2text.get(ent_id, "")),
        "url": entity.get("Entity_URL", ""),
        "level": ipc_entry.get("level", 1),
        "ent_ipc": ipc_entry.get("IPC_Classification", ""),
        "parent_ent": id2title.get(parent_ent_id) if parent_ent_id else None
    }


def _add_additional_entities(
    entities: List[Dict[str, Any]],
    additional_entity_names: List[str],
    title2id: Dict[str, int],
    id2text: Dict[int, str],
    logger
) -> List[Dict[str, Any]]:
    """Add additional entities from whitelist, supporting IPC format like 'bread machine[A21]'."""
    import re

    # Pattern to parse "entity_name[IPC_CODE]" format
    pattern = re.compile(r"(.+)\[(\w+)\]")
    not_found_titles = []

    for entity_info in additional_entity_names:
        match = pattern.match(entity_info)
        if match:
            entity_name = match.group(1).strip().capitalize()
            ent_ipc = match.group(2).strip()

            if entity_name in title2id:
                ent_id = title2id[entity_name]
                entities.append({
                    "id": ent_id,
                    "name": entity_name,
                    "text": id2text.get(ent_id, "No text available"),
                    "url": f"https://en.wikipedia.org/wiki?curid={ent_id}",
                    "level": 1,
                    "ent_ipc": ent_ipc,
                    "parent_ent": None
                })
            else:
                not_found_titles.append(entity_name)
        else:
            # Simple name without IPC code
            entity_name = entity_info.strip()
            if entity_name in title2id:
                ent_id = title2id[entity_name]
                entities.append({
                    "id": ent_id,
                    "name": entity_name,
                    "text": id2text.get(ent_id, "No text available"),
                    "url": f"https://en.wikipedia.org/wiki?curid={ent_id}",
                    "level": 2,
                    "ent_ipc": "",
                    "parent_ent": None
                })
            else:
                # Add with placeholder ID if not found
                entities.append({
                    "id": -1,
                    "name": entity_name,
                    "text": "",
                    "url": "",
                    "level": 2,
                    "ent_ipc": "",
                    "parent_ent": None
                })
                not_found_titles.append(entity_name)

    if not_found_titles:
        logger.warning("Additional entities not found in catalog: %s", not_found_titles)

    logger.info("Added %d additional entities from whitelist", len(additional_entity_names))
    return entities
