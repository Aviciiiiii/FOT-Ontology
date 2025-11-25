"""
Chunk Merging with Complete FOT Processing

Ported from /src/new_merge_script.py (110 lines).
This module implements sophisticated chunk merging with:
- FOT name cleaning (39s/39 suffix removal, patent number removal)
- Third layer entity filtering (static ontology integration)
- IPC code aggregation for same-name FOTs
- FOT ID reindexing from max(third_layer_id) + 1
- Patent FOT reference updating
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..utils.logging import get_logger

logger = get_logger("merge_chunks")


def clean_fot_name(name: str) -> str:
    """Clean FOT name by removing encoding artifacts and patent numbers.

    Removes:
    - '39s' and '39' suffixes from words (encoding artifacts)
    - Patent number patterns like US123456, EP789012

    Args:
        name: Raw FOT name from extraction

    Returns:
        Cleaned FOT name

    Examples:
        >>> clean_fot_name("bread making39s")
        'bread making'
        >>> clean_fot_name("device39")
        'device'
        >>> clean_fot_name("semiconductor device US123456")
        'semiconductor device'
    """
    # Clean each word's '39s' and '39' suffix
    sep_name = name.split(' ')
    cleaned_words = []
    for one_name in sep_name:
        if one_name.endswith("39s"):
            one_name = one_name[:-3]
        elif one_name.endswith("39"):
            one_name = one_name[:-2]
        cleaned_words.append(one_name)
    name = ' '.join(cleaned_words)

    # Remove patent number patterns (e.g., US123456, EP789012)
    name = re.sub(r'\b[a-zA-Z]{2}\d+[a-zA-Z0-9]*$', '', name).strip()

    return name


def load_third_layer_entities(third_layer_file: str) -> Tuple[Set[Tuple[str, str]], int]:
    """Load third layer static entities for deduplication.

    Args:
        third_layer_file: Path to fot_level_3.txt (TSV: fot_id, name, ipc_code)

    Returns:
        Tuple of (entity_set, max_fot_id)
        - entity_set: Set of (name, ipc_code) tuples
        - max_fot_id: Maximum FOT ID in third layer
    """
    third_layer_set = set()
    max_fot_id = 0

    if not third_layer_file:
        logger.warning("No third layer file specified, skipping entity filtering")
        return third_layer_set, max_fot_id

    third_layer_path = Path(third_layer_file)
    if not third_layer_path.exists():
        logger.warning(f"Third layer file not found: {third_layer_file}")
        return third_layer_set, max_fot_id

    try:
        with third_layer_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                try:
                    fot_id = int(row.get('fot_id', 0))
                    max_fot_id = max(max_fot_id, fot_id)
                except ValueError:
                    continue

                name = row.get('name', '').strip()
                ipc_code = row.get('ipc_code', '').strip()
                if name:
                    third_layer_set.add((name, ipc_code))

        logger.info(
            f"Loaded {len(third_layer_set)} third layer entities, max_id={max_fot_id}"
        )
    except Exception as e:
        logger.error(f"Failed to load third layer file: {e}")

    return third_layer_set, max_fot_id


def aggregate_fot_data(
    all_fot_data: List[List[Tuple[int, str, str]]],
    third_layer_set: Set[Tuple[str, str]]
) -> Dict[str, Dict]:
    """Aggregate FOT data from multiple chunks with cleaning and filtering.

    Args:
        all_fot_data: List of chunk FOT data (each is list of (id, name, ipc) tuples)
        third_layer_set: Set of third layer entities for filtering

    Returns:
        Dictionary mapping cleaned FOT name to aggregated data:
        {
            'fot_name': {
                'original_ids': [1, 5, 12],  # Original IDs from chunks
                'ipc_codes': {'A21C', 'A21C1', 'A21D'},  # Unique IPC codes
                'first_id': 1  # First occurrence ID
            }
        }
    """
    fot_aggregated = defaultdict(lambda: {
        'original_ids': [],
        'ipc_codes': set(),
        'first_id': None
    })

    filtered_count = 0

    for chunk_fots in all_fot_data:
        for fot_id, fot_name, ipc_code in chunk_fots:
            # Clean FOT name
            cleaned_name = clean_fot_name(fot_name)

            # Filter against third layer
            if third_layer_set and (cleaned_name, ipc_code) in third_layer_set:
                filtered_count += 1
                logger.debug(f"Filtered duplicate: {cleaned_name} ({ipc_code})")
                continue

            # Track original IDs
            fot_aggregated[cleaned_name]['original_ids'].append(fot_id)

            # Set first ID
            if fot_aggregated[cleaned_name]['first_id'] is None:
                fot_aggregated[cleaned_name]['first_id'] = fot_id

            # Aggregate IPC codes
            if ipc_code:
                for ipc in ipc_code.split(','):
                    ipc_stripped = ipc.strip()
                    if ipc_stripped:
                        fot_aggregated[cleaned_name]['ipc_codes'].add(ipc_stripped)

    logger.info(
        f"Aggregated {len(fot_aggregated)} unique FOTs, filtered {filtered_count} duplicates"
    )
    return dict(fot_aggregated)


def create_id_mapping(
    fot_aggregated: Dict[str, Dict],
    start_id: int
) -> Tuple[List[Tuple[int, str, str]], Dict[int, int]]:
    """Create new FOT IDs and mapping from old to new IDs.

    Args:
        fot_aggregated: Aggregated FOT data from aggregate_fot_data()
        start_id: Starting FOT ID (usually max_third_layer_id + 1)

    Returns:
        Tuple of (fot_list, id_mapping)
        - fot_list: List of (new_id, name, aggregated_ipc) tuples
        - id_mapping: Dict mapping old_id -> new_id
    """
    fot_list = []
    old_to_new_id = {}
    current_id = start_id

    for fot_name, data in sorted(fot_aggregated.items()):
        # Sort and join IPC codes
        aggregated_ipc = ','.join(sorted(data['ipc_codes']))

        # Assign new ID
        new_id = current_id
        current_id += 1

        # Map all original IDs to this new ID
        for old_id in data['original_ids']:
            old_to_new_id[old_id] = new_id

        fot_list.append((new_id, fot_name, aggregated_ipc))

    logger.info(
        f"Created ID mapping: {len(old_to_new_id)} old IDs -> {len(fot_list)} new IDs"
    )
    return fot_list, old_to_new_id


def update_patent_fot_references(
    patent_rows: List[List[str]],
    old_to_new_id: Dict[int, int]
) -> List[List[str]]:
    """Update FOT ID references in patent data.

    Args:
        patent_rows: List of [patentid, title, FOT] rows
        old_to_new_id: Mapping from old FOT IDs to new FOT IDs

    Returns:
        Updated patent rows with new FOT IDs
    """
    updated_rows = []

    for row in patent_rows:
        if len(row) < 3:
            updated_rows.append(row)
            continue

        patentid, title, fot_string = row[0], row[1], row[2]

        # Update FOT string
        if fot_string and fot_string != 'nan' and fot_string.strip():
            updated_fot = _update_fot_string(fot_string, old_to_new_id)
            updated_rows.append([patentid, title, updated_fot])
        else:
            updated_rows.append(row)

    return updated_rows


def _update_fot_string(fot_string: str, old_to_new_id: Dict[int, int]) -> str:
    """Update FOT ID in FOT string format: "1:0:1, 5:2:3" -> "1003:0:1, 1007:2:3".

    Args:
        fot_string: Original FOT string with old IDs
        old_to_new_id: Mapping from old to new IDs

    Returns:
        Updated FOT string with new IDs
    """
    updated_entries = []

    for entry in fot_string.split(','):
        entry = entry.strip()
        if ':' in entry:
            parts = entry.split(':')
            if len(parts) == 3:
                try:
                    old_id = int(parts[0])
                    new_id = old_to_new_id.get(old_id, old_id)
                    updated_entries.append(f"{new_id}:{parts[1]}:{parts[2]}")
                except ValueError:
                    updated_entries.append(entry)
            else:
                updated_entries.append(entry)
        else:
            updated_entries.append(entry)

    return ','.join(updated_entries)


def run_merge(
    chunks_dir: str,
    merged_fot_mapping: str,
    merged_patent_fot: str,
    *,
    third_layer_file: Optional[str] = None,
    num_chunks: Optional[int] = None,
    clean_names: bool = True,
    aggregate_ipc: bool = True,
    reindex_from_third_layer: bool = True
) -> Tuple[str, str]:
    """Merge chunk outputs into consolidated files with complete processing.

    This function implements the complete merge logic from the original script:
    1. Read all chunk FOT and patent data
    2. Clean FOT names (remove 39s/39 suffixes and patent numbers)
    3. Load third layer entities and filter duplicates
    4. Aggregate IPC codes for same-name FOTs
    5. Reindex FOT IDs starting from max(third_layer_id) + 1
    6. Update patent data with new FOT ID references
    7. Write merged output files with statistics

    Args:
        chunks_dir: Directory containing chunk_* folders
        merged_fot_mapping: Output FOT mapping file path
        merged_patent_fot: Output patent data file path
        third_layer_file: Path to fot_level_3.txt for deduplication (optional)
        num_chunks: Number of chunks to merge (None = all)
        clean_names: Whether to clean FOT names (default: True)
        aggregate_ipc: Whether to aggregate IPC codes (default: True)
        reindex_from_third_layer: Whether to start IDs from max_third_layer_id+1 (default: True)

    Returns:
        Tuple of (fot_mapping_path, patent_data_path)
    """
    logger.info("Starting chunk merge with complete processing")

    base = Path(chunks_dir)
    chunk_dirs = sorted([p for p in base.glob("chunk_*") if p.is_dir()])

    if num_chunks:
        chunk_dirs = chunk_dirs[:num_chunks]

    logger.info(f"Found {len(chunk_dirs)} chunks to merge")

    # Load third layer entities
    third_layer_set, max_third_layer_id = load_third_layer_entities(
        third_layer_file
    ) if third_layer_file and reindex_from_third_layer else (set(), 0)

    # Read all chunk data
    all_fot_data = []
    patent_rows = []

    for cdir in chunk_dirs:
        # Read FOT mapping
        fot_file = cdir / "fot_id_mapping.txt"
        if fot_file.exists():
            chunk_fots = []
            with fot_file.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    try:
                        fot_id = int(row.get('fot_id', 0))
                        fot_name = row.get('fot_name', '').strip()
                        ipc_code = row.get('ipc_code', '').strip()
                        if fot_name:
                            chunk_fots.append((fot_id, fot_name, ipc_code))
                    except ValueError:
                        continue
            all_fot_data.append(chunk_fots)

        # Read patent data
        patent_file = cdir / "patent_title_with_FOT.txt"
        if patent_file.exists():
            with patent_file.open('r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader, None)  # Skip header
                for row in reader:
                    patent_rows.append(row)

    logger.info(f"Loaded {sum(len(c) for c in all_fot_data)} FOT entries, {len(patent_rows)} patent records")

    # Aggregate FOT data with cleaning and filtering
    if clean_names and aggregate_ipc:
        fot_aggregated = aggregate_fot_data(all_fot_data, third_layer_set)
    else:
        # Fallback to simple deduplication (backward compatible)
        fot_aggregated = {}
        for chunk_fots in all_fot_data:
            for fot_id, fot_name, ipc_code in chunk_fots:
                if fot_name not in fot_aggregated:
                    fot_aggregated[fot_name] = {
                        'original_ids': [fot_id],
                        'ipc_codes': {ipc_code} if ipc_code else set(),
                        'first_id': fot_id
                    }

    # Create new FOT IDs and mapping
    start_id = max_third_layer_id + 1 if reindex_from_third_layer else 1
    fot_list, old_to_new_id = create_id_mapping(fot_aggregated, start_id)

    # Update patent FOT references
    updated_patent_rows = update_patent_fot_references(patent_rows, old_to_new_id)

    # Write merged FOT mapping
    mfot = Path(merged_fot_mapping)
    mfot.parent.mkdir(parents=True, exist_ok=True)
    with mfot.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['fot_id', 'fot_name', 'ipc_code'])
        for fot_id, fot_name, ipc_code in fot_list:
            writer.writerow([fot_id, fot_name, ipc_code])

    # Write merged patent data
    mpat = Path(merged_patent_fot)
    mpat.parent.mkdir(parents=True, exist_ok=True)
    with mpat.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['patentid', 'title', 'FOT'])
        for row in updated_patent_rows:
            writer.writerow(row)

    # Statistics
    logger.info(f"Merged chunks={len(chunk_dirs)} -> patents={len(updated_patent_rows)} unique_fots={len(fot_list)}")
    logger.info(f"FOT ID range: {start_id} to {start_id + len(fot_list) - 1}")
    if third_layer_set:
        logger.info(f"Third layer entities: {len(third_layer_set)}, max_id={max_third_layer_id}")
    logger.info(f"Saved FOT mapping to {mfot}")
    logger.info(f"Saved patent data to {mpat}")

    return str(mfot), str(mpat)