from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Set, List, Optional

from ..utils.logging import get_logger

# Increase CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)


def generate_patent_id(counter: int) -> str:
    """Generate patent ID in format A0000001, A0000002, ..., B0000001, ...

    Supports up to 260M patents (26 letters × 10M each).

    Args:
        counter: 1-based counter (1, 2, 3, ...)

    Returns:
        Patent ID string (e.g., "A0000001")
    """
    prefix = chr(ord('A') + (counter - 1) // 10000000)
    suffix = str(((counter - 1) % 10000000) + 1).zfill(7)
    return f"{prefix}{suffix}"


def process_single_csv_file(
    csv_path: Path,
    title_to_patent_id: Dict[str, str],
    patent_id_counter: int,
    publication_numbers: Dict[str, Set[str]],
    ipc_codes: Dict[str, Set[str]],
    titles: Dict[str, str],
    logger,
    progress_interval: int = 10000
) -> int:
    """Process a single CSV file with deduplication and aggregation.

    Args:
        csv_path: Path to CSV file
        title_to_patent_id: Mapping from title to patent_id (for deduplication)
        patent_id_counter: Current patent ID counter
        publication_numbers: Aggregated publication numbers per patent_id
        ipc_codes: Aggregated IPC codes per patent_id
        titles: Patent ID to title mapping
        logger: Logger instance
        progress_interval: Log progress every N rows

    Returns:
        Updated patent_id_counter
    """
    logger.info(f"Processing file: {csv_path.name}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Count total rows for progress reporting
        rows = list(reader)
        total_rows = len(rows)
        logger.info(f"  Found {total_rows} rows in {csv_path.name}")

        for row_number, row in enumerate(rows, 1):
            if progress_interval > 0 and (row_number % progress_interval == 0 or row_number == total_rows):
                logger.info(f"  Progress: {row_number}/{total_rows} rows ({row_number/total_rows*100:.2f}%)")

            try:
                publication_number = row.get('publication_number', '').strip()
                title = row.get('title', '').strip()
                ipc_code = row.get('ipc_code', '').strip()

                if not title:
                    logger.warning(f"Empty title at row {row_number} in {csv_path.name}, skipping")
                    continue

                # Title-based deduplication: reuse patent_id for same title
                if title not in title_to_patent_id:
                    patent_id_counter += 1
                    patent_id = generate_patent_id(patent_id_counter)
                    title_to_patent_id[title] = patent_id
                    titles[patent_id] = title
                else:
                    patent_id = title_to_patent_id[title]

                # Aggregate publication_numbers and ipc_codes
                if publication_number:
                    publication_numbers[patent_id].add(publication_number)
                if ipc_code:
                    ipc_codes[patent_id].add(ipc_code)

            except Exception as e:
                logger.error(f"Error processing row {row_number} in {csv_path.name}: {str(e)}")
                logger.error(f"Row content: {row}")
                continue

    return patent_id_counter


def write_temp_files(
    temp_dir: Path,
    file_prefix: str,
    titles: Dict[str, str],
    publication_numbers: Dict[str, Set[str]],
    ipc_codes: Dict[str, Set[str]]
) -> Tuple[Path, Path, Path]:
    """Write temporary TSV files for batch merging.

    Args:
        temp_dir: Temporary directory
        file_prefix: Prefix for temp file names
        titles: Patent ID to title mapping
        publication_numbers: Aggregated publication numbers
        ipc_codes: Aggregated IPC codes

    Returns:
        Tuple of (temp_titles_path, temp_pubnums_path, temp_ipcs_path)
    """
    temp_titles_path = temp_dir / f"temp_titles_{file_prefix}.tsv"
    temp_pubnums_path = temp_dir / f"temp_pubnums_{file_prefix}.tsv"
    temp_ipcs_path = temp_dir / f"temp_ipcs_{file_prefix}.tsv"

    # Write titles
    with temp_titles_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        for patent_id, title in titles.items():
            writer.writerow([patent_id, title])

    # Write publication numbers (aggregated, comma-separated)
    with temp_pubnums_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        for patent_id, pubnums in publication_numbers.items():
            writer.writerow([patent_id, ','.join(sorted(pubnums))])

    # Write IPC codes (aggregated, comma-separated)
    with temp_ipcs_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        for patent_id, codes in ipc_codes.items():
            writer.writerow([patent_id, ','.join(sorted(codes))])

    return temp_titles_path, temp_pubnums_path, temp_ipcs_path


def merge_temp_files_batch(
    temp_files: List[Path],
    output_path: Path,
    field_name: str,
    logger,
    batch_size: int = 100000,
    is_aggregated: bool = True
) -> None:
    """Merge temporary TSV files with batch processing.

    Args:
        temp_files: List of temporary file paths
        output_path: Output file path
        field_name: Field name for header (e.g., "title", "publication_number")
        logger: Logger instance
        batch_size: Batch size for writing
        is_aggregated: If True, merge values with comma separator
    """
    logger.info(f"Merging {len(temp_files)} temp files to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow(['patent_id', field_name])

        if is_aggregated:
            # Merge aggregated data (for publication_numbers and ipc_codes)
            merged_data = defaultdict(set)
            total_processed = 0

            for temp_file in temp_files:
                logger.info(f"  Processing temp file: {temp_file.name}")
                with temp_file.open("r", encoding="utf-8") as infile:
                    reader = csv.reader(infile, delimiter='\t')
                    for row in reader:
                        if len(row) >= 2:
                            patent_id = row[0]
                            values = row[1].split(',')
                            merged_data[patent_id].update(values)

                            total_processed += 1
                            if total_processed % batch_size == 0:
                                # Write batch
                                for pid, vals in merged_data.items():
                                    writer.writerow([pid, ','.join(sorted(vals))])
                                merged_data.clear()
                                logger.info(f"  Processed {total_processed} records")

            # Write remaining data
            if merged_data:
                for pid, vals in merged_data.items():
                    writer.writerow([pid, ','.join(sorted(vals))])

            logger.info(f"Merge complete, total {total_processed} records")
        else:
            # Simple merge for titles (no aggregation needed)
            seen_ids = set()
            for temp_file in temp_files:
                with temp_file.open("r", encoding="utf-8") as infile:
                    reader = csv.reader(infile, delimiter='\t')
                    for row in reader:
                        if len(row) >= 2:
                            patent_id = row[0]
                            if patent_id not in seen_ids:
                                writer.writerow(row)
                                seen_ids.add(patent_id)


def process_raw_csv(
    raw_dir: str,
    titles_out: str,
    pubnum_out: str,
    ipc_out: str,
    *,
    enable_deduplication: bool = True,
    enable_aggregation: bool = True,
    use_temp_files: Optional[bool] = None,
    batch_size: int = 100000,
    progress_interval: int = 10000
) -> Tuple[str, str, str]:
    """Process CSV files with title deduplication and data aggregation.

    Implements the complete logic from original /src/process_patent.py:
    - Title-based deduplication: same title shares same patent_id
    - Data aggregation: multiple publication_numbers and IPC codes per patent
    - Large-scale processing: temp files and batch merging for 88M+ patents

    Args:
        raw_dir: Input CSV directory
        titles_out: Output path for titles.txt
        pubnum_out: Output path for publication_number.txt
        ipc_out: Output path for ipc_codes.txt
        enable_deduplication: Enable title-based deduplication
        enable_aggregation: Enable data aggregation
        use_temp_files: Use temporary files (auto-detect if None)
        batch_size: Batch size for merging
        progress_interval: Log progress every N rows (0 to disable)

    Returns:
        Tuple of (titles_path, pubnum_path, ipc_path)
    """
    logger = get_logger("process_csv")
    raw = Path(raw_dir)
    csv_files = sorted(raw.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {raw_dir}")
        # Create empty output files
        for out_path, field_name in [(titles_out, "title"),
                                      (pubnum_out, "publication_number"),
                                      (ipc_out, "ipc_code")]:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['patent_id', field_name])
        return str(Path(titles_out)), str(Path(pubnum_out)), str(Path(ipc_out))

    logger.info(f"Found {len(csv_files)} CSV files to process")

    # Auto-detect temp file usage based on total file size
    if use_temp_files is None:
        total_size = sum(f.stat().st_size for f in csv_files)
        use_temp_files = total_size > 100 * 1024 * 1024  # > 100MB
        logger.info(f"Auto-detected use_temp_files={use_temp_files} (total size: {total_size/1024/1024:.2f} MB)")

    if not enable_deduplication and not enable_aggregation:
        # Simple mode: no deduplication or aggregation (original simplified logic)
        logger.info("Running in simple mode (no deduplication/aggregation)")
        return _process_simple(csv_files, titles_out, pubnum_out, ipc_out, logger)

    # Full mode: with deduplication and aggregation
    logger.info("Running in full mode (with deduplication and aggregation)")

    if use_temp_files:
        return _process_with_temp_files(
            csv_files, titles_out, pubnum_out, ipc_out, logger,
            batch_size, progress_interval
        )
    else:
        return _process_in_memory(
            csv_files, titles_out, pubnum_out, ipc_out, logger,
            progress_interval
        )


def _process_simple(
    csv_files: List[Path],
    titles_out: str,
    pubnum_out: str,
    ipc_out: str,
    logger
) -> Tuple[str, str, str]:
    """Simple processing without deduplication (for dry-run mode)."""
    rows = []
    for p in csv_files:
        with p.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    logger.info("Read raw CSV rows=%d from %d files", len(rows), len(csv_files))

    def gen_pid(i: int) -> str:
        return f"A{(i+1):07d}"

    titles = []
    pubnums = []
    ipcs = defaultdict(set)
    for i, r in enumerate(rows):
        pid = gen_pid(i)
        title = r.get("title", "").strip()
        pub = r.get("publication_number", "").strip()
        ipc = r.get("ipc_code", "").strip()
        titles.append((pid, title))
        pubnums.append((pid, pub))
        if ipc:
            ipcs[pid].add(ipc)

    # Write outputs (TSV)
    def write_tsv(path: str, header: tuple, pairs):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(header)
            for a, b in pairs:
                w.writerow([a, b])
        return str(p)

    titles_path = write_tsv(titles_out, ("patent_id", "title"), titles)
    pub_path = write_tsv(pubnum_out, ("patent_id", "publication_number"), pubnums)
    ipc_pairs = [(pid, ",".join(sorted(vals))) for pid, vals in ipcs.items()]
    ipc_path = write_tsv(ipc_out, ("patent_id", "ipc_code"), ipc_pairs)

    logger.info("Wrote titles=%d pubnums=%d ipccodes=%d", len(titles), len(pubnums), len(ipc_pairs))
    return titles_path, pub_path, ipc_path


def _process_in_memory(
    csv_files: List[Path],
    titles_out: str,
    pubnum_out: str,
    ipc_out: str,
    logger,
    progress_interval: int
) -> Tuple[str, str, str]:
    """Process all files in memory with deduplication and aggregation."""
    title_to_patent_id = {}
    publication_numbers = defaultdict(set)
    ipc_codes = defaultdict(set)
    titles = {}
    patent_id_counter = 0

    for csv_file in csv_files:
        patent_id_counter = process_single_csv_file(
            csv_file,
            title_to_patent_id,
            patent_id_counter,
            publication_numbers,
            ipc_codes,
            titles,
            logger,
            progress_interval
        )

    logger.info(f"Total unique patents: {len(titles)}")
    logger.info(f"Total unique titles: {len(title_to_patent_id)}")

    # Write outputs
    def write_tsv(path: str, header: tuple, data_dict, is_aggregated=False):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(header)
            for pid, value in sorted(data_dict.items()):
                if is_aggregated and isinstance(value, set):
                    w.writerow([pid, ','.join(sorted(value))])
                else:
                    w.writerow([pid, value])
        return str(p)

    titles_path = write_tsv(titles_out, ("patent_id", "title"), titles)
    pub_path = write_tsv(pubnum_out, ("patent_id", "publication_number"),
                         publication_numbers, is_aggregated=True)
    ipc_path = write_tsv(ipc_out, ("patent_id", "ipc_code"),
                         ipc_codes, is_aggregated=True)

    logger.info("Wrote titles=%d pubnums=%d ipccodes=%d",
                len(titles), len(publication_numbers), len(ipc_codes))
    return titles_path, pub_path, ipc_path


def _process_with_temp_files(
    csv_files: List[Path],
    titles_out: str,
    pubnum_out: str,
    ipc_out: str,
    logger,
    batch_size: int,
    progress_interval: int
) -> Tuple[str, str, str]:
    """Process files using temporary files for large-scale data."""
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Using temporary directory: {temp_dir}")

    try:
        temp_titles_files = []
        temp_pubnum_files = []
        temp_ipc_files = []

        # Global deduplication state
        title_to_patent_id = {}
        patent_id_counter = 0

        for file_index, csv_file in enumerate(csv_files, 1):
            logger.info(f"\nProcessing file {file_index}/{len(csv_files)}: {csv_file.name}")

            # Process each file separately
            publication_numbers = defaultdict(set)
            ipc_codes = defaultdict(set)
            titles = {}

            patent_id_counter = process_single_csv_file(
                csv_file,
                title_to_patent_id,
                patent_id_counter,
                publication_numbers,
                ipc_codes,
                titles,
                logger,
                progress_interval
            )

            # Write temporary files
            temp_titles, temp_pubnums, temp_ipcs = write_temp_files(
                temp_dir,
                f"{file_index:04d}",
                titles,
                publication_numbers,
                ipc_codes
            )

            temp_titles_files.append(temp_titles)
            temp_pubnum_files.append(temp_pubnums)
            temp_ipc_files.append(temp_ipcs)

        logger.info("\nAll files processed, merging temporary files...")

        # Merge temporary files
        merge_temp_files_batch(temp_titles_files, Path(titles_out), "title",
                              logger, batch_size, is_aggregated=False)
        merge_temp_files_batch(temp_pubnum_files, Path(pubnum_out), "publication_number",
                              logger, batch_size, is_aggregated=True)
        merge_temp_files_batch(temp_ipc_files, Path(ipc_out), "ipc_code",
                              logger, batch_size, is_aggregated=True)

        logger.info("Processing complete")
        return str(Path(titles_out)), str(Path(pubnum_out)), str(Path(ipc_out))

    finally:
        shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory deleted: {temp_dir}")