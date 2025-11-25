from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from ..utils.logging import get_logger
from ..utils.validate import validate_stage3_outputs
from ..utils.runinfo import new_run_id
from ..utils import paths as pathutil
from ..patents.bq_export import run_export
from ..patents.gcs_download import run_download
from ..patents.process_csv import process_raw_csv
from ..extraction.generate_fot import run_generate
from ..extraction.merge_chunks import run_merge
from ..ontology.dynamic.hierarchy import run as run_hierarchy


def run(cfg: Dict[str, Any], *, dry_run: bool = True, fast: bool = True, run_id: str | None = None) -> Dict[str, str]:
    logger = get_logger("stage3_patents_fot")
    run_id = run_id or new_run_id()
    outputs = cfg.get("outputs", {})

    raw_dir = outputs.get("raw_patents_dir", "data/raw_patents")
    if not (raw_dir.startswith("/") or raw_dir.startswith(".")):
        raw_dir = pathutil.expand("data.raw", raw_dir.split("/", 1)[-1] if raw_dir.startswith("data/") else raw_dir)
    titles = outputs.get("titles", "data/processed/titles.txt")
    pubnums = outputs.get("publication_numbers", "data/processed/publication_number.txt")
    ipc = outputs.get("ipc_codes", "data/processed/ipc_codes.txt")
    chunks_dir = outputs.get("chunks_dir", "artifacts/chunks")
    if not (chunks_dir.startswith("/") or chunks_dir.startswith(".")):
        chunks_dir = pathutil.expand("artifacts.chunks", "")
    merged_fot_mapping = outputs.get("merged_fot_mapping", "data/processed/new_total_merged_fot_mapping.txt")
    merged_patent_fot = outputs.get("merged_patent_fot", "data/processed/new_total_merged_patent_title_with_FOT.txt")

    # Check for existing files to skip unnecessary steps
    raw_dir_path = Path(raw_dir)
    raw_files_exist = raw_dir_path.exists() and any(raw_dir_path.glob("*.csv"))

    titles_path_obj = Path(titles)
    pub_path_obj = Path(pubnums)
    ipc_path_obj = Path(ipc)
    processed_files_exist = titles_path_obj.exists() and pub_path_obj.exists() and ipc_path_obj.exists()

    # 1) Export & Download
    if processed_files_exist:
        logger.info("=" * 80)
        logger.info("Processed files already exist, skipping export/download/processing steps:")
        logger.info(f"  ✓ Titles: {titles}")
        logger.info(f"  ✓ Publication numbers: {pubnums}")
        logger.info(f"  ✓ IPC codes: {ipc}")
        logger.info("=" * 80)
        titles_path = str(titles_path_obj)
        pub_path = str(pub_path_obj)
        ipc_path = str(ipc_path_obj)
        exported = None
        downloaded = None
    elif raw_files_exist and not dry_run:
        logger.info("=" * 80)
        logger.info(f"Raw patent CSV files already exist in {raw_dir}")
        logger.info("Skipping BigQuery export and GCS download steps")
        logger.info("=" * 80)
        exported = None
        downloaded = str(raw_dir)
    elif dry_run:
        # Dry-run mode: Generate synthetic CSV
        logger.info("Running in dry-run mode: generating synthetic data")
        exported = run_export(raw_dir, dry_run=True)
        downloaded = run_download(exported_csv=exported, raw_dir=raw_dir, dry_run=True)
    else:
        # Real mode: Execute BigQuery export and GCS download
        logger.info("Running in real mode: using BigQuery and GCS APIs")
        exported = run_export(raw_dir, dry_run=False, config=cfg)
        downloaded = run_download(exported_csv=exported, raw_dir=raw_dir, dry_run=False, config=cfg)

    # 2) Process CSVs → TSVs
    if processed_files_exist:
        # Already checked and logged above, skip processing
        pass
    else:
        # Get processing configuration
        proc_cfg = cfg.get("processing", {})
        enable_dedup = proc_cfg.get("enable_deduplication", True)
        enable_agg = proc_cfg.get("enable_aggregation", True)
        use_temp = proc_cfg.get("use_temp_files", None)
        batch_size = proc_cfg.get("batch_size", 100000)
        progress_interval = proc_cfg.get("progress_interval", 10000)

        # In dry-run mode, disable deduplication/aggregation for simplicity
        if dry_run:
            enable_dedup = False
            enable_agg = False
            use_temp = False
            progress_interval = 0  # Disable progress logging in dry-run

        titles_path, pub_path, ipc_path = process_raw_csv(
            raw_dir, titles, pubnums, ipc,
            enable_deduplication=enable_dedup,
            enable_aggregation=enable_agg,
            use_temp_files=use_temp,
            batch_size=batch_size,
            progress_interval=progress_interval
        )

    # 3) Generate FOT chunks
    # Get extraction configuration
    extraction_mode = cfg.get("mode", "simple" if dry_run else "ner")
    ner_cfg = cfg.get("ner_model", {})
    model_path = ner_cfg.get("model_path") if not dry_run else None

    # Check if NER model exists (for real mode)
    if extraction_mode == "ner" and model_path and not dry_run:
        if not Path(model_path).exists():
            logger.warning(f"NER model not found at {model_path}, falling back to simple mode")
            extraction_mode = "simple"
            model_path = None

    # Build extraction config
    extraction_config = {
        "batch_size": ner_cfg.get("batch_size", 128),
        "max_len": ner_cfg.get("max_len", 48),
        "num_chunks": ner_cfg.get("num_chunks", 20),
        "use_amp": ner_cfg.get("use_amp", True),
        "pos_weight_dict": cfg.get("pos_weights", {
            'NOUN': 1.3, 'PROPN': 1.3, 'ADJ': 1.1, 'VERB': 0.9, 'NUM': 0.8,
            'ADP': 0.6, 'DET': 0.5, 'CCONJ': 0.6, 'PART': 0.6, 'PRON': 0.5,
            'AUX': 0.5, 'ADV': 0.7, 'SCONJ': 0.6, 'INTJ': 0.4, 'SYM': 0.7, 'X': 0.8,
        })
    }

    chunk_path = run_generate(
        titles_path, ipc_path, chunks_dir,
        dry_run=(dry_run or extraction_mode == "simple"),
        fast=fast,
        model_path=model_path,
        config=extraction_config
    )

    # 4) Merge chunks with complete processing
    # Get merge configuration
    merge_cfg = cfg.get("merge", {})
    third_layer_file = merge_cfg.get("third_layer_file")
    num_chunks = merge_cfg.get("num_chunks")
    clean_names = merge_cfg.get("clean_names", True)
    aggregate_ipc = merge_cfg.get("aggregate_ipc", True)
    reindex_from_third_layer = merge_cfg.get("reindex_from_third_layer", True)

    # For dry-run, disable advanced features to keep simple
    if dry_run:
        third_layer_file = None
        num_chunks = None
        clean_names = False
        aggregate_ipc = False
        reindex_from_third_layer = False

    mfot, mpat = run_merge(
        chunks_dir, merged_fot_mapping, merged_patent_fot,
        third_layer_file=third_layer_file,
        num_chunks=num_chunks,
        clean_names=clean_names,
        aggregate_ipc=aggregate_ipc,
        reindex_from_third_layer=reindex_from_third_layer
    )

    # 5) Build hierarchy (optional, only in real mode with config)
    hierarchy_result = None
    hierarchy_cfg = cfg.get("hierarchy", {})
    build_hierarchy = hierarchy_cfg.get("enabled", False)

    if build_hierarchy and not dry_run:
        logger.info("=" * 80)
        logger.info("Building dynamic FOT hierarchy (layer assignment + parent linking)")
        logger.info("=" * 80)

        try:
            # Load hierarchy configuration
            hierarchy_config_path = pathutil.expand("configs", "hierarchy.yaml")
            if Path(hierarchy_config_path).exists():
                from ..utils.paths import load_yaml_once
                hierarchy_full_cfg = load_yaml_once(hierarchy_config_path)
            else:
                logger.warning(f"Hierarchy config not found: {hierarchy_config_path}, using defaults")
                hierarchy_full_cfg = {}

            # Prepare paths
            inputs = hierarchy_full_cfg.get('inputs', {})
            outputs_hier = hierarchy_full_cfg.get('outputs', {})

            # Use merged FOT mapping as input for hierarchy
            fot_entities_path = mfot  # Use the merged output from previous step
            fot_layer_12_path = pathutil.expand("data.interim", inputs.get('fot_layer_12', 'fot_level_12.txt'))
            fot_layer_3_path = pathutil.expand("data.interim", inputs.get('fot_layer_3', 'fot_level_3.txt'))

            # Check if static layer files exist
            if not Path(fot_layer_3_path).exists():
                logger.warning(f"Static layer L3 not found: {fot_layer_3_path}")
                logger.warning("Skipping hierarchy construction (requires Stage 1 output)")
                logger.warning("To enable: run `python -m fot.cli build-static-ontology --config-dir configs`")
            else:
                output_dynamic_path = pathutil.expand("data.processed", outputs_hier.get('dynamic_hierarchy', 'new_fot_hierarchy_dynamic.txt'))
                output_full_path = pathutil.expand("data.processed", outputs_hier.get('full_library', 'full_fot_library.txt'))

                embedding_cfg = hierarchy_full_cfg.get('embeddings', {})
                model_path = embedding_cfg.get('model_path', 'files/scibert_scivocab_uncased')
                embedding_cache = pathutil.expand("artifacts.embeddings", outputs_hier.get('embedding_cache', 'hierarchy_embeddings.npz'))
                id_offset = hierarchy_full_cfg.get('hierarchy', {}).get('id_offset', 11834)
                use_gpu = embedding_cfg.get('use_gpu', False)

                hierarchy_result = run_hierarchy(
                    cfg=hierarchy_full_cfg,
                    fot_entities_path=fot_entities_path,
                    fot_layer_12_path=fot_layer_12_path,
                    fot_layer_3_path=fot_layer_3_path,
                    output_dynamic_path=output_dynamic_path,
                    output_full_path=output_full_path,
                    model_path=model_path,
                    embedding_cache=embedding_cache,
                    id_offset=id_offset,
                    use_gpu=use_gpu
                )

                logger.info("✓ Hierarchy construction complete!")
                logger.info(f"  Dynamic hierarchy: {hierarchy_result['dynamic_hierarchy']}")
                logger.info(f"  Full library: {hierarchy_result['full_library']}")
                logger.info(f"  Dynamic entities: {hierarchy_result['num_dynamic_entities']}")
                logger.info(f"  Total entities: {hierarchy_result['num_total_entities']}")

        except Exception as e:
            logger.error(f"Hierarchy construction failed: {e}")
            logger.exception("Full traceback:")
            logger.warning("Continuing without hierarchy (Stage 3 outputs are still valid)")

    elif build_hierarchy and dry_run:
        logger.info("Hierarchy construction skipped in dry-run mode")
    else:
        logger.info("Hierarchy construction disabled (set hierarchy.enabled=true in extraction.yaml to enable)")

    # Helpers
    def _size_kb(p: str) -> int:
        try:
            return max(1, int((Path(p).stat().st_size + 1023) / 1024))
        except Exception:
            return 0

    def _count_tsv(path: str) -> int:
        try:
            with open(path, "r", encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1
                return max(0, n)
        except Exception:
            return 0

    logger.info(
        "Counts | titles=%d pubnums=%d ipcs=%d | chunk_patents=%d | merged patents&fots=%d & %d",
        _count_tsv(titles_path),
        _count_tsv(pub_path),
        _count_tsv(ipc_path),
        _count_tsv(Path(chunks_dir) / "chunk_000" / "patent_title_with_FOT.txt"),
        _count_tsv(mpat),
        _count_tsv(mfot),
    )

    # Validate TSV headers
    val = validate_stage3_outputs(titles_path, pub_path, ipc_path)
    if not val.get("ok"):
        logger.error("Stage3 validation failed: %s", val)
    else:
        logger.info("Stage3 validation passed: %s", val)

    base_result = {
        "titles": str(Path(titles_path)),
        "publication_numbers": str(Path(pub_path)),
        "ipc_codes": str(Path(ipc_path)),
        "chunk_000": str(Path(chunks_dir) / "chunk_000"),
        "merged_fot_mapping": str(Path(mfot)),
        "merged_patent_fot": str(Path(mpat)),
    }
    result_with_sizes = dict(base_result)
    for k, v2 in base_result.items():
        vv = v2 if not str(v2).endswith("chunk_000") else str(Path(v2) / "patent_title_with_FOT.txt")
        result_with_sizes[f"{k}_kb"] = _size_kb(vv)
    result_with_sizes["validation_ok"] = bool(val.get("ok"))
    result_with_sizes["run_id"] = run_id
    return result_with_sizes
