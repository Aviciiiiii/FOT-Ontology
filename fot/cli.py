from __future__ import annotations

import argparse
import os

from .utils import logging as logutil
from .utils import paths as pathutil
from .utils.paths import load_yaml_once
from .utils.runinfo import new_run_id


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config-dir", default="configs", help="Directory containing YAML configs (default: ./configs)")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    p.add_argument("--log-to-file", action="store_true", help="Also write logs to file under artifacts.logs")


def cmd_build_static_ontology(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("build_static_ontology", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))

    cfg_path = os.path.join(args.config_dir, "ontology_static.yaml")
    cfg = load_yaml_once(cfg_path)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid config format: {cfg_path}")
    # Override blink mode from CLI if provided
    if args.blink_mode:
        cfg.setdefault("blink", {})["mode"] = args.blink_mode

    from .pipelines.stage1_static_ontology import run as stage1_run
    run_id = new_run_id()
    result = stage1_run(cfg, dry_run=args.dry_run, fast=args.fast, skip_blink=args.skip_blink, generate_rename_map=args.rename_map, run_id=run_id)
    logger.info("Artifacts: %s", result)
    print(result)
    return 0


def cmd_build_dynamic_ontology(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("build_dynamic_ontology", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    cfg_path = os.path.join(args.config_dir, "ontology_dynamic.yaml")
    cfg = load_yaml_once(cfg_path)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid config format: {cfg_path}")
    from .pipelines.stage2_dynamic_ontology import run as stage2_run
    run_id = new_run_id()
    result = stage2_run(cfg, dry_run=args.dry_run, fast=args.fast, run_id=run_id)
    logger.info("Artifacts: %s", result)
    print(result)
    return 0


def cmd_patents_etl(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("patents_etl", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    logger.info("TODO: not implemented yet")
    return 0


def cmd_extract_fot(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("extract_fot", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    logger.info("TODO: not implemented yet")
    return 0


def cmd_build_exp(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("build_exp", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    logger.info("TODO: not implemented yet")
    return 0


def _load_data_cfg(config_dir: str) -> dict:
    cfg_path = os.path.join(config_dir, "data.yaml")
    cfg = load_yaml_once(cfg_path)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid data config format: {cfg_path}")
    return cfg


def cmd_data_download(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("data_download", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    cfg = _load_data_cfg(args.config_dir)
    from .data.raw.download import run as stage_run

    result = stage_run(cfg, dry_run=args.dry_run, fast=args.fast)
    logger.info("Artifacts: %s", result)
    print(result)
    return 0


def cmd_data_process_patent(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("data_process_patent", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    cfg = _load_data_cfg(args.config_dir)
    from .data.process.patent import run as stage_run

    result = stage_run(cfg, dry_run=args.dry_run, fast=args.fast)
    logger.info("Artifacts: %s", result)
    print(result)
    return 0


def cmd_data_generate_fot(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("data_generate_fot", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    cfg = _load_data_cfg(args.config_dir)
    from .ontology.generate_fot import run as stage_run

    result = stage_run(cfg, dry_run=args.dry_run, fast=args.fast)
    logger.info("Artifacts: %s", result)
    print(result)
    return 0


def cmd_data_merge(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("data_merge", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    cfg = _load_data_cfg(args.config_dir)
    from .data.process.merge import run as stage_run

    result = stage_run(cfg, dry_run=args.dry_run, fast=args.fast)
    logger.info("Artifacts: %s", result)
    print(result)
    return 0


def cmd_data_build_exp(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("data_build_exp", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    cfg = _load_data_cfg(args.config_dir)
    from .data.build_exp_dataset import run as stage_run

    result = stage_run(cfg, dry_run=args.dry_run, fast=args.fast)
    logger.info("Artifacts: %s", result)
    print(result)
    return 0


def cmd_data_simulate_history(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("data_simulate_history", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    cfg = _load_data_cfg(args.config_dir)
    from .data.simulate_user_history import run as stage_run

    result = stage_run(cfg, dry_run=args.dry_run, fast=args.fast)
    logger.info("Artifacts: %s", result)
    print(result)
    return 0


def cmd_recsys(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("recsys", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    logger.info("TODO: not implemented yet")
    return 0


def cmd_build_hierarchy(args: argparse.Namespace) -> int:
    """Build dynamic FOT hierarchy with layer assignment and parent linking."""
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("build_hierarchy", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))

    cfg_path = os.path.join(args.config_dir, "hierarchy.yaml")
    cfg = load_yaml_once(cfg_path)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid config format: {cfg_path}")

    from .ontology.dynamic.hierarchy import run as hierarchy_run

    # Expand paths
    inputs = cfg.get('inputs', {})
    outputs = cfg.get('outputs', {})

    fot_entities_path = pathutil.expand("artifacts.chunks", inputs.get('fot_entities', 'merged/updated_fot_mapping.txt'))
    fot_layer_12_path = pathutil.expand("data.interim", inputs.get('fot_layer_12', 'fot_level_12.txt'))
    fot_layer_3_path = pathutil.expand("data.interim", inputs.get('fot_layer_3', 'fot_level_3.txt'))

    output_dynamic_path = pathutil.expand("data.processed", outputs.get('dynamic_hierarchy', 'new_fot_hierarchy_dynamic.txt'))
    output_full_path = pathutil.expand("data.processed", outputs.get('full_library', 'full_fot_library.txt'))

    embedding_cfg = cfg.get('embeddings', {})
    model_path = embedding_cfg.get('model_path', 'files/scibert_scivocab_uncased')
    embedding_cache = pathutil.expand("artifacts.embeddings", outputs.get('embedding_cache', 'hierarchy_embeddings.npz'))
    id_offset = cfg.get('hierarchy', {}).get('id_offset', 11834)
    use_gpu = embedding_cfg.get('use_gpu', False)

    logger.info(f"Building hierarchy from {fot_entities_path}")
    logger.info(f"Static layer L1/L2: {fot_layer_12_path}")
    logger.info(f"Static layer L3: {fot_layer_3_path}")

    result = hierarchy_run(
        cfg=cfg,
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

    logger.info("Hierarchy construction complete!")
    logger.info(f"Dynamic hierarchy: {result['dynamic_hierarchy']}")
    logger.info(f"Full library: {result['full_library']}")
    logger.info(f"Dynamic entities: {result['num_dynamic_entities']}")
    logger.info(f"Total entities: {result['num_total_entities']}")
    print(result)
    return 0


def cmd_ner_comparison(args: argparse.Namespace) -> int:
    """Run NER model comparison experiments."""
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("ner_comparison", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))

    # Load configuration
    cfg_path = os.path.join(args.config_dir, "ner_comparison.yaml")
    cfg = load_yaml_once(cfg_path)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid config format: {cfg_path}")

    from .exp.ner_comparison import run_ner_comparison_experiments

    # Run experiments
    results = run_ner_comparison_experiments(
        config_dir=args.config_dir,
        data_dir=cfg.get("data", {}).get("train_dataset", "data/processed"),
        output_path=cfg.get("experiment", {}).get("results_file", "reports/ner_comparison_results.json"),
        pretrain_model_path=args.pretrain_model if hasattr(args, 'pretrain_model') else None,
        num_epochs=cfg.get("training", {}).get("num_epochs", 20),
        batch_size=cfg.get("data", {}).get("batch_size", 64),
        sample_size=cfg.get("data", {}).get("sample_size", None),
        skip_commercial=args.skip_commercial if hasattr(args, 'skip_commercial') else True,
        dry_run=args.dry_run if hasattr(args, 'dry_run') else False,
    )

    logger.info("NER comparison complete. Results: %s", results)
    return 0


def cmd_build_patents_fot(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("build_patents_fot", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))

    # Load both extraction.yaml and patents.yaml
    cfg_path = os.path.join(args.config_dir, "extraction.yaml")
    cfg = load_yaml_once(cfg_path)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid config format: {cfg_path}")

    # Merge patents.yaml config (for BigQuery/GCS settings in real mode)
    patents_cfg_path = os.path.join(args.config_dir, "patents.yaml")
    if os.path.exists(patents_cfg_path):
        patents_cfg = load_yaml_once(patents_cfg_path)
        if isinstance(patents_cfg, dict):
            # Merge patents config into main config
            cfg.update(patents_cfg)
            logger.info("Loaded patents configuration from %s", patents_cfg_path)

    from .pipelines.stage3_patents_fot import run as stage3_run
    run_id = new_run_id()
    result = stage3_run(cfg, dry_run=args.dry_run, fast=args.fast, run_id=run_id)
    logger.info("Artifacts: %s", result)
    print(result)
    return 0 if result.get("validation_ok", True) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fot", description="FOT Ontology and Recommender CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("build-static-ontology", help="Build static FOT ontology (BLINK/GAT/parent link)")
    _add_common_args(p1)
    p1.add_argument("--dry-run", action="store_true", help="Run with small fixtures and without external calls")
    p1.add_argument("--fast", action="store_true", help="Reduce work for quick smoke runs")
    p1.add_argument("--blink-mode", choices=["both", "123", "3"], default="both", help="BLINK run mode")
    p1.add_argument("--skip-blink", action="store_true", help="Skip BLINK entity linking if output files exist")
    p1.add_argument("--rename-map", action="store_true", help="Generate docs/ARTIFACTS_RENAME_MAP.md")
    p1.set_defaults(func=cmd_build_static_ontology)

    p2 = sub.add_parser("build-dynamic-ontology", help="Build dynamic FOT ontology (PatentNER data + train)")
    _add_common_args(p2)
    p2.add_argument("--dry-run", action="store_true", help="Run with small fixtures and without external calls")
    p2.add_argument("--fast", action="store_true", help="Reduce work for quick smoke runs")
    p2.set_defaults(func=cmd_build_dynamic_ontology)

    p3 = sub.add_parser("patents-etl", help="Export/download patents and preprocess CSVs")
    _add_common_args(p3)
    p3.set_defaults(func=cmd_patents_etl)

    p4 = sub.add_parser("extract-fot", help="Run FOT extraction over patent titles")
    _add_common_args(p4)
    p4.set_defaults(func=cmd_extract_fot)

    p5 = sub.add_parser("build-exp", help="Build experiment datasets and user simulation")
    _add_common_args(p5)
    p5.set_defaults(func=cmd_build_exp)

    p6 = sub.add_parser("recsys", help="Train/evaluate recommender components")
    _add_common_args(p6)
    p6.set_defaults(func=cmd_recsys)

    p7 = sub.add_parser("build-patents-fot", help="Build patents ETL + FOT extraction (Stage 3)")
    _add_common_args(p7)
    p7.add_argument("--dry-run", action="store_true", help="Run with synthetic CSV and no models")
    p7.add_argument("--fast", action="store_true", help="Reduce work for smoke runs")
    p7.set_defaults(func=cmd_build_patents_fot)

    p8 = sub.add_parser("build-exp-dataset", help="Build experiment dataset and recommender stubs (Stage 4)")
    _add_common_args(p8)
    p8.add_argument("--dry-run", action="store_true", help="Run with minimal stubs")
    p8.add_argument("--fast", action="store_true", help="Reduce work for smoke runs")
    p8.set_defaults(func=cmd_build_exp_dataset)

    p9 = sub.add_parser("run-all", help="Run all stages (1→2→3→4) with a single run_id")
    _add_common_args(p9)
    p9.add_argument("--dry-run", action="store_true", help="Run with stubs and synthetic data")
    p9.add_argument("--fast", action="store_true", help="Faster smoke run")
    p9.set_defaults(func=cmd_run_all)

    p10 = sub.add_parser("ner-comparison", help="Run comprehensive NER model comparison experiments")
    _add_common_args(p10)
    p10.add_argument("--dry-run", action="store_true", help="Run with minimal test data")
    p10.add_argument("--pretrain-model", type=str, help="Path to pretrained PatentNER checkpoint")
    p10.add_argument("--skip-commercial", action="store_true", default=True, help="Skip SpaCy/Stanford NER (default: True)")
    p10.set_defaults(func=cmd_ner_comparison)

    p11 = sub.add_parser("build-hierarchy", help="Build dynamic FOT hierarchy (layer assignment + parent linking)")
    _add_common_args(p11)
    p11.add_argument("--gpu", action="store_true", help="Use GPU acceleration for embedding/clustering")
    p11.set_defaults(func=cmd_build_hierarchy)

    for name, handler in [
        ("data-download", cmd_data_download),
        ("data-process-patent", cmd_data_process_patent),
        ("data-generate-fot", cmd_data_generate_fot),
        ("data-merge", cmd_data_merge),
        ("data-build-exp", cmd_data_build_exp),
        ("data-simulate-history", cmd_data_simulate_history),
    ]:
        p = sub.add_parser(name, help=f"M4 data pipeline step: {name.replace('-', ' ')}")
        _add_common_args(p)
        p.add_argument("--dry-run", action="store_true", help="Write deterministic stubs")
        p.add_argument("--fast", action="store_true", help="Use reduced workloads")
        p.set_defaults(func=handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def cmd_build_exp_dataset(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("build_exp_dataset", level=args.log_level, to_file=args.log_to_file)
    logger.info("Args: %s", vars(args))
    # Load recsys outputs
    rec_path = os.path.join(args.config_dir, "recsys.yaml")
    rec_cfg = load_yaml_once(rec_path)
    # Load Stage 3 outputs to supply inputs
    ext_path = os.path.join(args.config_dir, "extraction.yaml")
    ext_cfg = load_yaml_once(ext_path)
    inputs = {
        "merged_fot_mapping": ext_cfg.get("outputs", {}).get("merged_fot_mapping", "data/processed/new_total_merged_fot_mapping.txt"),
        "merged_patent_fot": ext_cfg.get("outputs", {}).get("merged_patent_fot", "data/processed/new_total_merged_patent_title_with_FOT.txt"),
        "ipc_codes": ext_cfg.get("outputs", {}).get("ipc_codes", "data/processed/ipc_codes.txt"),
    }
    # Normalize to actual paths (they are already relative under data/processed)
    from .pipelines.stage4_exp_build import run as stage4_run
    run_id = new_run_id()
    result = stage4_run(rec_cfg, dry_run=args.dry_run, fast=args.fast, inputs=inputs, run_id=run_id)
    logger.info("Artifacts: %s", result)
    print(result)
    return 0 if result.get("validation_ok", True) else 1


def cmd_run_all(args: argparse.Namespace) -> int:
    os.environ["FOT_CONFIG_DIR"] = args.config_dir
    pathutil.set_config_dir(args.config_dir)
    logger = logutil.get_logger("run_all", level=args.log_level, to_file=args.log_to_file)
    run_id = new_run_id()
    from pathlib import Path

    # Lazy imports to avoid circular deps
    from .pipelines.stage1_static_ontology import run as stage1_run
    from .pipelines.stage2_dynamic_ontology import run as stage2_run
    from .pipelines.stage3_patents_fot import run as stage3_run
    from .pipelines.stage4_exp_build import run as stage4_run

    def load_cfg(name: str):
        return load_yaml_once(os.path.join(args.config_dir, name))

    results = {}
    ok = {"stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0}
    sizes = {}
    try:
        r1 = stage1_run(load_cfg("ontology_static.yaml"), dry_run=args.dry_run, fast=args.fast, generate_rename_map=True, run_id=run_id)
        ok["stage1"] = 1
        results.update({f"stage1.{k}": v for k, v in r1.items()})
        sizes.update({k: v for k, v in r1.items() if k.endswith("_kb")})

        r2 = stage2_run(load_cfg("ontology_dynamic.yaml"), dry_run=args.dry_run, fast=args.fast, run_id=run_id)
        ok["stage2"] = 1
        results.update({f"stage2.{k}": v for k, v in r2.items()})
        sizes.update({k: v for k, v in r2.items() if k.endswith("_kb")})

        r3 = stage3_run(load_cfg("extraction.yaml"), dry_run=args.dry_run, fast=args.fast, run_id=run_id)
        ok["stage3"] = 1 if r3.get("validation_ok", True) else 0
        results.update({f"stage3.{k}": v for k, v in r3.items()})
        sizes.update({k: v for k, v in r3.items() if k.endswith("_kb")})

        ext_cfg = load_cfg("extraction.yaml")
        inputs = {
            "merged_fot_mapping": ext_cfg.get("outputs", {}).get("merged_fot_mapping", "data/processed/new_total_merged_fot_mapping.txt"),
            "merged_patent_fot": ext_cfg.get("outputs", {}).get("merged_patent_fot", "data/processed/new_total_merged_patent_title_with_FOT.txt"),
            "ipc_codes": ext_cfg.get("outputs", {}).get("ipc_codes", "data/processed/ipc_codes.txt"),
        }
        r4 = stage4_run(load_cfg("recsys.yaml"), dry_run=args.dry_run, fast=args.fast, inputs=inputs, run_id=run_id)
        ok["stage4"] = 1 if r4.get("validation_ok", True) else 0
        results.update({f"stage4.{k}": v for k, v in r4.items()})
        sizes.update({k: v for k, v in r4.items() if k.endswith("_kb")})
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)

    # Append RUNS.md
    try:
        from datetime import datetime as _dt
        import json as _json
        p = Path("reports") / "RUNS.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.stat().st_size == 0:
            p.write_text("run_id,utc_time,stage1_ok,stage2_ok,stage3_ok,stage4_ok,sizes_json\n", encoding="utf-8")
        line = f"{run_id},{_dt.utcnow().isoformat()}Z,{ok['stage1']},{ok['stage2']},{ok['stage3']},{ok['stage4']},{_json.dumps(sizes)}\n"
        with p.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        logger.exception("Failed to append RUNS.md")

    out = {"run_id": run_id, **results}
    print(out)
    return 0 if all(ok.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
