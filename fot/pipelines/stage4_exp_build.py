from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from ..utils.logging import get_logger
from ..utils import paths as pathutil
from ..utils.validate import validate_stage4_exp
from ..utils.runinfo import new_run_id
from ..exp.build_dataset import run as build_dataset
from ..exp.user_simulation import run as simulate_users
from ..exp.kg.convert_kgat import run as convert_kg
from ..exp.kg.kgat_runner import run as kgat_run
from ..exp.ablation.runner import run as ablation_run


def run(cfg: Dict[str, Any], *, dry_run: bool = True, fast: bool = True, inputs: Dict[str, str] | None = None, run_id: str | None = None) -> Dict[str, str]:
    logger = get_logger("stage4_exp_build")
    run_id = run_id or new_run_id()
    outputs = cfg.get("outputs", {})

    inputs = inputs or {}

    # 1) Build experiment dataset files
    ds = build_dataset(inputs, cfg)
    # Validate dataset key files
    val = validate_stage4_exp(ds["exp_multi_fots"], ds["exp_fot_mapping"])
    if not val.get("ok"):
        logger.error("Stage4 dataset validation failed: %s", val)
    else:
        logger.info("Stage4 dataset validation passed: %s", val)

    # 2) Simulate users
    fake_interactions = outputs.get("fake_interactions", "data/processed/fake_interactions.csv")
    fake_user_prefs = outputs.get("fake_user_prefs", "data/processed/fake_user_preferences.csv")
    fake_interactions = fake_interactions if fake_interactions.startswith("/") else pathutil.expand("data.processed", Path(fake_interactions).name)
    fake_user_prefs = fake_user_prefs if fake_user_prefs.startswith("/") else pathutil.expand("data.processed", Path(fake_user_prefs).name)
    sim = simulate_users(ds["exp_multi_fots"], ds["exp_multi_fots_ipc_fixed"], fake_interactions, fake_user_prefs)

    # 3) Convert to KGAT inputs
    kg_paths = {k: v for k, v in outputs.items() if k.startswith("kg_")}
    resolved_kg = {}
    for k, v in kg_paths.items():
        if k == "kg_final":
            base = "artifacts.exp"  # store under artifacts/exp
        else:
            base = "artifacts.exp"
        sub = Path(v)
        # Place under artifacts/exp keeping relative under artifacts/exp if present
        rel = sub.relative_to("artifacts/exp").as_posix() if str(sub).startswith("artifacts/exp") else sub.as_posix()
        resolved_kg[k] = pathutil.expand("artifacts.exp", rel.split("/", 1)[-1] if "/" in rel else rel)
    kg_split = (cfg.get("kgat") or {}).get("split_strategy", "last") if isinstance(cfg, dict) else "last"
    kg = convert_kg(
        ds["exp_multi_fots"],
        resolved_kg,
        interactions_path=sim.get("interactions") or sim.get("fake_interactions"),
        fot_mapping_path=ds.get("exp_fot_mapping"),
        ipc_path=ds.get("exp_multi_fots_ipc_fixed"),
        split_strategy=kg_split,
    )

    # 4) KGAT run (stub)
    kgat_model = outputs.get("kgat_model", "artifacts/models/kgat_model_stub.json")
    kgat_model = kgat_model if kgat_model.startswith("/") else pathutil.expand("artifacts.models", Path(kgat_model).name)
    kgat = kgat_run(kg, kgat_model, dry_run=dry_run, fast=fast, run_id=run_id)

    # 5) Ablation report (stub)
    ablation_report = outputs.get("ablation_report", "reports/nn_ablation_results_stub.json")
    ablation_report = ablation_report if ablation_report.startswith("/") else pathutil.expand("reports", Path(ablation_report).name)
    ab = ablation_run(kg, ablation_report, dry_run=dry_run, fast=fast, run_id=run_id)

    # Sizes helper
    def _size_kb(path: str) -> int:
        try:
            return max(1, int((Path(path).stat().st_size + 1023) / 1024))
        except Exception:
            return 0

    result = {
        **ds,
        **sim,
        **kg,
        "kgat_model": kgat,
        "ablation_report": ab,
        "validation_ok": bool(val.get("ok")),
        "run_id": run_id,
    }
    result_with_sizes = dict(result)
    for k, v in result.items():
        result_with_sizes[f"{k}_kb"] = _size_kb(v)
    return result_with_sizes
