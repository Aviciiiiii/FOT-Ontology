from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

_CONFIG_DIR = Path(os.getenv("FOT_CONFIG_DIR", "configs"))
_CACHE: Dict[Path, Dict[str, Any]] = {}


def set_config_dir(config_dir: str | os.PathLike) -> None:
    """Set the directory where YAML configs reside (default: ./configs).

    This can be overridden by setting env var FOT_CONFIG_DIR or via CLI.
    """
    global _CONFIG_DIR
    _CONFIG_DIR = Path(config_dir)


def _paths_yaml() -> Path:
    return (_CONFIG_DIR / "paths.yaml").resolve()


def load_yaml_once(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML file and cache its parsed content by absolute path.

    - Prefers PyYAML when available.
    - Falls back to minimal, opinionated defaults to enable "dry-run" scaffolding
      in environments without third-party deps (e.g., CI sandboxes).
    """
    global _CACHE
    p = Path(path).resolve()
    if p in _CACHE:
        return _CACHE[p]

    # Try PyYAML first
    try:
        import yaml  # type: ignore

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping: {p}")
        _CACHE[p] = data
        return data
    except Exception:
        # Minimal fallback: provide sensible defaults for paths.yaml,
        # and an empty mapping for other config files so code-level defaults apply.
        if p.name == "paths.yaml":
            data = {
                "data": {
                    "raw": "data/raw",
                    "interim": "data/interim",
                    "processed": "data/processed",
                },
                "artifacts": {
                    "models": "artifacts/models",
                    "chunks": "artifacts/chunks",
                    "exp": "artifacts/exp",
                    "logs": "artifacts/logs",
                },
                "reports": "reports",
                "docs": "docs",
            }
            _CACHE[p] = data
            return data
        if p.name == "recsys.yaml":
            # Provide default Stage 4 outputs so downstream steps don't KeyError
            data = {
                "outputs": {
                    "exp_fot_mapping": "data/processed/exp_fot_mapping.txt",
                    "exp_multi_fots": "data/processed/exp_multiple_fots_patents.txt",
                    "exp_multi_fots_ipc_fixed": "data/processed/exp_multiple_fots_patents_ipc_fixed.txt",
                    "fake_interactions": "data/processed/fake_interactions.csv",
                    "fake_user_prefs": "data/processed/fake_user_preferences.csv",
                    "kg_user_list": "artifacts/exp/kg/user_list.txt",
                    "kg_item_list": "artifacts/exp/kg/item_list.txt",
                    "kg_entity_list": "artifacts/exp/kg/entity_list.txt",
                    "kg_relation_list": "artifacts/exp/kg/relation_list.txt",
                    "kg_train": "artifacts/exp/kg/train.txt",
                    "kg_test": "artifacts/exp/kg/test.txt",
                    "kg_final": "artifacts/exp/kg/kg_final.txt",
                    "kgat_model": "artifacts/models/kgat_model_stub.json",
                    "ablation_report": "reports/nn_ablation_results_stub.json",
                },
                "eval": {"topk": [5, 10, 20], "max_neg": 100},
                "kgat": {
                    "embedding_dim": 32,
                    "lr": 0.01,
                    "l2": 0.0001,
                    "kg_weight": 0.05,
                    "kg_batch_size": 128,
                    "epochs": 20,
                    "batch_size": 64,
                    "seed": 42,
                    "kg_alpha": 0.5,
                    "kg_layers": 1,
                    "n_neighbors": 0,
                },
                "mf": {
                    "embedding_dim": 32,
                    "lr": 0.01,
                    "l2": 0.0001,
                    "epochs": 20,
                    "batch_size": 64,
                    "seed": 42,
                },
                "ablation": {
                    "grid": {"model": ["pop", "mf", "kgat"], "dim": [32], "lr": [0.01]},
                    "repeats": 1,
                    "topk": [5, 10, 20],
                    "output_csv": "reports/nn_ablation_results.csv",
                    "output_md": "reports/nn_ablation_results.md",
                    "visualize": False,
                },
            }
            _CACHE[p] = data
            return data
        # For other YAMLs, return empty mapping letting callers use in-code defaults
        _CACHE[p] = {}
        return {}


def _get_from_dot(mapping: Dict[str, Any], dot_key: str) -> Any:
    cur: Any = mapping
    for part in dot_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Key not found in config: {dot_key}")
        cur = cur[part]
    return cur


def get_path(dot_key: str) -> str:
    """Return a path value from configs/paths.yaml via dot notation.

    Example: get_path("artifacts.logs") -> "artifacts/logs"
    """
    cfg = load_yaml_once(_paths_yaml())
    val = _get_from_dot(cfg, dot_key)
    if not isinstance(val, str):
        raise TypeError(f"Config value for '{dot_key}' must be a string path")
    return val


def expand(dot_key: str, *parts: str) -> str:
    """Join base path from config with additional parts and ensure directory exists.

    - If parts represent a file path, ensure parent directory exists.
    - Returns the full path as a string.
    """
    base = Path(get_path(dot_key))
    full = base.joinpath(*parts)
    # Create directory (prefer parent to handle file paths gracefully)
    target_dir = full if full.suffix == "" else full.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return str(full)
