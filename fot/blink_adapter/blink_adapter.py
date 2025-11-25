from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

from ..utils.logging import get_logger
import subprocess


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vendor_blink_path() -> Path:
    return _project_root() / "vendor" / "BLINK-main"


def _src_blink_path() -> Path:
    return _project_root() / "src" / "BLINK-main"


def run_blink(
    floors: Dict[str, str],
    out_path: str,
    *,
    mode: str = "123",
    ipc_data_dir: Optional[str] = None,
    faiss_index: Optional[str] = None,
    index_path: Optional[str] = None,
    dry_run: bool = False,
    fast: bool = False,
) -> str:
    """Run BLINK entity linking via a stable adapter.

    This adapter calls the original BLINK scripts which read CSV files directly.
    The scripts have hardcoded paths that need to be set up properly.

    Args:
        floors: Dict mapping level names (l1, l2, l3) to CSV file paths
        out_path: Where to write the final output JSON
        mode: "123" (all levels), "3" (level 3 only), or "both"
        ipc_data_dir: Directory to copy CSV files to (for original script compatibility)
        dry_run: Generate synthetic output instead of running real scripts
        fast: Pass --fast flag to original scripts
    """
    logger = get_logger("blink_adapter")
    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        # Minimal synthetic structure compatible with downstream steps
        sample = []
        if mode in ("123", "both"):
            # emulate multi-level candidates
            for level in [1, 2, 3]:
                sample.append(
                    {
                        "Separator_start": "#" * 300,
                        "IPC_Classification": f"A21C{level:02d}",
                        "IPC_Description": f"Level {level} classification - Bread-making machines",
                        "Texts_Entities": [
                            {
                                "Text": f"Level {level} extracted text",
                                "Biencoder_Recommended_Entities": [
                                    {"Entity_ID": level, "Entity_Name": f"Entity {level}", "Entity_Text": f"Description for entity {level}...", "Entity_URL": f"https://en.wikipedia.org/?curid={level}", "Score": 10.0 - level}
                                ],
                                "Separator_mid": "#" * 300,
                                "Crossencoder_Recommended_Entities": [
                                    {"Entity_ID": level, "Entity_Name": f"Entity {level}", "Entity_Text": f"Description for entity {level}...", "Entity_URL": f"https://en.wikipedia.org/?curid={level}", "Score": 9.9 - level}
                                ],
                            }
                        ],
                        "level": level,
                    }
                )
        elif mode == "3":
            # emulate level-3 only format
            sample.append(
                {
                    "Separator_start": "#" * 300,
                    "IPC_Classification": "A21C03",
                    "IPC_Description": "Level 3 classification - Dough kneading machines",
                    "Texts_Entities": [
                        {
                            "Text": "Dough kneading",
                            "Biencoder_Recommended_Entities": [
                                {"Entity_ID": 3, "Entity_Name": "Kneading", "Entity_Text": "Process of working dough...", "Entity_URL": "https://en.wikipedia.org/?curid=3", "Score": 8.8}
                            ],
                            "Separator_mid": "#" * 300,
                            "Crossencoder_Recommended_Entities": [
                                {"Entity_ID": 3, "Entity_Name": "Kneading", "Entity_Text": "Process of working dough...", "Entity_URL": "https://en.wikipedia.org/?curid=3", "Score": 8.7}
                            ],
                        }
                    ],
                    "level": 3,
                }
            )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=4)
        logger.info("[dry_run] Wrote synthetic BLINK output to %s", out_path)
        return out_path

    # Real run: Setup temporary directory structure for original scripts
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create directory structure that original scripts expect
        sample_dir = temp_path / "Sample"
        sample_dir.mkdir(parents=True)

        # Create models symlink to resolve model file path issues
        _create_models_symlink(temp_path, logger)

        # Copy CSV files to expected locations
        for level_key, csv_path in floors.items():
            if csv_path and Path(csv_path).exists():
                level_num = level_key.replace('l', '')
                dest_path = sample_dir / f"floor_{level_num}.csv"
                shutil.copy2(csv_path, dest_path)
                logger.info(f"Copied {csv_path} to {dest_path}")

        # Create output directory for level 3 script
        if mode in ("3", "both"):
            ipc_lack_dir = sample_dir / "ipc_of_lack"
            ipc_lack_dir.mkdir(parents=True)

        # Determine which scripts to run
        scripts_to_run = []
        if mode == "123":
            scripts_to_run = [("main_dense_123.py", temp_path / "ipc_entities_4.json")]
        elif mode == "3":
            scripts_to_run = [("main_dense_3.py", sample_dir / "ipc_of_lack" / "new_ipc_entities_level_3.json")]
        elif mode == "both":
            scripts_to_run = [
                ("main_dense_123.py", temp_path / "ipc_entities_4.json"),
                ("main_dense_3.py", sample_dir / "ipc_of_lack" / "new_ipc_entities_level_3.json")
            ]

        # Try to run scripts
        results = []
        for script_name, expected_output in scripts_to_run:
            success = False

            # Try environment variable first
            env_key = f"FOT_BLINK_CMD_{script_name.replace('main_dense_', '').replace('.py', '').upper()}"
            cmd_tpl = os.getenv(env_key)

            if cmd_tpl:
                # Use environment variable command template
                cmd = cmd_tpl.format(
                    ipc_path=str(temp_path),
                    output_path=str(expected_output),
                    fast="--fast" if fast else ""
                )
                logger.info(f"Running BLINK via env command: {cmd}")
                try:
                    # Set environment variable for script to find data
                    env = os.environ.copy()
                    env['FOT_IPC_PATH'] = str(temp_path)
                    subprocess.run(cmd, shell=True, check=True, env=env)
                    success = True
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Environment command failed for {script_name}: {e}")

            if not success:
                # Try to find and run original scripts directly
                script_paths = [
                    _src_blink_path() / "blink" / script_name,
                    _vendor_blink_path() / "blink" / script_name,
                    _project_root() / "src" / "BLINK-main" / "blink" / script_name
                ]

                for script_path in script_paths:
                    if script_path.exists():
                        logger.info(f"Found script at {script_path}")

                        # Create a modified version that uses our temp directory
                        modified_script = _create_modified_script(script_path, temp_path, fast)

                        try:
                            env = os.environ.copy()
                            env['PYTHONPATH'] = str(script_path.parent.parent) + ':' + env.get('PYTHONPATH', '')

                            cmd = f"python {modified_script}"
                            logger.info(f"Running modified script: {cmd}")

                            # Set working directory to temp_path where models symlink exists
                            subprocess.run(cmd, shell=True, check=True, env=env, cwd=str(temp_path))
                            success = True
                            break
                        except subprocess.CalledProcessError as e:
                            logger.warning(f"Failed to run {script_path}: {e}")
                            continue

            if success and expected_output.exists():
                results.append(expected_output)
            else:
                raise RuntimeError(f"BLINK script {script_name} failed to produce expected output: {expected_output}")

        # Merge results and write to final output
        _merge_and_write_results(results, out_path, mode)

        logger.info(f"BLINK processing completed, output written to {out_path}")
        return out_path


def _create_modified_script(original_script_path: Path, temp_dir: Path, fast: bool = False) -> Path:
    """Create a modified version of the original script with updated paths and interactive mode."""
    modified_script = temp_dir / f"modified_{original_script_path.name}"

    with open(original_script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace hardcoded paths
    content = content.replace('"/root/autodl-tmp/Sample"', f'"{temp_dir / "Sample"}"')
    content = content.replace('"/root/autodl-tmp/ipc_entities_4.json"', f'"{temp_dir / "ipc_entities_4.json"}"')
    content = content.replace('"/root/autodl-tmp/Sample/ipc_of_lack/new_ipc_entities_level_3.json"', f'"{temp_dir / "Sample" / "ipc_of_lack" / "new_ipc_entities_level_3.json"}"')

    # Enable interactive mode by default to avoid requiring --interactive flag
    content = content.replace(
        'parser.add_argument(\n        "--interactive", "-i", action="store_true", help="Interactive mode."\n    )',
        'parser.add_argument(\n        "--interactive", "-i", action="store_true", default=True, help="Interactive mode."\n    )'
    )

    # Alternative pattern for interactive argument (more robust)
    content = content.replace(
        '"--interactive", "-i", action="store_true", help="Interactive mode."',
        '"--interactive", "-i", action="store_true", default=True, help="Interactive mode."'
    )

    # Add fast mode if needed
    if fast and '--fast' not in content:
        # Add --fast to argument parser defaults
        content = content.replace('action="store_true", help="only biencoder mode"', 'action="store_true", default=True, help="only biencoder mode"')

    # Ensure args.interactive is set to True after argument parsing
    # Insert after args = parser.parse_args()
    if 'args = parser.parse_args()' in content:
        content = content.replace(
            'args = parser.parse_args()',
            'args = parser.parse_args()\n    \n    # Force interactive mode for automated execution\n    args.interactive = True'
        )

    # Force CPU usage instead of CUDA to avoid GPU dependency
    content = content.replace('device="cuda"', 'device="cpu"')
    content = content.replace("device='cuda'", "device='cpu'")

    # Also replace any hardcoded cuda references in _run_crossencoder function
    content = content.replace(
        'def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda"):',
        'def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cpu"):'
    )

    with open(modified_script, 'w', encoding='utf-8') as f:
        f.write(content)

    return modified_script


def _create_models_symlink(temp_path: Path, logger) -> None:
    """Create symlink to models directory to resolve model path issues."""
    models_symlink = temp_path / "models"

    # Try to find models directory in various locations
    possible_model_paths = [
        _src_blink_path() / "models",
        _vendor_blink_path() / "models",
        _project_root() / "src" / "BLINK-main" / "models",
        _project_root() / "vendor" / "BLINK-main" / "models",
        _project_root() / "files",  # Sometimes models are in files directory
    ]

    for model_path in possible_model_paths:
        if model_path.exists():
            # Check if it contains expected model files
            expected_files = [
                "biencoder_wiki_large.json",
                "biencoder_wiki_large.bin",
                "crossencoder_wiki_large.json",
                "crossencoder_wiki_large.bin",
                "entity.jsonl"
            ]

            if any((model_path / f).exists() for f in expected_files):
                try:
                    models_symlink.symlink_to(model_path)
                    logger.info(f"Created models symlink: {models_symlink} -> {model_path}")
                    return
                except OSError as e:
                    logger.warning(f"Failed to create symlink to {model_path}: {e}")
                    continue

    # If no suitable models directory found, create empty directory
    # and warn user about missing models
    models_symlink.mkdir()
    logger.warning(
        "No BLINK models directory found. Created empty models directory. "
        "You may need to download BLINK models or set up model paths correctly."
    )


def _merge_and_write_results(result_files: list, output_path: str, mode: str) -> None:
    """Merge results from multiple BLINK script runs and write to final output."""
    all_data = []

    for result_file in result_files:
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)

    # Write merged results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
        f.write('\n')