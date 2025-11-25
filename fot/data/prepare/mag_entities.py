from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

from ...utils.logging import get_logger


def extract_mag_entity_names(mag_file_path: str) -> List[str]:
    """
    从 .nt 文件中提取所有 MAG 实体的名称。
    """
    titles = []
    logger = get_logger("extract_mag_entity_names")
    logger.info("正在从 %s 中提取 MAG 实体名称...", mag_file_path)

    try:
        with open(mag_file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line_num % 1000000 == 0:  # Progress logging for large files
                    logger.info("已处理 %d 行，已提取 %d 个实体", line_num, len(titles))

                components = line.strip().split(' ')
                if len(components) < 4:
                    continue

                predicate = components[1]
                obj = ' '.join(components[2:])

                if '<http://xmlns.com/foaf/0.1/name>' in predicate:
                    try:
                        # Extract content within the first double quotes
                        title_value = obj.split('"')[1]
                        titles.append(title_value)
                    except IndexError:
                        # Skip malformed lines, e.g., if no closing quote
                        continue

        logger.info("已提取 %d 个实体名称", len(titles))
        return titles

    except FileNotFoundError:
        logger.error("错误：文件未找到。请确保 %s 存在", mag_file_path)
        return []
    except Exception as e:
        logger.error("读取 %s 时发生错误：%s", mag_file_path, e)
        return []


def select_and_save_random_entities(input_file_path: str, output_file_path: str, num_to_select: int, seed: int = 42) -> List[str]:
    """
    从输入文件中提取实体名称，随机选取指定数量的实体。

    Args:
        input_file_path: Path to input .nt file
        output_file_path: Path to output JSON file
        num_to_select: Number of entities to sample
        seed: Random seed for reproducibility (default: 42)
    """
    logger = get_logger("select_and_save_random_entities")
    all_entities = extract_mag_entity_names(input_file_path)

    if not all_entities:
        logger.warning("没有实体可供选择")
        return []

    if len(all_entities) < num_to_select:
        logger.warning("警告：文件中只有 %d 个实体，不足以选取 %d 个。将选取所有可用实体",
                      len(all_entities), num_to_select)
        selected_entities = all_entities
    else:
        # Set random seed for reproducibility
        random.seed(seed)
        logger.info("正在从 %d 个实体中随机选取 %d 个 (seed=%d)...", len(all_entities), num_to_select, seed)
        selected_entities = random.sample(all_entities, num_to_select)

    return selected_entities


def build_mag_entities(nt_path: str, out_path: str, *, dry_run: bool = False, num_entities: int = 13560,
                       force_regenerate: bool = False, seed: int = 42) -> str:
    """Build a list of MAG entity names from .nt file.

    Args:
        nt_path: Path to the .nt file containing MAG entities
        out_path: Output path for the JSON file
        dry_run: if True or input missing, fallback to fixtures or minimal set
        num_entities: Number of entities to randomly sample (default: 13560)
        force_regenerate: If True, regenerate even if output file exists (default: False)
        seed: Random seed for reproducible sampling (default: 42)

    Returns:
        Path to the output JSON file
    """
    logger = get_logger("build_mag_entities")
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Check if output file already exists
    if out_p.exists() and not force_regenerate:
        logger.info("MAG实体文件已存在: %s", out_p)
        logger.info("跳过生成，直接使用已有文件（使用 force_regenerate=True 强制重新生成）")

        # Load and verify existing file
        try:
            existing_entities = json.loads(out_p.read_text(encoding="utf-8"))
            logger.info("已加载 %d 个现有实体", len(existing_entities))
            return str(out_p)
        except Exception as e:
            logger.warning("加载现有文件失败: %s，将重新生成", e)
            # Continue to regeneration

    if force_regenerate and out_p.exists():
        logger.info("强制重新生成模式：将覆盖现有文件 %s", out_p)

    names: List[str]
    fixtures = Path("tests/fixtures/stage2/mag_entities.json")

    if not dry_run and nt_path and Path(nt_path).exists():
        logger.info("--- 阶段 1: 从 .nt 文件中提取并抽样实体名称 ---")
        names = select_and_save_random_entities(nt_path, str(out_p), num_entities, seed=seed)

        if not names:  # Fallback to fixtures if extraction failed
            if fixtures.exists():
                names = json.loads(fixtures.read_text(encoding="utf-8"))
                logger.info("提取失败，使用fixture数据：%d 个实体", len(names))
            else:
                names = ["machine learning", "data mining", "graph neural network"]
                logger.info("提取失败，使用默认数据：%d 个实体", len(names))
    else:
        if fixtures.exists():
            names = json.loads(fixtures.read_text(encoding="utf-8"))
            logger.info("使用fixture数据：%d 个实体", len(names))
        else:
            names = ["machine learning", "data mining", "graph neural network"]
            logger.info("使用默认数据：%d 个实体", len(names))

    # Write with same formatting as original script (indent=4)
    out_p.write_text(json.dumps(names, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info("已将 %d 个随机选取的实体保存到 %s 中 (seed=%d)", len(names), out_p, seed)
    return str(out_p)
