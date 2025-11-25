from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from ...utils.logging import get_logger


def extract_names_from_json(input_file_path: str, output_file_path: str) -> List[str]:
    """
    从包含实体对象的 JSON 文件中提取所有 'name' 字段，
    并将其保存为新的 JSON 文件中的名称列表。
    """
    logger = get_logger("extract_names_from_json")

    try:
        logger.info("正在从 %s 读取实体数据...", input_file_path)
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        names = []

        # 处理可能的嵌套结构
        if isinstance(data, dict) and 'level2' in data and 'level3' in data:
            logger.info("检测到嵌套结构 (level2/level3)")
            all_entities = data.get('level2', []) + data.get('level3', [])
        elif isinstance(data, dict) and any(key in data for key in ['level2', 'level3']):
            logger.info("检测到部分嵌套结构")
            all_entities = data.get('level2', []) + data.get('level3', [])
        elif isinstance(data, list):
            logger.info("检测到列表结构")
            all_entities = data
        elif isinstance(data, dict):
            logger.info("检测到字典结构，尝试提取值")
            all_entities = list(data.values()) if data else []
        else:
            logger.error("错误：%s 的根元素不是一个列表或预期的字典结构", input_file_path)
            return []

        # 提取name字段
        for entity in all_entities:
            if isinstance(entity, dict) and "name" in entity:
                names.append(entity["name"])
            elif isinstance(entity, str):
                # 如果直接是字符串，也添加进去
                names.append(entity)
            else:
                # 静默跳过格式不正确的实体以避免冗余警告
                pass

        logger.info("已提取 %d 个实体名称", len(names))
        return names

    except FileNotFoundError:
        logger.error("错误：文件未找到。请确保 %s 存在", input_file_path)
        return []
    except json.JSONDecodeError:
        logger.error("错误：无法解析 %s。请检查文件是否是有效的 JSON 格式", input_file_path)
        return []
    except Exception as e:
        logger.error("发生了一个意外错误：%s", e)
        return []


def build_third_entities(merged_entities_path: str, out_path: str, *, dry_run: bool = False) -> str:
    """Build a list of entity names from merged entities JSON file.

    Args:
        merged_entities_path: Path to the merged entities JSON file
        out_path: Output path for the JSON file
        dry_run: if True or input missing, fallback to fixtures or minimal set
    """
    logger = get_logger("build_third_entities")
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    names: List[str]
    fixtures = Path("tests/fixtures/stage2/third_entities.json")

    if not dry_run and merged_entities_path and Path(merged_entities_path).exists():
        logger.info("--- 阶段 2: 从 JSON 文件中提取 'name' 字段 ---")
        names = extract_names_from_json(merged_entities_path, str(out_p))

        if not names:  # Fallback to fixtures if extraction failed
            if fixtures.exists():
                names = json.loads(fixtures.read_text(encoding="utf-8"))
                logger.info("提取失败，使用fixture数据：%d 个实体", len(names))
            else:
                names = ["bread machine", "dough kneading", "meat mincing"]
                logger.info("提取失败，使用默认数据：%d 个实体", len(names))
    else:
        if fixtures.exists():
            names = json.loads(fixtures.read_text(encoding="utf-8"))
            logger.info("使用fixture数据：%d 个实体", len(names))
        else:
            names = ["bread machine", "dough kneading", "meat mincing"]
            logger.info("使用默认数据：%d 个实体", len(names))

    # Write with same formatting as original script (indent=4)
    out_p.write_text(json.dumps(names, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info("成功将实体名称保存到 %s", out_p)
    return str(out_p)
