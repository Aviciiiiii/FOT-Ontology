from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict


def tsv_has_header(path: str, expected: List[str]) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        first = p.read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return False
    cols = first.split("\t")
    return cols == expected


def json_is_list(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return isinstance(data, list)
    except Exception:
        return False


_SPAN_RE = re.compile(r"^\s*\d+:\d+:\d+(\s*,\s*\d+:\d+:\d+)*\s*$")


def fot_span_field_ok(span: str) -> bool:
    if span is None:
        return False
    span = span.strip()
    if span == "":
        return True
    return bool(_SPAN_RE.match(span))


def validate_stage3_outputs(titles: str, pubnums: str, ipcs: str) -> Dict[str, bool]:
    ok_titles = tsv_has_header(titles, ["patent_id", "title"]) or tsv_has_header(
        titles, ["patentid", "title"]
    )
    ok_pub = tsv_has_header(pubnums, ["patent_id", "publication_number"]) or tsv_has_header(
        pubnums, ["patentid", "publication_number"]
    )
    ok_ipc = tsv_has_header(ipcs, ["patent_id", "ipc_code"]) or tsv_has_header(ipcs, ["patentid", "ipc_code"])
    return {"titles": ok_titles, "pubnums": ok_pub, "ipcs": ok_ipc, "ok": ok_titles and ok_pub and ok_ipc}


def validate_stage4_exp(exp_multi: str, exp_map: str) -> Dict[str, bool]:
    ok_multi = tsv_has_header(exp_multi, ["patentid", "title", "FOT"]) and _exp_fot_field_ok(exp_multi)
    ok_map = tsv_has_header(exp_map, ["id", "name", "ipc_code"]) or tsv_has_header(exp_map, ["fot_id", "fot_name", "ipc_code"]) 
    return {"exp_multi": ok_multi, "exp_map": ok_map, "ok": ok_multi and ok_map}


def _exp_fot_field_ok(path: str) -> bool:
    try:
        for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
            if i == 0:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            if not fot_span_field_ok(parts[2]):
                return False
            if i > 20:
                break
        return True
    except Exception:
        return False

