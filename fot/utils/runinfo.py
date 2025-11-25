from __future__ import annotations

import os
import subprocess
from datetime import datetime


def current_git_sha_short() -> str:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        if sha:
            return sha
    except Exception:
        pass
    return "nogit"


def new_run_id() -> str:
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return f"{ts}-{current_git_sha_short()}"

