from __future__ import annotations

import os
from typing import Optional


def truthy_env(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off", "disabled"}


def int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw) if raw else int(default)
    except Exception:
        return int(default)


def float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)


def str_env(name: str, default: str) -> str:
    v = (os.environ.get(name) or "").strip()
    return v if v else default


def opt_str_env(name: str) -> Optional[str]:
    v = (os.environ.get(name) or "").strip()
    return v or None

