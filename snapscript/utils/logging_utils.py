from __future__ import annotations

import logging
from typing import Optional


def setup_logging(*, level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    logging.basicConfig(
        level=level,
        format=fmt or "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

