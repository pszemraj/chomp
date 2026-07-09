"""Shared readers for test artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON object lines from a test artifact.

    :param Path path: JSONL artifact path.
    :return list[dict[str, Any]]: Decoded rows in file order.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
