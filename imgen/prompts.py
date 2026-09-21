"""Load categorized preset prompts shipped with the application."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_FILE = Path(__file__).parent / "data" / "prompts.json"


@lru_cache(maxsize=1)
def load_presets() -> dict[str, Any]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def catalog() -> dict[str, Any]:
    data = load_presets()
    generate = data.get("generate") or []
    edit = data.get("edit") or []
    return {
        "generate": generate,
        "edit": edit,
        "counts": {
            "generate_categories": len(generate),
            "generate_prompts": sum(len(cat.get("prompts") or []) for cat in generate),
            "edit_categories": len(edit),
            "edit_prompts": sum(len(cat.get("prompts") or []) for cat in edit),
        },
    }
