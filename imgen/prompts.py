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
    """Everything the panel needs to draw its preset libraries.

    `generate` and `edit` are the positive-prompt libraries, one per mode, read
    by the presets popover. `negative` is mode-independent: a negative prompt
    says what to steer away from, which reads the same in both modes, so it is
    one flat list of bundles rather than a per-mode tree.
    """
    data = load_presets()
    generate = data.get("generate") or []
    edit = data.get("edit") or []
    negative = data.get("negative") or []
    return {
        "generate": generate,
        "edit": edit,
        "negative": negative,
        "counts": {
            "generate_categories": len(generate),
            "generate_prompts": sum(len(cat.get("prompts") or []) for cat in generate),
            "edit_categories": len(edit),
            "edit_prompts": sum(len(cat.get("prompts") or []) for cat in edit),
            "negative_bundles": len(negative),
            "negative_terms": sum(
                len(bundle.get("terms_zh") or []) + len(bundle.get("terms_en") or [])
                for bundle in negative
            ),
        },
    }
