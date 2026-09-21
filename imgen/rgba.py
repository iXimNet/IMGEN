"""Official transparent-image prompt wrapper from Qwen-Image-2.1."""

from __future__ import annotations

from .constants import RGBA_PREFIX, RGBA_SUFFIX


def is_rgba_prompt(prompt: str) -> bool:
    text = prompt.strip()
    return RGBA_PREFIX in text and "transparent" in text.lower()


def wrap_rgba_prompt(prompt: str) -> str:
    """Apply the official RGBA template if the user has not already used it."""
    text = prompt.strip()
    if not text:
        return text
    if is_rgba_prompt(text):
        return text
    body = text.rstrip(".")
    return f"{RGBA_PREFIX} {body}. {RGBA_SUFFIX}"
