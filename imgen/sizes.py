"""Aspect-ratio tables and dimension helpers aligned with Qwen-Image-2.1."""

from __future__ import annotations

from .constants import ASPECT_RATIOS_1K, ASPECT_RATIOS_2K, DEFAULT_SCALE, RESOLUTION_SCALES


def snap32(value: int) -> int:
    return max(32, int(round(value / 32) * 32))


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    """Match Diffusers `calculate_dimensions` used by QwenImage21Pipeline."""
    width = (target_area * ratio) ** 0.5
    height = width / ratio
    return snap32(width), snap32(height)


def size_for(scale: str, aspect: str) -> tuple[int, int]:
    table = RESOLUTION_SCALES[scale]["table"]
    if aspect not in table:
        raise ValueError(f"Unsupported aspect ratio {aspect!r} for scale {scale!r}")
    return table[aspect]


def output_resolution_for(scale: str) -> int:
    return int(RESOLUTION_SCALES[scale]["output_resolution"])


def follow_reference_size(output_resolution: int, ref_width: int, ref_height: int) -> tuple[int, int]:
    """Reference area is the side length squared, keeping the reference's ratio.

    Takes the resolved area directly (not a scale key) so the studio's readout
    and the engine compute the same size from the same input.
    """
    ratio = ref_width / max(ref_height, 1)
    side = int(output_resolution)
    return calculate_dimensions(side * side, ratio)


def catalog() -> dict:
    return {
        "2k": ASPECT_RATIOS_2K,
        "1k": ASPECT_RATIOS_1K,
        "default_scale": DEFAULT_SCALE,
        "scales": {
            key: {
                "key": spec["key"],
                "label_zh": spec["label_zh"],
                "label_en": spec["label_en"],
                "note_zh": spec["note_zh"],
                "note_en": spec["note_en"],
                "output_resolution": spec["output_resolution"],
                "sizes": spec["table"],
            }
            for key, spec in RESOLUTION_SCALES.items()
        },
    }
