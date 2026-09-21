"""Aspect-ratio tables and dimension helpers aligned with Qwen-Image-2.1."""

from __future__ import annotations

from .constants import ASPECT_RATIOS_1K, ASPECT_RATIOS_2K, RESOLUTION_SCALES


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


def follow_reference_size(scale: str, ref_width: int, ref_height: int) -> tuple[int, int]:
    output_resolution = output_resolution_for(scale)
    ratio = ref_width / max(ref_height, 1)
    return calculate_dimensions(output_resolution * output_resolution, ratio)


def catalog() -> dict:
    return {
        "2k": ASPECT_RATIOS_2K,
        "1k": ASPECT_RATIOS_1K,
        "scales": {
            key: {
                "key": spec["key"],
                "label_zh": spec["label_zh"],
                "label_en": spec["label_en"],
                "output_resolution": spec["output_resolution"],
                "sizes": spec["table"],
            }
            for key, spec in RESOLUTION_SCALES.items()
        },
    }
