"""Detect CUDA / MPS / CPU and produce a UI-friendly hardware report."""

from __future__ import annotations

import platform
import sys
from typing import Any


def _try_torch():
    try:
        import torch  # noqa: F401

        return sys.modules["torch"]
    except Exception:
        return None


# The reference edge is the one setting whose *square* is the workload, and it
# is not a quality dial: the pipeline resizes every reference to `side²` of
# pixels while keeping that reference's own aspect, so the vision tower reads
# `refs × side²` regardless of how big the output is. Measured back to back on a
# 32 GB RTX 5090 with Image21-INT8 resident, two references and the output
# pinned at 2048²: 2048 put the conditioning pass at 737.6s, 1024 finished it in
# 1.5s. Following the last reference also makes this number the canvas area,
# which is the run that left the card reading 31.9/32 GB in use and then died at
# sampling. So the report names the edge the machine in front of it is measured
# to hold — a reading of its own VRAM, never a vendor figure. (The Image21-INT8
# card withdrew its 2048px advice for exactly this reason: the studio endorses
# no size, it just declines to hand a user the one that kills the run.)
REFERENCE_SIDE_ROOMY = 1024
REFERENCE_SIDE_TIGHT = 768
# Where a card stops being able to keep the weights and a 1K run beside them.
REFERENCE_SIDE_ROOMY_VRAM_GB = 24


def reference_side_advice(info: dict[str, Any]) -> dict[str, Any]:
    """The reference edge this machine has the room to run.

    Only a CUDA card gets a say in the number. Too small to keep the weights and
    a 1K run side by side — or too small to read — and the recommendation comes
    down; with no accelerator at all there is nothing to measure, and guessing
    downwards would only make every edit needlessly soft.
    """
    vram = info.get("vram_gb")
    tight = bool(info.get("device") == "cuda" and (not vram or vram < REFERENCE_SIDE_ROOMY_VRAM_GB))
    return {
        "side": REFERENCE_SIDE_TIGHT if tight else REFERENCE_SIDE_ROOMY,
        # Why the number is what it is, so the panel can say it without guessing.
        "bound": "vram" if tight else "default",
        "vram_gb": vram,
    }


def probe() -> dict[str, Any]:
    torch = _try_torch()
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": None,
        "cuda": False,
        "mps": False,
        "device": "cpu",
        "device_name": "CPU",
        "vram_gb": None,
        "bf16": False,
        "int8_ready": False,
        "warnings": [],
        "recommendations": [],
        # Filled on both exits, so a caller never has to ask whether it ran.
        "reference_side": None,
    }
    if torch is None:
        info["warnings"].append("PyTorch is not installed. Follow the README to install a platform wheel.")
        info["recommendations"].append("qwen-image-2.1")
        info["reference_side"] = reference_side_advice(info)
        return info

    info["torch"] = getattr(torch, "__version__", "unknown")
    cuda = bool(torch.cuda.is_available())
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    info["cuda"] = cuda
    info["mps"] = mps

    if cuda:
        info["device"] = "cuda"
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(total / (1024**3), 2)
        except Exception:
            info["device_name"] = "CUDA GPU"
        info["bf16"] = True
        info["int8_ready"] = True
        vram = info["vram_gb"] or 0
        if vram and vram < 12:
            info["warnings"].append(
                "Less than 12 GiB VRAM. Prefer Image21-INT4 at 1K with automatic CPU offload."
            )
        info["recommendations"].extend(["image21-int4", "image21-int8", "qwen-image-2.1"])
    elif mps:
        info["device"] = "mps"
        info["device_name"] = "Apple Silicon (MPS)"
        info["bf16"] = True
        info["int8_ready"] = False
        info["warnings"].append(
            "Image21-INT8 requires NVIDIA CUDA. On macOS use Image21-INT4 or Qwen-Image-2.1."
        )
        info["recommendations"].extend(["image21-int4", "qwen-image-2.1"])
    else:
        info["device"] = "cpu"
        info["device_name"] = "CPU"
        info["warnings"].append(
            "No CUDA or MPS device detected. CPU inference is technically possible but extremely slow."
        )
        info["recommendations"].extend(["image21-int4", "qwen-image-2.1"])
    info["reference_side"] = reference_side_advice(info)
    return info


def generator_device(preferred: str) -> str:
    """INT8 evaluation used a CPU generator; it is also portable across CUDA/MPS."""
    return "cpu" if preferred in {"cpu", "mps", "cuda"} else "cpu"
