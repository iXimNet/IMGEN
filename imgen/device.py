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
    }
    if torch is None:
        info["warnings"].append("PyTorch is not installed. Follow the README to install a platform wheel.")
        info["recommendations"].append("qwen-image-2.1")
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
                "Less than 12 GiB VRAM. Prefer Image21-INT8, 1K resolution, and CPU offload."
            )
        info["recommendations"].extend(["image21-int8", "qwen-image-2.1"])
    elif mps:
        info["device"] = "mps"
        info["device_name"] = "Apple Silicon (MPS)"
        info["bf16"] = True
        info["int8_ready"] = False
        info["warnings"].append(
            "Image21-INT8 uses bitsandbytes and requires NVIDIA CUDA. Use Qwen-Image-2.1 on macOS."
        )
        info["recommendations"].append("qwen-image-2.1")
    else:
        info["device"] = "cpu"
        info["device_name"] = "CPU"
        info["warnings"].append(
            "No CUDA or MPS device detected. CPU inference is technically possible but extremely slow."
        )
        info["recommendations"].append("qwen-image-2.1")
    return info


def generator_device(preferred: str) -> str:
    """INT8 evaluation used a CPU generator; it is also portable across CUDA/MPS."""
    return "cpu" if preferred in {"cpu", "mps", "cuda"} else "cpu"
