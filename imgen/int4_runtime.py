"""Portable SDNQ UINT4 loading for Image21-INT4.

Adapted from ixim/Image21-INT4 scripts/runtime.py and scripts/device.py:
https://huggingface.co/ixim/Image21-INT4

Keep the saved quantization and eager PyTorch dequantization. Small CUDA GPUs
use block/leaf offload, keep VAE encode on the CPU, and borrow the GPU for VAE
decode (it runs after sampling, when the VRAM is idle). MPS and CPU stay
resident. Imports are lazy so the studio and BF16/INT8 paths do not require
SDNQ.
"""

from __future__ import annotations

import os

SMALL_CUDA_BYTES = 10 * 1024**3


def _empty_cache(torch) -> None:
    """Hand cached blocks back when the runtime offers the allocator hook."""
    clear = getattr(getattr(torch, "cuda", None), "empty_cache", None)
    if callable(clear):
        clear()


def choose_runtime(device_type: str, accelerator_bytes: int | None = None) -> dict:
    if device_type not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device type: {device_type}")
    if device_type == "cuda":
        if accelerator_bytes is None or accelerator_bytes <= 0:
            raise ValueError("CUDA memory must be positive")
        mode = "group" if accelerator_bytes <= SMALL_CUDA_BYTES else "model"
    else:
        mode = "resident"
    return {"offload": mode, "dtype": "bfloat16", "use_stream": False}


def quantization_dict(module) -> dict:
    config = getattr(module, "config", None)
    raw = getattr(config, "quantization_config", None)
    if raw is None:
        return {}
    return raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)


def _config_value(value) -> str:
    return str(getattr(value, "value", value)).split(".")[-1].lower()


def _require_int4(module, name: str) -> None:
    quant = quantization_dict(module)
    method = _config_value(quant.get("quant_method", ""))
    dtype = _config_value(quant.get("weights_dtype", ""))
    if method != "sdnq" or dtype != "uint4":
        raise ValueError(f"{name} is not an SDNQ UINT4 checkpoint: {method} {dtype}")
    if quant.get("use_quantized_matmul"):
        raise ValueError(f"{name} enables quantized matmul; portable INT4 requires it off")


def _enable_group(module, device, offload_type: str) -> None:
    import torch
    from diffusers.hooks.group_offloading import apply_group_offloading

    kwargs = {
        "onload_device": device,
        "offload_device": torch.device("cpu"),
        "offload_type": offload_type,
        "use_stream": False,
        "non_blocking": False,
    }
    if offload_type == "block_level":
        kwargs["num_blocks_per_group"] = 1
    if hasattr(module, "enable_group_offload"):
        module.enable_group_offload(**kwargs)
    else:
        apply_group_offloading(module, **kwargs)


def _vae_decode_device(device):
    """The accelerator the VAE may borrow for decoding; CPU stays CPU."""
    import torch

    device = torch.device(device)
    if device.type == "cuda":
        return device
    if device.type == "mps":
        return device
    return torch.device("cpu")


def attach_split_vae(pipe, spec: dict, device) -> None:
    """Keep VAE encode on the CPU, run decode on the accelerator.

    Encode happens *before* sampling, while the transformer still owns the
    GPU, so it stays on the CPU — upstream's 8GB recipe: untiled encode
    activations would compete with the denoiser's cache.

    Decode happens *after* sampling, when the transformer is offloaded back to
    the CPU and the GPU is idle. Keeping it on the CPU is what made a 1024px
    run sit in "decoding" for tens of minutes; moving the VAE over for its turn
    turns that into seconds. If the decode still runs out of VRAM it falls back
    to the CPU for the rest of the process, so a run always finishes.
    """
    import torch

    decode_device = _vae_decode_device(device)
    pipe.vae.to("cpu")
    decode = pipe.vae.decode
    encode_image = pipe._encode_vae_image
    # Latched: once a GPU decode has OOMed there is no point retrying it.
    state = {"device": decode_device.type}

    def decode_on_cpu(latents, *args, **kwargs):
        if torch.is_tensor(latents):
            latents = latents.detach().to("cpu")
        return decode(latents, *args, **kwargs)

    def decode_on_accelerator(latents, *args, **kwargs):
        if state["device"] == "cpu" or not torch.is_tensor(latents):
            return decode_on_cpu(latents, *args, **kwargs)
        try:
            pipe.vae.to(decode_device)
            moved = latents.detach().to(decode_device)
            return decode(moved, *args, **kwargs)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            # Back to the CPU *before* retrying, or the retry decodes on a VAE
            # whose weights are still sitting in the VRAM that just overflowed.
            pipe.vae.to("cpu")
            state["device"] = "cpu"
            spec["vae_decode"] = "cpu"
            print(
                "[imgen] VAE decode ran out of VRAM; decoding on the CPU from now on",
                flush=True,
            )
            _empty_cache(torch)
            return decode_on_cpu(latents, *args, **kwargs)
        finally:
            pipe.vae.to("cpu")
            _empty_cache(torch)

    def encode_on_cpu(image, generator):
        encoded = encode_image(image.detach().to("cpu"), generator)
        return encoded.to(device=image.device, dtype=image.dtype)

    pipe.vae.decode = decode_on_accelerator
    pipe._encode_vae_image = encode_on_cpu
    spec["vae_decode"] = state["device"]


def apply_offload(pipe, spec: dict, device):
    import torch

    mode = spec["offload"]
    if mode == "resident":
        pipe.to(device)
        spec["vae_decode"] = torch.device(device).type
    elif mode == "model":
        if device.type != "cuda":
            raise ValueError("Model CPU offload requires CUDA")
        pipe.enable_model_cpu_offload(gpu_id=device.index or 0)
        spec["vae_decode"] = device.type
    elif mode == "group":
        if device.type != "cuda":
            raise ValueError("Group offload requires CUDA")
        _enable_group(pipe.transformer, device, "block_level")
        _enable_group(pipe.text_encoder, device, "leaf_level")
        attach_split_vae(pipe, spec, device)
    else:
        raise ValueError(f"Unknown offload mode: {mode}")
    pipe.image21_runtime = dict(spec, device=str(device))
    return pipe


def load_int4_pipeline(model, *, device=None, local_files_only=False):
    import torch

    # SDNQ registers its saved-checkpoint loader for both Diffusers and the
    # Qwen3-VL Transformers encoder. Do this before importing the pipeline.
    os.environ.setdefault("DIFFUSERS_SDNQ_TRANSFORMERS", "1")
    try:
        import sdnq  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Image21-INT4 requires SDNQ. Install the INT4 runtime with "
            "`pip install -r requirements-int4.txt` and restart IMGEN."
        ) from exc
    from diffusers import QwenImage21Pipeline

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    memory = (
        torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None
    )
    spec = choose_runtime(device.type, memory)
    dtype_name = os.environ.get("IMAGE21_DTYPE") or spec["dtype"]
    if dtype_name not in {"bfloat16", "float16", "float32"}:
        raise ValueError("IMAGE21_DTYPE must be bfloat16, float16, or float32")
    spec["dtype"] = dtype_name
    pipe = QwenImage21Pipeline.from_pretrained(
        str(model), dtype=getattr(torch, dtype_name), local_files_only=local_files_only
    )
    _require_int4(pipe.transformer, "transformer")
    _require_int4(pipe.text_encoder, "text_encoder")
    if quantization_dict(pipe.vae):
        raise ValueError("VAE is expected to stay in floating point")
    return apply_offload(pipe, spec, device)
