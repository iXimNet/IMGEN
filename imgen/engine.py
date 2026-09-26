"""Load pipelines and run text-to-image / multi-reference editing."""

from __future__ import annotations

import inspect
import random
import threading
import time
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from .constants import (
    DEFAULT_CFG,
    DEFAULT_KV_CACHE,
    DEFAULT_SCALE,
    DEFAULT_STEPS,
    MAX_REFERENCE_IMAGES,
    MODELS,
)
from .device import probe
from .hub import model_local_path, resolve_snapshot_dir
from .rgba import wrap_rgba_prompt
from .sizes import follow_reference_size, output_resolution_for, size_for

ProgressFn = Callable[[dict], None]


def resolve_vae_tiling(vae_tiling) -> bool:
    """Tiling is a VRAM tradeoff, never a quality win, so it stays opt-in.

    The Image21-INT8 card traced short vertical stains at 2048px to tiled VAE
    decoding — the same latent decoded untiled was clean — and withdrew its
    earlier general 2048px recommendation. Nothing turns tiling on implicitly
    any more: only an explicit "on" does, whatever the mode or resolution.
    """
    if vae_tiling is None:
        return False
    if isinstance(vae_tiling, str):
        # "auto" is a legacy value from when 2K edits enabled tiling by default.
        return vae_tiling.strip().lower() in {"1", "true", "yes", "on"}
    return bool(vae_tiling)


def apply_vae_tiling(pipe, enabled: bool) -> None:
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    if enabled:
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        else:
            vae.use_tiling = True
        return
    if hasattr(vae, "disable_tiling"):
        vae.disable_tiling()
    elif hasattr(vae, "use_tiling"):
        vae.use_tiling = False


class EngineError(RuntimeError):
    def __init__(self, message: str, code: str = "ENGINE") -> None:
        super().__init__(message)
        self.code = code


class Engine:
    def __init__(self, demo: bool = False) -> None:
        self.demo = demo
        self._lock = threading.Lock()
        self._pipe = None
        self._loaded: dict[str, Any] | None = None
        self._busy = False
        self._cancel = threading.Event()
        # Last load failure, cleared as soon as a load starts. Health is not the
        # same question as "is a pipeline resident" — an idle engine that has
        # never loaded anything is perfectly healthy.
        self._last_error: dict[str, Any] | None = None
        self.device_info = probe()

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def loaded(self) -> dict[str, Any] | None:
        return self._loaded

    def status(self) -> dict[str, Any]:
        return {
            "demo": self.demo,
            "busy": self._busy,
            "loaded": self._loaded,
            "last_error": self._last_error,
            "device": self.device_info,
        }

    def cancel(self) -> None:
        self._cancel.set()
        pipe = self._pipe
        if pipe is not None:
            try:
                pipe._interrupt = True
            except Exception:
                pass

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            self._loaded = None
            self._last_error = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def load(self, model_key: str, hub: str, callback: ProgressFn | None = None) -> dict[str, Any]:
        """Load a pipeline, remembering the failure so the studio can show it.

        Both the generate path and `/api/models/load` come through here, so the
        health flag stays correct whichever one hit the problem.
        """
        with self._lock:
            self._last_error = None
        try:
            return self._load(model_key, hub, callback)
        except Exception as exc:
            with self._lock:
                self._last_error = {
                    "model_key": model_key,
                    "hub": hub,
                    "code": getattr(exc, "code", "ENGINE"),
                    "message": str(exc),
                }
            raise

    def _load(self, model_key: str, hub: str, callback: ProgressFn | None = None) -> dict[str, Any]:
        callback = callback or (lambda _e: None)
        if model_key not in MODELS:
            raise EngineError(f"Unknown model {model_key}", "UNKNOWN_MODEL")
        spec = MODELS[model_key]
        device = self.device_info
        if spec["requires_cuda"] and not device.get("cuda"):
            raise EngineError(
                f"{spec['label']} requires an NVIDIA CUDA GPU and bitsandbytes. "
                "On macOS or CPU-only machines, use Qwen-Image-2.1 or Image21-INT4.",
                "CUDA_REQUIRED",
            )
        if self.demo:
            self._loaded = {
                "model_key": model_key,
                "hub": hub,
                "path": "(demo)",
                "device": device.get("device"),
                "loader": spec["loader"],
            }
            callback({"type": "load_complete", **self._loaded})
            return self._loaded

        path = model_local_path(model_key, hub)
        if path is None:
            from .hub import find_snapshot_dir, missing_weight_files

            found = find_snapshot_dir(model_key, hub)
            if found:
                missing = missing_weight_files(found)
                preview = ", ".join(missing[:4])
                extra = f" (+{len(missing) - 4} more)" if len(missing) > 4 else ""
                raise EngineError(
                    f"{spec['label']} is incomplete: missing {preview}{extra}. "
                    "Open Settings and download again to resume the remaining shards.",
                    "INCOMPLETE_WEIGHTS",
                )
            raise EngineError(
                f"{spec['label']} is not downloaded from {hub} yet.",
                "NOT_DOWNLOADED",
            )
        path = resolve_snapshot_dir(path)
        with self._lock:
            if (
                self._loaded
                and self._loaded.get("model_key") == model_key
                and self._loaded.get("path") == str(path)
                and self._pipe is not None
            ):
                return self._loaded
            callback({"type": "load_start", "model_key": model_key, "path": str(path)})
            if spec["loader"] == "int8":
                pipe = self._load_int8(path, callback)
                offload = True
            elif spec["loader"] == "int4":
                pipe = self._load_int4(path, callback)
                offload = pipe.image21_runtime["offload"] != "resident"
            else:
                pipe = self._load_bf16(path, callback)
                offload = self._should_offload()
                if offload:
                    callback({"type": "load_stage", "stage": "cpu_offload"})
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(device.get("device") or "cpu")
            self._pipe = pipe
            self._loaded = {
                "model_key": model_key,
                "hub": hub,
                "path": str(path),
                "device": device.get("device"),
                "loader": spec["loader"],
                "cpu_offload": offload,
            }
            if spec["loader"] == "int4":
                self._loaded["runtime"] = dict(pipe.image21_runtime)
            callback({"type": "load_complete", **self._loaded})
            return self._loaded

    def _should_offload(self) -> bool:
        device = self.device_info.get("device")
        if device != "cuda":
            return False
        vram = self.device_info.get("vram_gb") or 0
        return vram < 40

    def _load_bf16(self, path, callback: ProgressFn):
        import torch
        from diffusers import QwenImage21Pipeline

        callback({"type": "load_stage", "stage": "pipeline"})
        dtype = torch.bfloat16 if self.device_info.get("bf16") else torch.float32
        try:
            pipe = QwenImage21Pipeline.from_pretrained(str(path), dtype=dtype, local_files_only=True)
        except TypeError:
            pipe = QwenImage21Pipeline.from_pretrained(
                str(path), torch_dtype=dtype, local_files_only=True
            )
        return pipe

    def _load_int8(self, path, callback: ProgressFn):
        from .int8_runtime import load_int8_pipeline

        callback({"type": "load_stage", "stage": "int8_sequential"})
        callback(
            {
                "type": "log",
                "message": "Image21-INT8: bitsandbytes INT8 kernels cast bf16 activations to fp16. "
                "This is expected and not an error.",
            }
        )
        return load_int8_pipeline(str(path), local_files_only=True)

    def _load_int4(self, path, callback: ProgressFn):
        from .int4_runtime import load_int4_pipeline

        callback({"type": "load_stage", "stage": "int4_sdnq"})
        pipe = load_int4_pipeline(
            str(path), device=self.device_info.get("device") or "cpu", local_files_only=True
        )
        runtime = pipe.image21_runtime
        callback({
            "type": "log",
            "message": (
                f"Image21-INT4: SDNQ UINT4 on {runtime['device']}, "
                f"{runtime['offload']} offload, {runtime['dtype']}."
            ),
        })
        return pipe

    def generate(self, request: dict[str, Any], callback: ProgressFn | None = None) -> dict[str, Any]:
        callback = callback or (lambda _e: None)
        if self._busy:
            raise EngineError("A job is already running.", "BUSY")
        self._busy = True
        self._cancel.clear()
        started = time.time()
        try:
            mode = request.get("mode") or "generate"
            model_key = request["model_key"]
            hub = request["hub"]
            refs: list[Image.Image] = list(request.get("images") or [])
            if mode == "edit" and not refs:
                raise EngineError("Image editing needs at least one reference image.", "NEED_REFS")
            if len(refs) > MAX_REFERENCE_IMAGES:
                raise EngineError(
                    f"Qwen-Image-2.1 accepts at most {MAX_REFERENCE_IMAGES} reference images.",
                    "TOO_MANY_REFS",
                )

            prompt = (request.get("prompt") or "").strip()
            if not prompt:
                raise EngineError("Prompt is empty.", "EMPTY_PROMPT")
            if request.get("transparent"):
                prompt = wrap_rgba_prompt(prompt)

            scale = request.get("scale") or DEFAULT_SCALE
            aspect = request.get("aspect") or "1:1"
            # The reference area is its own control; when absent fall back to the
            # scale's table. Read it before the geometry so following the
            # reference and resizing it agree on the same number — the readout
            # in the studio is computed from this input.
            output_resolution = int(
                request.get("output_resolution") or output_resolution_for(scale)
            )
            follow_ref = bool(request.get("follow_ref_aspect")) and mode == "edit" and bool(refs)
            if follow_ref:
                width, height = follow_reference_size(
                    output_resolution, refs[-1].width, refs[-1].height
                )
            elif request.get("width") and request.get("height"):
                width, height = int(request["width"]), int(request["height"])
            else:
                width, height = size_for(scale, aspect)
            steps = int(request.get("steps") or DEFAULT_STEPS)
            cfg = float(request.get("true_cfg_scale") if request.get("true_cfg_scale") is not None else DEFAULT_CFG)
            negative = (request.get("negative_prompt") or "").strip() or None
            seed = request.get("seed")
            if seed is None or int(seed) < 0:
                seed = random.randint(0, 2**31 - 1)
            seed = int(seed)
            kv_cache = (
                bool(request.get("use_kv_cache"))
                if request.get("use_kv_cache") is not None
                else DEFAULT_KV_CACHE
            )
            n_images = max(1, min(4, int(request.get("num_images") or 1)))
            enable_tiling = resolve_vae_tiling(request.get("vae_tiling"))

            if self.demo:
                images = [
                    self._demo_image(prompt, width, height, seed + i, mode)
                    for i in range(n_images)
                ]
                for step in range(min(steps, 8)):
                    if self._cancel.is_set():
                        raise EngineError("Cancelled.", "CANCELLED")
                    time.sleep(0.08)
                    callback(
                        {
                            "type": "generate_progress",
                            "step": step + 1,
                            "total": min(steps, 8),
                        }
                    )
                # The real pipeline announces the VAE decode (see _run_pipe);
                # demo keeps the same event shape so the studio — and its
                # browser tests — exercise one code path. The short pause makes
                # the decode state observable instead of a single-frame flash.
                callback({"type": "generate_phase", "phase": "decode", "slow": False})
                time.sleep(0.5)
            else:
                self.load(model_key, hub, callback=callback)
                pipe = self._pipe
                if pipe is None:
                    raise EngineError("Pipeline failed to load.", "LOAD_FAILED")
                if MODELS[model_key].get("int8"):
                    from .int8_runtime import silence_int8_bf16_cast_warnings

                    silence_int8_bf16_cast_warnings()
                apply_vae_tiling(pipe, enable_tiling)
                callback({"type": "generate_start", "width": width, "height": height, "steps": steps})
                images = self._run_pipe(
                    pipe,
                    prompt=prompt,
                    negative_prompt=negative,
                    images=refs if mode == "edit" else None,
                    width=width,
                    height=height,
                    output_resolution=output_resolution,
                    steps=steps,
                    cfg=cfg,
                    seed=seed,
                    kv_cache=kv_cache,
                    n_images=n_images,
                    follow_ref=follow_ref,
                    callback=callback,
                )

            duration_ms = int((time.time() - started) * 1000)
            return {
                "images": images,
                "prompt": prompt,
                "negative_prompt": negative or "",
                "seed": seed,
                "width": width,
                "height": height,
                "output_resolution": output_resolution,
                "steps": steps,
                "true_cfg_scale": cfg,
                "use_kv_cache": kv_cache,
                "num_images": n_images,
                "scale": scale,
                "aspect": aspect,
                "follow_ref_aspect": follow_ref,
                "transparent": bool(request.get("transparent")),
                "vae_tiling": enable_tiling,
                "duration_ms": duration_ms,
                "mode": mode,
                "model_key": model_key,
                "hub": hub,
                "loaded": self._loaded,
            }
        finally:
            self._busy = False

    def _cpu_decode(self) -> bool:
        """True when the resident pipeline decodes its VAE on the CPU.

        INT4 group offload (small CUDA cards, <=10 GB) keeps the VAE on the CPU
        to protect VRAM, so a 1024px decode there can take minutes. INT8 and
        BF16 model-offload move the whole VAE onto the GPU when its turn comes,
        which is fast; only the INT4 small-card recipe pays the slow path.
        """
        loaded = self._loaded or {}
        if loaded.get("loader") == "int4":
            runtime = loaded.get("runtime") or {}
            return runtime.get("offload") == "group"
        return False

    def _run_pipe(
        self,
        pipe,
        *,
        prompt: str,
        negative_prompt: str | None,
        images: list[Image.Image] | None,
        width: int,
        height: int,
        output_resolution: int,
        steps: int,
        cfg: float,
        seed: int,
        kv_cache: bool,
        n_images: int,
        follow_ref: bool,
        callback: ProgressFn,
    ) -> list[Image.Image]:
        import torch

        sig = inspect.signature(pipe.__call__)
        names = set(sig.parameters)

        def on_step_end(pipe_ref, step, timestep, callback_kwargs):
            if self._cancel.is_set():
                try:
                    pipe_ref._interrupt = True
                except Exception:
                    pass
            callback({"type": "generate_progress", "step": int(step) + 1, "total": steps})
            return callback_kwargs

        generator = torch.Generator(device="cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "true_cfg_scale": cfg,
            "num_images_per_prompt": n_images,
            "generator": generator,
        }
        if negative_prompt and cfg > 1:
            kwargs["negative_prompt"] = negative_prompt
        if images:
            prepared = [img.convert("RGBA") if img.mode != "RGBA" else img for img in images]
            kwargs["image"] = prepared if len(prepared) > 1 else prepared[0]
        if not follow_ref or not images:
            kwargs["width"] = width
            kwargs["height"] = height
        if "output_resolution" in names:
            kwargs["output_resolution"] = output_resolution
        if "use_kv_cache" in names:
            kwargs["use_kv_cache"] = kv_cache
        if "callback_on_step_end" in names:
            kwargs["callback_on_step_end"] = on_step_end

        # The sampler reports per step, but the VAE decode afterwards used to
        # be a silent stretch: the studio froze on "100%, ~0s left" while a
        # small-GPU INT4 run decoded on the CPU for minutes. Wrap the VAE's
        # decode so the phase is announced the moment it is actually entered.
        slow_decode = self._cpu_decode()
        vae = getattr(pipe, "vae", None)
        original_decode = getattr(vae, "decode", None) if vae is not None else None
        # INT4 group offload stores an instance-level CPU-decode wrapper on the
        # VAE (a plain function, not the class method). Remember that so the
        # restore below re-attaches the exact callable that was there before.
        instance_level = vae is not None and "decode" in vars(vae)
        announced = False

        def decode_with_phase(latents, *args, **kw):
            nonlocal announced
            if not announced:
                announced = True
                callback({"type": "generate_phase", "phase": "decode", "slow": slow_decode})
            return original_decode(latents, *args, **kw)

        if vae is not None and original_decode is not None:
            vae.decode = decode_with_phase
        try:
            result = pipe(**kwargs)
        finally:
            if vae is not None and original_decode is not None:
                if instance_level:
                    vae.decode = original_decode
                else:
                    # Drop the shadowing instance attribute; the class method
                    # shows through again.
                    vars(vae).pop("decode", None)

        images_out = list(result.images)
        return images_out

    def _demo_image(self, prompt: str, width: int, height: int, seed: int, mode: str) -> Image.Image:
        width = max(64, min(width, 2048))
        height = max(64, min(height, 2048))
        rng = random.Random(seed)
        image = Image.new("RGBA", (width, height), (18, 18, 22, 255))
        draw = ImageDraw.Draw(image)
        for _ in range(6):
            x0, y0 = rng.randint(-80, width), rng.randint(-80, height)
            x1 = x0 + rng.randint(width // 4, width)
            y1 = y0 + rng.randint(height // 4, height)
            color = (
                rng.randint(40, 90),
                rng.randint(50, 110),
                rng.randint(70, 140),
                70,
            )
            draw.ellipse([x0, y0, x1, y1], fill=color)
        try:
            font = ImageFont.truetype("arial.ttf", size=max(18, width // 28))
            small = ImageFont.truetype("arial.ttf", size=max(14, width // 42))
        except OSError:
            font = ImageFont.load_default()
            small = font
        title = "IMGEN · demo"
        draw.text((48, 48), title, fill=(226, 182, 87, 255), font=font)
        badge = "EDIT" if mode == "edit" else "GENERATE"
        draw.text((48, 48 + max(28, width // 24)), badge, fill=(180, 190, 210, 255), font=small)
        wrapped = prompt[:280]
        draw.text((48, height - 160), wrapped, fill=(244, 241, 234, 230), font=small)
        return image
