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
    MAX_NUM_IMAGES,
    MAX_REFERENCE_IMAGES,
    MODELS,
)
from .device import probe
from .hub import model_local_path, resolve_snapshot_dir
from .rgba import wrap_rgba_prompt
from .sizes import follow_reference_size, output_resolution_for, size_for

ProgressFn = Callable[[dict], None]


# What Image21-INT8 actually holds once it is on the card. Measured on an idle
# RTX 5090 by loading the resident placement and reading the process's own
# counters before any run: 16.74 GiB allocated, 16.78 GiB reserved, 18.33 GiB
# device-used with the CUDA context. The catalogue's `approx_gb` (18.6) is the
# weight files on disk and lands close; the number this rule used to be built on
# was not that, it was a 29.83 GiB reading taken from a card that had already run
# jobs — allocator cache included, and 13 GiB of it was never weights at all.
INT8_RESIDENT_GB = 16.7

# Working room on top of those weights for the run itself. This is the number
# that decides everything, and until now it had never been measured: 7.0 was
# inherited from the BF16 path's reasoning (its 40 GB threshold minus 33.1 GB of
# weights) and applied to INT8 as if activation cost scaled with weights.
#
# It does not. Measured on the same idle card, one 2K run at 30 steps:
# peak_allocated 43.68 GiB, peak_reserved 49.10 GiB — against a card of
# 31.82 GiB. So a "2K is fine on a 32 GB card" reading was never true in the
# sense it implied: the run only completed because Windows paged the overflow to
# system memory, which is also why its VAE decode took 94 seconds. Subtracting
# the weights leaves ~27 GiB of activations, and that is the figure here.
#
# The headroom scales with the output area. 27.0 GiB is the 2K measurement; a 1K
# run needs about a quarter (≈6.8 GiB), which fits beside the weights on a
# 32 GB card — so 1K stays resident and fast while 2K streams.
INT8_RESIDENT_HEADROOM_GB = 27.0

# The output size the headroom above is quoted for.
INT8_HEADROOM_REFERENCE_PX = 2048


def int8_headroom_gb(output_resolution: int | None) -> float:
    """Working room a run at this output size needs beside the weights."""
    resolution = int(output_resolution or 0)
    if resolution <= 0:
        # No size known (a bare model load) — assume the worst case.
        return INT8_RESIDENT_HEADROOM_GB
    ratio = min(1.0, (resolution / INT8_HEADROOM_REFERENCE_PX) ** 2)
    return INT8_RESIDENT_HEADROOM_GB * ratio

# Conditioning cost is driven by the reference workload, not just the recipe:
# the pipeline resizes every reference to the output-resolution *area*, so a 2K
# edit hands the vision tower about four times the pixels of a 1K one. Measured
# on the INT8 path (2 references, CFG 4.0, weights resident): 1K → 3.1s, 2K →
# over two minutes. The threshold is the geometric midpoint of those two
# points, so the flag errs towards announcing a stage the reader will wait on.
SLOW_CONDITION_MEGAPIXELS = 4.0

# How long a release waits for the run in flight before reporting back. A stop
# request lands at a stage boundary, and nothing inside a stage checks it — a
# single CUDA op cannot be interrupted from Python — so this is a first attempt
# rather than a guarantee. The studio keeps asking until the run really ends.
RELEASE_WAIT_S = 15.0


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
        # Why the last VRAM reading came back empty, and why the last attempt to
        # hand memory back failed. Both used to be swallowed without a word,
        # which turned a device that had run out of room into a machine that
        # simply stopped showing a number — the reading went blank and nothing,
        # anywhere, said why. `None` here means the last attempt worked.
        self._vram_error: str | None = None
        self._reclaim_error: str | None = None
        # The stage the job in flight is inside (None when idle). The studio
        # polls it: a run sitting silently in a minutes-long encode looks
        # exactly like a wedged one unless the stage is reported.
        self._phase: dict[str, Any] | None = None
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
            "phase": self.phase_status(),
            "device": self.device_info,
            # What the card is holding right now, so the studio can show the
            # number a release would change instead of a total from boot.
            "vram": self.vram_usage(),
            # Empty when that reading worked. It is evaluated first on purpose:
            # this describes *that* attempt, not an older one.
            "vram_error": self._vram_error,
            "reclaim_error": self._reclaim_error,
        }

    def phase_status(self) -> dict[str, Any] | None:
        """Which stage the run is in, and how long it has been there.

        The studio polls this for the job it is tracking. "Still in the VAE
        encode for 4 minutes" is a fact the reader can act on; a silent bar is
        not.
        """
        phase = self._phase
        if not phase:
            return None
        return {
            "name": phase["name"],
            "slow": bool(phase["slow"]),
            "elapsed_ms": int((time.time() - phase["since"]) * 1000),
        }

    def _mark_phase(self, name: str, slow: bool) -> None:
        self._phase = {"name": name, "slow": bool(slow), "since": time.time()}

    def _note_sample(self) -> None:
        """Mark the sampling stage once, so a reload sees "sampling" rather
        than the encoding stage it has already left."""
        if not self._phase or self._phase.get("name") != "sample":
            self._mark_phase("sample", False)

    def _clear_phase(self) -> None:
        self._phase = None

    def _check_cancel(self) -> None:
        """Honour a stop request before the next long stretch begins.

        The sampler checks the flag between steps, but the stretches around it
        — conditioning, VAE encode, VAE decode — can each run for minutes on a
        small card. Checking at stage boundaries is what makes "stop" land in
        seconds instead of after the whole run finishes.
        """
        if self._cancel.is_set():
            raise EngineError("Cancelled.", "CANCELLED")

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
        self._free_cuda_cache()

    def vram_usage(self) -> dict[str, Any] | None:
        """How much of the card is in use, or None when there is no CUDA device.

        Device-wide in intent: the question the studio asks is "is the card
        free", and another process holding memory is part of that answer. On
        Windows the OS answers a narrower question than that — the figure is a
        per-process view, measured here at 30.26 GiB free from a second process
        while the studio held 16.7 GiB of the same 31.8 GiB card — so read it as
        "what this process believes it can still get", not as the truth about
        the card.

        A None reading is honest — the number is unknown — but it is not reason
        enough to say nothing. `cudaMemGetInfo` is what fails here, and that is
        also what fails once the card fills up, so the blank readout showed up
        exactly when the reader most needed to see a number. The reason is kept
        in `_vram_error` and travels out through `status()`.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                self._vram_error = "torch.cuda.is_available() is False"
                return None
            free, total = torch.cuda.mem_get_info()
            self._vram_error = None
            return {
                "used_gb": round((total - free) / (1024**3), 2),
                "total_gb": round(total / (1024**3), 2),
            }
        except Exception as exc:
            self._vram_error = f"{type(exc).__name__}: {exc}"
            return None

    def _free_cuda_cache(self) -> None:
        """Hand back whatever the allocator is holding but nothing is using.

        Called after the weights are gone (unload, release) and again just
        before the VAE decode, which is the other moment the card needs room
        it is not using.

        Letting go of the reference is only half of it. A diffusers pipeline is
        a web of reference cycles, so its modules stay alive until the cyclic
        collector runs, and `empty_cache()` only returns blocks the *allocator*
        is still holding — it cannot free memory live tensors are using. Without
        the collection step the cache is emptied and nothing is given back.
        """
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            self._reclaim_error = None
        except Exception as exc:
            # Silence here hid the one case that matters: a device already in
            # trouble, where the reclaim is exactly what would have helped.
            self._reclaim_error = f"{type(exc).__name__}: {exc}"

    def wait_idle(self, timeout: float) -> bool:
        """Wait for the run in flight to finish. False if it is still going."""
        deadline = time.time() + max(0.0, timeout)
        while self._busy and time.time() < deadline:
            time.sleep(0.05)
        return not self._busy

    def release(self, timeout: float | None = None) -> dict[str, Any]:
        """Stop what is running and give the card its memory back.

        Deliberately not the same thing as `cancel()`. Cancelling keeps the
        weights where they are so the next run starts in a second, and frees
        nothing at all — the studio used to announce a release every time it
        cancelled, which was simply untrue. Freeing costs a reload afterwards,
        so it is something the reader asks for.

        The unload only happens once the run has actually ended. Dropping the
        pipeline mid-stage frees nothing anyway — the worker holds its own
        reference and is inside a CUDA op — and it would leave the studio
        reporting a loaded model next to a job that is still going.
        """
        before = self.vram_usage()
        started = time.time()
        if self._busy:
            self.cancel()
        stopped = self.wait_idle(RELEASE_WAIT_S if timeout is None else timeout)
        freed_gb = None
        if stopped:
            self.unload()
            after = self.vram_usage()
            if before and after:
                freed_gb = round(max(0.0, before["used_gb"] - after["used_gb"]), 2)
        phase = self.phase_status()
        return {
            "ok": True,
            "released": stopped,
            "stopping": not stopped,
            "waited_ms": int((time.time() - started) * 1000),
            "freed_gb": freed_gb,
            "vram": self.vram_usage(),
            "phase": (phase or {}).get("name"),
        }

    def load(
        self,
        model_key: str,
        hub: str,
        callback: ProgressFn | None = None,
        output_resolution: int | None = None,
    ) -> dict[str, Any]:
        """Load a pipeline, remembering the failure so the studio can show it.

        Both the generate path and `/api/models/load` come through here, so the
        health flag stays correct whichever one hit the problem.

        `output_resolution` is the run this load is for, when there is one. It
        decides nothing for BF16 or INT4, and for INT8 it decides where the
        weights live: the same card can hold them resident for a 1K run and not
        for a 2K one.
        """
        with self._lock:
            self._last_error = None
        try:
            return self._load(model_key, hub, callback, output_resolution)
        except Exception as exc:
            with self._lock:
                self._last_error = {
                    "model_key": model_key,
                    "hub": hub,
                    "code": getattr(exc, "code", "ENGINE"),
                    "message": str(exc),
                }
            raise

    def _load(
        self,
        model_key: str,
        hub: str,
        callback: ProgressFn | None = None,
        output_resolution: int | None = None,
    ) -> dict[str, Any]:
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
            want_offload = (
                self._should_offload_int8(output_resolution)
                if spec["loader"] == "int8"
                else None
            )
            if (
                self._loaded
                and self._loaded.get("model_key") == model_key
                and self._loaded.get("path") == str(path)
                and self._pipe is not None
            ):
                # For INT8 the placement is part of whether this load answers
                # the question. A pipeline kept resident for a 1K run is the
                # wrong shape for a 2K one, and reusing it as-is is exactly how
                # the decode ran out of room. Same model, different recipe: drop
                # it and load again.
                if (
                    want_offload is None
                    or bool(self._loaded.get("cpu_offload")) == want_offload
                ):
                    return self._loaded
                callback({"type": "load_stage", "stage": "reload"})
                self._pipe = None
                self._loaded = None
                self._free_cuda_cache()
            callback({"type": "load_start", "model_key": model_key, "path": str(path)})
            if spec["loader"] == "int8":
                offload = want_offload
                if offload:
                    self._announce_int8_streaming(callback, output_resolution)
                    callback({"type": "load_stage", "stage": "cpu_offload"})
                pipe = self._load_int8(path, callback, offload=offload)
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

    def _available_gb(self) -> float | None:
        """Free VRAM right now, in GB, or None when it cannot be read."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            free, _total = torch.cuda.mem_get_info()
            return free / (1024**3)
        except Exception:
            return None

    def _should_offload_int8(self, output_resolution: int | None = None) -> bool:
        """INT8 keeps its weights on the card when they fit *beside this run*.

        ``enable_model_cpu_offload()`` hands every module to the accelerator for
        its turn and takes it back afterwards, so a card that can hold the
        weights pays a per-module copy for nothing. Only a card that cannot fit
        them alongside the run's working set streams them. bitsandbytes INT8 is
        CUDA-only, and ``_load`` refuses other devices before reaching here, so
        a non-CUDA answer is never used.

        Two numbers decide it, and the second one is the whole story. The first
        is the weights — ``INT8_RESIDENT_GB``, 16.7 GiB measured. The second is
        the room the run needs beside them, and that is where the old rule went
        wrong in both directions: it guessed the weights at 18.6 GiB (the
        on-disk figure, close enough) but then inherited the *activation* budget
        from the BF16 path's reasoning, 7 GiB, as though activation cost scaled
        with weight size. It does not. A measured 2K run peaks at 43.68 GiB
        allocated and 49.10 GiB reserved against a 31.82 GiB card, and only
        finishes at all because Windows pages the overflow to system memory —
        which is also why its VAE decode took 94 seconds.

        And that floor was compared against the device *total*, a number that is
        only the truth on a card nothing else is using. So a 32 GB card
        concluded "31.82 ≥ 25.6", kept the weights resident, and died in the VAE
        decode at 2K with three minutes of sampling already spent.

        Reading what is actually free is the better question, but it is not a
        complete answer either, and the limits are worth knowing before trusting
        it. On Windows the figure is a per-process view: a second process on
        this machine reads 30.26 GiB free while the studio holds 16.7 GiB of the
        same 31.8 GiB card with nothing running. And it stops answering at all
        once the calling process's own card is full — which is exactly the state
        this gets asked from; see `vram_usage()`. So the total stays as the
        fallback, and it happens to give the same answer as the free reading on
        this card: with the corrected headroom the 2K floor is 43.7 GiB, above
        any 32 GB card either way.
        """
        device = self.device_info.get("device")
        if device != "cuda":
            return False
        free_gb = self._available_gb()
        if free_gb is None:
            # No reading available. Fall back to the device total — what this
            # used to use always. Optimistic, but it is the only number left.
            free_gb = float(self.device_info.get("vram_gb") or 0)
        return free_gb < INT8_RESIDENT_GB + int8_headroom_gb(output_resolution)

    def _announce_int8_streaming(
        self, callback: ProgressFn, output_resolution: int | None
    ) -> None:
        """Say why the weights are being streamed rather than kept in place.

        Without this the studio simply gets slower at the larger sizes for no
        stated reason, and the difference is invisible: same model, same
        settings, one run per module and one run not.
        """
        free_gb = self._available_gb()
        if free_gb is None:
            return
        callback(
            {
                "type": "notice",
                "code": "int8_streamed",
                "free_gb": round(free_gb, 1),
                "resident_gb": INT8_RESIDENT_GB,
                "headroom_gb": round(int8_headroom_gb(output_resolution), 1),
            }
        )

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

    def _load_int8(self, path, callback: ProgressFn, offload: bool = True):
        from .int8_runtime import load_int8_pipeline

        callback({"type": "load_stage", "stage": "int8_sequential"})
        callback(
            {
                "type": "log",
                "message": "Image21-INT8: bitsandbytes INT8 kernels cast bf16 activations to fp16. "
                "This is expected and not an error.",
            }
        )
        callback(
            {
                "type": "log",
                "message": (
                    "Image21-INT8: weights stay resident on the accelerator."
                    if not offload
                    else "Image21-INT8: low-VRAM mode streams each module to the "
                    "accelerator for its turn."
                ),
            }
        )
        return load_int8_pipeline(
            str(path),
            device=self.device_info.get("device") or "cuda",
            offload=offload,
            local_files_only=True,
        )

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
            n_images = max(1, min(MAX_NUM_IMAGES, int(request.get("num_images") or 1)))
            enable_tiling = resolve_vae_tiling(request.get("vae_tiling"))

            if self.demo:
                # Same event shape as a real run so the studio — and its browser
                # tests — exercise one code path: the conditioning and VAE-encode
                # stages land before the first sampler step, the decode after the
                # last one. The short pauses keep each stage observable instead of
                # a single-frame flash.
                self._mark_phase("condition", False)
                callback({"type": "generate_phase", "phase": "condition", "slow": False})
                time.sleep(0.3)
                self._mark_phase("encode", False)
                callback({"type": "generate_phase", "phase": "encode", "slow": False})
                time.sleep(0.3)
                images = [
                    self._demo_image(prompt, width, height, seed + i, mode)
                    for i in range(n_images)
                ]
                for step in range(min(steps, 8)):
                    if self._cancel.is_set():
                        raise EngineError("Cancelled.", "CANCELLED")
                    time.sleep(0.08)
                    self._note_sample()
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
                self._mark_phase("decode", False)
                callback({"type": "generate_phase", "phase": "decode", "slow": False})
                time.sleep(0.5)
            else:
                # The load is told the size this run will ask for: INT8 decides
                # where its weights live from whether they fit beside *this*
                # picture, and 1K and 2K give different answers on one card.
                self.load(
                    model_key,
                    hub,
                    callback=callback,
                    output_resolution=output_resolution,
                )
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
            self._clear_phase()

    def _runtime(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The load record plus the INT4 runtime as it stands right now.

        A decode or encode that ran out of VRAM latches the runtime back to the
        CPU, so the live copy on the pipeline wins over what the load recorded.
        """
        loaded = dict(self._loaded or {})
        runtime = dict(loaded.get("runtime") or {})
        live = getattr(self._pipe, "image21_runtime", None)
        if live:
            runtime.update(live)
        return loaded, runtime

    def _vae_where(self, direction: str) -> str:
        """Where the VAE actually runs for ``encode`` or ``decode`` right now."""
        loaded, runtime = self._runtime()
        explicit = runtime.get(f"vae_{direction}")
        if explicit:
            return str(explicit)
        return str(loaded.get("device") or "cpu")

    def _cpu_vae(self, direction: str) -> bool:
        """Only the small-card INT4 recipe reports where the VAE runs.

        BF16 and INT8 move the whole VAE onto the accelerator when its turn
        comes; their either-direction work is fast.
        """
        loaded, _runtime = self._runtime()
        if loaded.get("loader") != "int4":
            return False
        return self._vae_where(direction) == "cpu"

    def _cpu_decode(self) -> bool:
        """True when decoding this pipeline's VAE will be slow on the CPU.

        INT4 group offload used to pin the VAE to the CPU entirely, which made a
        1024px decode take tens of minutes. It now borrows the accelerator for
        decode and reports where it landed, so the studio's warning follows the
        real path instead of the recipe.
        """
        return self._cpu_vae("decode")

    def _cpu_encode(self) -> bool:
        """Same question for the encode direction (edit runs encode references)."""
        return self._cpu_vae("encode")

    def _slow_condition(
        self, refs: list[Image.Image] | None = None, output_resolution: int = 0
    ) -> bool:
        """True when the conditioning pass is expected to take a long time.

        Two separate things make the prompt + reference encode long, and only
        announcing the first one is what let a two-minute stage look silent:

        * **Streamed weights.** INT4 group offload reads the whole vision tower
          back leaf by leaf; an offloaded INT8 encoder pulls each quantized
          matmul's weight over for its turn. A 2K edit sat here for 19 minutes
          that way. With the weights resident the same pass is seconds.
        * **A large reference workload.** The pipeline resizes every reference
          to the output-resolution area, so 2K is roughly four times the vision
          tokens of 1K — and the cost grows faster than the token count. Two 1K
          references conditioned in 3.1s; the same two at 2K ran past two
          minutes even with the weights resident.

        This used to be ``_group_offload()``, which knew only the INT4 path and
        reported an offloaded INT8 run as fast while it was the one genuinely
        stuck.
        """
        loaded, runtime = self._runtime()
        loader = loaded.get("loader")
        if loader == "int4" and runtime.get("offload") == "group":
            return True
        if loader == "int8" and loaded.get("cpu_offload"):
            return True
        if not refs:
            # A text-only prompt never runs the vision tower.
            return False
        area = max(1, int(output_resolution or 0)) ** 2
        return len(refs) * area >= SLOW_CONDITION_MEGAPIXELS * 1_000_000

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
            self._note_sample()
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

        # Three stretches of a run carry no per-step signal: reading the prompt
        # and the reference images, encoding those references through the VAE,
        # and the VAE decode at the end. Each can take minutes on a small card,
        # and all three used to be silent — the studio sat on a stale label
        # ("loading model") while the console said nothing, which is
        # indistinguishable from a hang. Wrap whichever hooks this pipeline
        # exposes, announce a stage the moment it is really entered, and time it.
        vae = getattr(pipe, "vae", None)
        stages = {
            "condition": {
                "owner": pipe,
                "name": "encode_prompt",
                "label": "prompt and reference encoding",
                "where": str(self.device_info.get("device") or "cpu"),
                "slow": self._slow_condition(images, output_resolution),
            },
            "encode": {
                "owner": pipe,
                "name": "_encode_vae_image",
                "label": "VAE encode",
                "where": self._vae_where("encode"),
                "slow": self._cpu_encode(),
            },
            "decode": {
                "owner": vae,
                "name": "decode",
                "label": "VAE decode",
                "where": self._vae_where("decode"),
                "slow": self._cpu_decode(),
            },
        }

        announced: set[str] = set()
        calls: dict[str, int] = {}
        opened: dict[str, float] = {}
        installed: list[tuple[Any, str, Any, bool]] = []

        def install(spec: dict[str, Any], kind: str) -> None:
            owner = spec["owner"]
            if owner is None:
                return
            original = getattr(owner, spec["name"], None)
            if original is None:
                # Not every pipeline names its conditioning hook the same way;
                # skip the stage rather than guess at a proxy for it.
                return
            # INT4 group offload stores an instance-level callable (its CPU
            # wrapper) while a stock pipeline only has the class method. Both
            # must come back exactly as they were.
            instance_level = spec["name"] in vars(owner)

            def shim(*args, _spec=spec, _kind=kind, _original=original, **kw):
                self._check_cancel()
                if _kind == "decode" and _kind not in announced:
                    # The sampler leaves the allocator holding blocks sized for
                    # its own peak — measured at 49.10 GiB reserved on a
                    # 31.82 GiB card, which means Windows was paging the
                    # overflow to system memory the whole time. The decode wants
                    # a different shape of room, and by this point none of the
                    # sampler's blocks are live. Hand them back first: on a card
                    # this full it is the difference between a render and the
                    # CUDA OOM that fails 0.8s in, after three minutes of
                    # sampling have already been spent on it.
                    self._free_cuda_cache()
                calls[_kind] = calls.get(_kind, 0) + 1
                if _kind not in announced:
                    announced.add(_kind)
                    opened[_kind] = time.time()
                    self._mark_phase(_kind, _spec["slow"])
                    print(f"[imgen] {_spec['label']} started on {_spec['where']}", flush=True)
                    callback(
                        {"type": "generate_phase", "phase": _kind, "slow": _spec["slow"]}
                    )
                try:
                    out = _original(*args, **kw)
                except BaseException:
                    print(
                        f"[imgen] {_spec['label']} failed after "
                        f"{time.time() - opened[_kind]:.1f}s",
                        flush=True,
                    )
                    raise
                extra = f" ({calls[_kind]} calls)" if calls[_kind] > 1 else ""
                print(
                    f"[imgen] {_spec['label']} finished in "
                    f"{time.time() - opened[_kind]:.1f}s{extra}",
                    flush=True,
                )
                return out

            setattr(owner, spec["name"], shim)
            installed.append((owner, spec["name"], original, instance_level))

        for kind, spec in stages.items():
            install(spec, kind)

        self._check_cancel()
        try:
            result = pipe(**kwargs)
        finally:
            for owner, name, original, instance_level in installed:
                if instance_level:
                    setattr(owner, name, original)
                else:
                    # Drop the shadowing instance attribute; the class method
                    # shows through again.
                    vars(owner).pop(name, None)

        # A stop pressed during a stage lands here, instead of being reported as
        # a finished run of work nobody asked for any more.
        self._check_cancel()

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
