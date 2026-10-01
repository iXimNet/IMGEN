"""FastAPI application: REST + WebSocket studio backend."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import __version__
from .config import ConfigStore
from .constants import (
    APP_NAME,
    DEFAULT_CFG,
    DEFAULT_SCALE,
    DEFAULT_STEPS,
    HUBS,
    MAX_NUM_IMAGES,
    MAX_REFERENCE_IMAGES,
    MAX_UPLOAD_BYTES,
    MODELS,
    NEGATIVE_PROMPT_PLACEHOLDER,
)
from .device import probe
from .engine import Engine, EngineError
from .events import EventBus
from .history import History, new_id
from .hub import (
    describe_weight_dir,
    download_model,
    extra_weight_dirs,
    hub_storage,
    model_status,
    resolve_local_hub,
    set_extra_weight_dirs,
)
from .paths import AppPaths
from .prompts import catalog as prompt_catalog
from .sizes import catalog as size_catalog

STATIC_DIR = Path(__file__).parent / "static"

# A job id is 16 hex characters (`new_id`), and an extra frame appends `_<n>`.
# The output/thumb routes join the id onto a folder, so the shape is checked
# before the join — anything else could climb out of it.
JOB_FILE_ID = re.compile(r"[0-9a-f]{16}(?:_[1-9][0-9]*)?")


def _as_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _read_image(data: bytes, filename: str) -> Image.Image:
    from io import BytesIO

    image = Image.open(BytesIO(data))
    image.load()
    return image


def _public_job(item: dict[str, Any], engine=None) -> dict[str, Any]:
    """Drop local filesystem paths; expose only URLs the browser can fetch.

    The detail view needs the reference-image count, so it is surfaced as
    ``ref_count`` plus ready-made ``ref_urls`` instead of leaking absolute
    paths on disk. Only a 改图 run publishes them — a generate record has no
    reference to show, even if one is sitting in its row (see the gate below).

    A run with ``num_images > 1`` keeps its extra frames in
    ``params.extra_images``, which holds absolute paths too. They become
    ``image_urls`` / ``thumb_urls`` here — one entry per frame, in the order the
    engine produced them — and the paths are dropped, so a single record can
    show everything it made without telling the browser where the files live.
    ``image_url`` / ``thumb_url`` stay as the first frame, which is what the
    older callers assume; ``output_count`` says how many there are.

    A running row also carries the engine's live stage. Only one job runs at a
    time, so the stage belongs to it — and reporting it is what lets the studio
    tell "still encoding, be patient" apart from "nothing is happening".
    """
    refs = item.pop("ref_paths", None) or []
    has_image = bool(item.get("image_path") or item.get("thumb_path"))
    item.pop("image_path", None)
    item.pop("thumb_path", None)
    job_id = item["id"]
    params = item.get("params")
    if not isinstance(params, dict):
        params = {}
        item["params"] = params
    extras = params.pop("extra_images", None) or []
    extra_ids = [
        row.get("id") for row in extras if isinstance(row, dict) and row.get("id")
    ]
    item["image_url"] = f"/api/outputs/{job_id}" if has_image else None
    item["thumb_url"] = f"/api/thumbs/{job_id}" if has_image else None
    item["image_urls"] = ([item["image_url"]] if has_image else []) + [
        f"/api/outputs/{extra_id}" for extra_id in extra_ids
    ]
    item["thumb_urls"] = ([item["thumb_url"]] if has_image else []) + [
        f"/api/thumbs/{extra_id}" for extra_id in extra_ids
    ]
    item["output_count"] = len(item["image_urls"])
    # References are a 改图 concept. A generate run is never guided by one, so it
    # must not publish them even if some row on disk has them stored: records
    # written before the server learned to ignore stray uploads would otherwise
    # keep showing a 参考图 group for a picture it had no part in. The stored
    # paths stay in `ref_paths` for `delete()` to clean up; only the browser-facing
    # view is gated.
    refs_in_use = refs if item.get("mode") == "edit" else []
    item["ref_count"] = len(refs_in_use)
    item["ref_urls"] = [f"/api/refs/{job_id}/{index}" for index in range(len(refs_in_use))]
    if item.get("status") == "running" and engine is not None:
        item["live_phase"] = engine.phase_status()
    return item


def create_app(demo: bool | None = None, home: Path | None = None) -> FastAPI:
    if demo is None:
        demo = os.environ.get("IMGEN_DEMO", "").strip() in {"1", "true", "yes"}

    paths = AppPaths(home)
    paths.ensure()
    config = ConfigStore(paths)
    config.load()
    # Push the stored folders into the lookup layer once at startup, and again
    # whenever the setting changes, so every caller sees the same list.
    set_extra_weight_dirs(config.load().get("extra_weight_dirs"))
    history = History(paths)
    # A previous process may have died with jobs marked "running" (console
    # window closed mid-decode, crash, reboot). Nothing survives the process,
    # so flip those rows to failed before the studio ever reads them.
    history.reconcile_running()
    bus = EventBus()
    engine = Engine(demo=demo)
    download_lock = threading.Lock()
    # Only one OS folder picker may be on screen at a time; Tk roots are not
    # safe to overlap in separate threads.
    pick_dir_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        from .int8_runtime import silence_int8_bf16_cast_warnings

        silence_int8_bf16_cast_warnings()
        bus.bind_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(
        title=APP_NAME,
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1", "http://localhost"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.paths = paths
    app.state.config = config
    app.state.history = history
    app.state.bus = bus
    app.state.engine = engine

    def emit(event: dict[str, Any]) -> None:
        bus.emit(event)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "demo": engine.demo}

    @app.get("/api/bootstrap")
    def bootstrap() -> dict[str, Any]:
        cfg = config.load()
        hub = cfg.get("hub") or "huggingface"
        models = model_status(hub)
        if engine.demo:
            for row in models:
                row["downloaded"] = True
                row["downloaded_any"] = True
                row["available_hubs"] = [hub]
                row["local_hub"] = hub
                row["incomplete"] = False
                row["incomplete_any"] = False
                row["incomplete_hubs"] = []
                row["missing_files"] = []
                row["missing_count"] = 0
                row["external"] = False
                row["path"] = "(demo)"
        return {
            "app": APP_NAME,
            "version": __version__,
            "demo": engine.demo,
            "config": config.public_view(),
            "device": probe(),
            "models": models,
            "hubs": HUBS,
            "model_catalog": MODELS,
            "sizes": size_catalog(),
            "prompts": prompt_catalog(),
            "storage": {
                "hub_dirs": {key: hub_storage(key) for key in HUBS},
                "extra_dirs": [describe_weight_dir(item) for item in extra_weight_dirs()],
                "outputs": str(paths.outputs),
                "home": str(paths.home),
            },
            "engine": engine.status(),
            "defaults": {
                "negative_placeholder": NEGATIVE_PROMPT_PLACEHOLDER,
                "max_reference_images": MAX_REFERENCE_IMAGES,
                # The panel labels these as defaults, so it reads them from the
                # engine's own constants rather than repeating the numbers.
                "steps": DEFAULT_STEPS,
                "cfg": DEFAULT_CFG,
                "max_images": MAX_NUM_IMAGES,
            },
        }

    @app.put("/api/config")
    def update_config(payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "language",
            "hub",
            "model_key",
            "setup_completed",
            "hf_token",
            "ms_token",
            "open_browser",
            "cpu_offload",
            "vae_tiling",
            "last_params",
            "extra_weight_dirs",
        }
        patch = {k: v for k, v in payload.items() if k in allowed}
        if "hf_token" in patch and patch["hf_token"] == "********":
            patch.pop("hf_token")
        if "ms_token" in patch and patch["ms_token"] == "********":
            patch.pop("ms_token")
        saved = config.save(patch)
        # Keep the lookup layer in step with what was just persisted, otherwise
        # the UI would report a folder as searched while nothing reads it.
        if "extra_weight_dirs" in patch:
            set_extra_weight_dirs(saved.get("extra_weight_dirs"))
        return config.public_view()

    @app.post("/api/weights/check-dir")
    def check_weight_dir(payload: dict[str, Any]) -> dict[str, Any]:
        """Classify a folder before it is saved as a weight search path."""
        return describe_weight_dir(payload.get("path") or "")

    @app.post("/api/weights/pick-dir")
    async def pick_weight_dir() -> dict[str, Any]:
        """Open the OS folder picker and describe the chosen folder.

        The dialog blocks its own worker thread, not the event loop, and only
        one may be open at a time. A picker that cannot be opened (headless
        host, no desktop session) or is cancelled answers `ok: false` with a
        reason, and the UI falls back to typing the path in.
        """
        if not pick_dir_lock.acquire(blocking=False):
            raise HTTPException(409, "A folder picker is already open.")
        try:
            language = str(config._data.get("language") or "zh").lower()
            title = "选择权重目录" if language == "zh" else "Choose a weights folder"

            def _pick() -> str | None:
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                except Exception:
                    return None
                try:
                    root = tk.Tk()
                except Exception:
                    return None
                try:
                    root.withdraw()
                    root.attributes("-topmost", True)
                    return filedialog.askdirectory(parent=root, title=title) or ""
                finally:
                    try:
                        root.destroy()
                    except Exception:
                        pass

            folder = await asyncio.to_thread(_pick)
        finally:
            pick_dir_lock.release()
        if folder is None:
            return {"ok": False, "reason": "unavailable", "path": None, "check": None}
        if not folder:
            return {"ok": False, "reason": "cancelled", "path": None, "check": None}
        return {"ok": True, "reason": "ok", "path": folder, "check": describe_weight_dir(folder)}

    @app.post("/api/models/download")
    def api_download(payload: dict[str, Any]) -> dict[str, Any]:
        model_key = payload.get("model_key") or config._data.get("model_key")
        hub = payload.get("hub") or config._data.get("hub") or "huggingface"
        if model_key not in MODELS:
            raise HTTPException(400, f"Unknown model {model_key}")
        if hub not in HUBS:
            raise HTTPException(400, f"Unknown hub {hub}")
        if not download_lock.acquire(blocking=False):
            raise HTTPException(409, "A download is already running.")
        token = config.token_for(hub)
        force = bool(payload.get("force"))

        if engine.demo:
            download_lock.release()
            emit(
                {
                    "type": "download_ok",
                    "model_key": model_key,
                    "hub": hub,
                    "path": "(demo)",
                }
            )
            return {"ok": True, "model_key": model_key, "hub": hub, "demo": True}

        def _run() -> None:
            try:
                path = download_model(model_key, hub, token=token, callback=emit, force=force)
                emit({"type": "download_ok", "model_key": model_key, "hub": hub, "path": str(path)})
            except Exception as exc:
                emit({"type": "error", "stage": "download", "message": str(exc)})
            finally:
                download_lock.release()

        threading.Thread(target=_run, daemon=True, name="imgen-download").start()
        return {"ok": True, "model_key": model_key, "hub": hub}

    @app.post("/api/models/load")
    def api_load(payload: dict[str, Any]) -> dict[str, Any]:
        model_key = payload.get("model_key") or config._data.get("model_key")
        hub = payload.get("hub") or config._data.get("hub") or "huggingface"
        hub = resolve_local_hub(model_key, hub) or hub
        try:
            loaded = engine.load(model_key, hub, callback=emit)
        except EngineError as exc:
            raise HTTPException(409 if exc.code == "BUSY" else 400, str(exc)) from exc
        config.save({"model_key": model_key, "hub": hub})
        return loaded

    @app.post("/api/models/unload")
    def api_unload() -> dict[str, Any]:
        engine.unload()
        return {"ok": True}

    @app.post("/api/engine/release")
    def api_release() -> dict[str, Any]:
        """Stop the run in flight, unload the weights and hand the VRAM back.

        The escape hatch for a run that looks wedged, and the only thing in the
        studio that really frees memory: cancelling a job keeps the weights
        resident so the next run starts in a second, and frees nothing.

        `released` is the honest half of the answer. A stage does not check the
        stop flag while it runs — one CUDA op cannot be interrupted from Python —
        so a request that arrives mid-stage comes back `stopping: true` with
        nothing freed, and the caller is expected to ask again.
        """
        return engine.release()

    @app.post("/api/jobs")
    async def create_job(
        mode: str = Form("generate"),
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        model_key: str = Form(""),
        hub: str = Form(""),
        scale: str = Form(DEFAULT_SCALE),
        aspect: str = Form("1:1"),
        width: int = Form(0),
        height: int = Form(0),
        output_resolution: int = Form(0),
        steps: int = Form(40),
        true_cfg_scale: float = Form(1.0),
        seed: int = Form(-1),
        use_kv_cache: str = Form("true"),
        num_images: int = Form(1),
        transparent: str = Form("false"),
        follow_ref_aspect: str = Form("true"),
        vae_tiling: str = Form("auto"),
        files: list[UploadFile] | None = File(None),
    ) -> dict[str, Any]:
        if engine.busy:
            raise HTTPException(409, "A job is already running.")
        cfg = config.load()
        model_key = model_key or cfg.get("model_key") or "qwen-image-2.1"
        requested_hub = hub or cfg.get("hub") or "huggingface"
        # The weights may already be on disk from the other source. Use them
        # rather than refusing to run while the file sits right there.
        hub = resolve_local_hub(model_key, requested_hub) or requested_hub
        # A generate run has no reference images, whatever the request carries.
        # The engine already ignores them (`images=refs if mode == "edit" else
        # None`), but the upload loop below would still decode them and the run
        # would save them — producing a record whose detail overlay lists a 参考图
        # that never influenced the picture. The client no longer sends them in
        # this mode; this is the second line, so a direct API call cannot create
        # such a record either. They are not even decoded: there is no point
        # validating a file the run will not read.
        refs: list[Image.Image] = []
        if mode == "edit":
            for upload in (files or [])[:MAX_REFERENCE_IMAGES]:
                data = await upload.read()
                if not data:
                    continue
                if len(data) > MAX_UPLOAD_BYTES:
                    raise HTTPException(400, f"{upload.filename} exceeds the 25 MB limit.")
                try:
                    refs.append(_read_image(data, upload.filename or "image.png"))
                except Exception as exc:
                    raise HTTPException(400, f"Could not read {upload.filename}: {exc}") from exc

        job_id = new_id()
        request = {
            "mode": mode,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_key": model_key,
            "hub": hub,
            "scale": scale,
            "aspect": aspect,
            "width": width or None,
            "height": height or None,
            "output_resolution": output_resolution or None,
            "steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "seed": seed,
            "use_kv_cache": _as_bool(use_kv_cache, True),
            "num_images": num_images,
            "transparent": _as_bool(transparent, False),
            "follow_ref_aspect": _as_bool(follow_ref_aspect, True),
            "vae_tiling": vae_tiling,
            "images": refs,
        }
        history.create(
            {
                "id": job_id,
                "mode": mode,
                "model_key": model_key,
                "hub": hub,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "params": {k: v for k, v in request.items() if k != "images"},
                "seed": seed if seed >= 0 else None,
                "status": "running",
            }
        )
        emit({"type": "job_queued", "id": job_id})
        if hub != requested_hub:
            emit(
                {
                    "type": "log",
                    "job_id": job_id,
                    "message": f"{model_key}: using weights from {hub} "
                    f"({requested_hub} has no local copy).",
                }
            )

        def _run() -> None:
            try:
                result = engine.generate(request, callback=lambda e: emit({**e, "job_id": job_id}))
                images = result.pop("images")
                image_path, thumb_path = history.save_image(job_id, images[0])
                extra_paths = []
                for index, extra in enumerate(images[1:], start=1):
                    extra_id = f"{job_id}_{index}"
                    p, t = history.save_image(extra_id, extra)
                    extra_paths.append({"id": extra_id, "image_path": p, "thumb_path": t})
                ref_paths = history.save_refs(job_id, refs) if refs else []
                history.update(
                    job_id,
                    status="succeeded",
                    prompt=result["prompt"],
                    negative_prompt=result["negative_prompt"],
                    seed=result["seed"],
                    width=result["width"],
                    height=result["height"],
                    duration_ms=result["duration_ms"],
                    image_path=image_path,
                    thumb_path=thumb_path,
                    ref_paths=ref_paths,
                    params={
                        **{k: v for k, v in result.items() if k not in {"loaded", "images"}},
                        "extra_images": extra_paths,
                    },
                )
                emit(
                    {
                        "type": "job_complete",
                        "id": job_id,
                        "thumb": f"/api/thumbs/{job_id}",
                        "image": f"/api/outputs/{job_id}",
                    }
                )
            except EngineError as exc:
                if exc.code == "CANCELLED":
                    # A stop request is not a failure. The stage boundaries now
                    # honour the flag, so the run really ends here — report the
                    # state the user asked for instead of a red error toast.
                    history.update(job_id, status="cancelled", error=str(exc))
                    emit({"type": "job_cancelled", "id": job_id})
                else:
                    history.update(job_id, status="failed", error=str(exc))
                    emit(
                        {
                            "type": "error",
                            "stage": "generate",
                            "job_id": job_id,
                            "message": str(exc),
                            "code": exc.code,
                        }
                    )
            except Exception as exc:
                history.update(job_id, status="failed", error=str(exc))
                emit({"type": "error", "stage": "generate", "job_id": job_id, "message": str(exc)})

        threading.Thread(target=_run, daemon=True, name=f"imgen-job-{job_id}").start()
        return {"id": job_id, "status": "running"}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        engine.cancel()
        history.update(job_id, status="cancelled", error="Cancelled by user.")
        emit({"type": "job_cancelled", "id": job_id})
        return {"ok": True}

    @app.get("/api/jobs")
    def list_jobs(limit: int = 80, offset: int = 0, q: str = "") -> dict[str, Any]:
        items = [
            _public_job(item, engine)
            for item in history.list(limit=limit, offset=offset, query=q)
        ]
        # `total` lets the browser know whether another page exists without
        # guessing from the page size.
        return {"items": items, "total": history.count(query=q), "offset": offset, "limit": limit}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        item = history.get(job_id)
        if not item:
            raise HTTPException(404, "Job not found.")
        return _public_job(item, engine)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, Any]:
        if not history.delete(job_id):
            raise HTTPException(404, "Job not found.")
        return {"ok": True}

    def _file_response(kind: str, job_id: str) -> FileResponse:
        item = history.get(job_id)
        if not item:
            # An extra frame is saved as `{job_id}_{n}` and has no row of its
            # own. Only that shape may reach the filesystem: a name is joined
            # onto the outputs folder, so anything else (`..`, a drive letter,
            # an absolute path) must be refused before the join, not after.
            if not JOB_FILE_ID.fullmatch(job_id):
                raise HTTPException(404, "Not found.")
            path = paths.outputs / f"{job_id}.png" if kind == "outputs" else paths.thumbs / f"{job_id}.jpg"
            if path.exists():
                return FileResponse(path)
            raise HTTPException(404, "Not found.")
        path = item.get("image_path") if kind == "outputs" else item.get("thumb_path")
        if not path or not Path(path).exists():
            raise HTTPException(404, "File missing.")
        return FileResponse(path)

    @app.get("/api/outputs/{job_id}")
    def get_output(job_id: str) -> FileResponse:
        return _file_response("outputs", job_id)

    @app.get("/api/thumbs/{job_id}")
    def get_thumb(job_id: str) -> FileResponse:
        return _file_response("thumbs", job_id)

    @app.get("/api/refs/{job_id}/{index}")
    def get_ref(job_id: str, index: int) -> FileResponse:
        item = history.get(job_id)
        if not item:
            raise HTTPException(404, "Job not found.")
        refs = item.get("ref_paths") or []
        if index < 0 or index >= len(refs):
            raise HTTPException(404, "Reference not found.")
        path = Path(refs[index])
        if not path.exists():
            raise HTTPException(404, "File missing.")
        return FileResponse(path)

    @app.post("/api/reveal")
    def reveal(payload: dict[str, Any]) -> dict[str, Any]:
        target = payload.get("path") or str(paths.outputs)
        folder = Path(target)
        if folder.is_file():
            folder = folder.parent
        if not folder.exists():
            raise HTTPException(404, "Path does not exist.")
        try:
            if sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc
        return {"ok": True, "path": str(folder)}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        queue = bus.subscribe()
        try:
            await ws.send_json({"type": "hello", "version": __version__, "demo": engine.demo})
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20)
                    await ws.send_json(event)
                except asyncio.TimeoutError:
                    await ws.send_json({"type": "ping"})
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            bus.unsubscribe(queue)

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app
