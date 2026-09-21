"""FastAPI application: REST + WebSocket studio backend."""

from __future__ import annotations

import asyncio
import os
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
    HUBS,
    MAX_REFERENCE_IMAGES,
    MAX_UPLOAD_BYTES,
    MODELS,
    NEGATIVE_PROMPT_PLACEHOLDER,
)
from .device import probe
from .engine import Engine, EngineError
from .events import EventBus
from .history import History, new_id
from .hub import download_model, model_status
from .paths import AppPaths
from .prompts import catalog as prompt_catalog
from .sizes import catalog as size_catalog

STATIC_DIR = Path(__file__).parent / "static"


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


def create_app(demo: bool | None = None, home: Path | None = None) -> FastAPI:
    if demo is None:
        demo = os.environ.get("IMGEN_DEMO", "").strip() in {"1", "true", "yes"}

    paths = AppPaths(home)
    paths.ensure()
    config = ConfigStore(paths)
    config.load()
    history = History(paths)
    bus = EventBus()
    engine = Engine(demo=demo)
    download_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
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
            "engine": engine.status(),
            "defaults": {
                "negative_placeholder": NEGATIVE_PROMPT_PLACEHOLDER,
                "max_reference_images": MAX_REFERENCE_IMAGES,
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
        }
        patch = {k: v for k, v in payload.items() if k in allowed}
        if "hf_token" in patch and patch["hf_token"] == "********":
            patch.pop("hf_token")
        if "ms_token" in patch and patch["ms_token"] == "********":
            patch.pop("ms_token")
        config.save(patch)
        return config.public_view()

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
                path = download_model(model_key, hub, token=token, callback=emit)
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

    @app.post("/api/jobs")
    async def create_job(
        mode: str = Form("generate"),
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        model_key: str = Form(""),
        hub: str = Form(""),
        scale: str = Form("2k"),
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
        hub = hub or cfg.get("hub") or "huggingface"
        refs: list[Image.Image] = []
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
                history.update(job_id, status="failed", error=str(exc))
                emit({"type": "error", "stage": "generate", "job_id": job_id, "message": str(exc), "code": exc.code})
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
        items = history.list(limit=limit, offset=offset, query=q)
        for item in items:
            item["image_url"] = f"/api/outputs/{item['id']}" if item.get("image_path") else None
            item["thumb_url"] = f"/api/thumbs/{item['id']}" if item.get("thumb_path") else None
        return {"items": items}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        item = history.get(job_id)
        if not item:
            raise HTTPException(404, "Job not found.")
        item["image_url"] = f"/api/outputs/{item['id']}" if item.get("image_path") else None
        item["thumb_url"] = f"/api/thumbs/{item['id']}" if item.get("thumb_path") else None
        return item

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, Any]:
        if not history.delete(job_id):
            raise HTTPException(404, "Job not found.")
        return {"ok": True}

    def _file_response(kind: str, job_id: str) -> FileResponse:
        item = history.get(job_id)
        if not item:
            # extra images saved as {id}_{n}
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
