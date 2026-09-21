"""Download Qwen-Image-2.1 / Image21-INT8 from Hugging Face or ModelScope."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

from .constants import MODELS, repo_id

ProgressFn = Callable[[dict], None]


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _looks_complete(path: Path) -> bool:
    if not path.exists():
        return False
    if (path / "model_index.json").exists():
        return True
    # ModelScope sometimes nests the snapshot one level deeper.
    for child in path.iterdir() if path.is_dir() else []:
        if child.is_dir() and (child / "model_index.json").exists():
            return True
    return False


def resolve_snapshot_dir(path: Path) -> Path:
    if (path / "model_index.json").exists():
        return path
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / "model_index.json").exists():
                return child
    return path


def model_local_path(model_key: str, hub: str, cache_root: Path | None = None) -> Path | None:
    """Best-effort lookup of an already-downloaded snapshot."""
    repo = repo_id(model_key, hub)
    candidates: list[Path] = []
    if hub == "huggingface":
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub_dir = hf_home / "hub" / ("models--" + repo.replace("/", "--"))
        snapshots = hub_dir / "snapshots"
        if snapshots.exists():
            for snap in sorted(snapshots.iterdir(), reverse=True):
                candidates.append(snap)
        candidates.append(hub_dir)
    else:
        ms_home = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
        candidates.extend(
            [
                ms_home / "hub" / repo,
                ms_home / repo,
                ms_home / "models" / repo,
            ]
        )
    if cache_root:
        candidates.insert(0, cache_root / hub / repo.replace("/", os.sep))
    for candidate in candidates:
        resolved = resolve_snapshot_dir(candidate)
        if _looks_complete(resolved):
            return resolved
    return None


def _progress_tqdm(callback: ProgressFn, model_key: str, hub: str, repo: str):
    from huggingface_hub.utils.tqdm import tqdm as hf_tqdm

    class HubTqdm(hf_tqdm):
        def update(self, n=1):
            super().update(n)
            try:
                callback(
                    {
                        "type": "download_progress",
                        "model_key": model_key,
                        "hub": hub,
                        "repo": repo,
                        "file": getattr(self, "desc", None) or "",
                        "n": int(self.n),
                        "total": int(self.total) if self.total else None,
                        "unit": getattr(self, "unit", "B"),
                    }
                )
            except Exception:
                pass

    return HubTqdm


def download_model(
    model_key: str,
    hub: str,
    token: str = "",
    callback: ProgressFn | None = None,
) -> Path:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model {model_key}")
    if hub not in {"huggingface", "modelscope"}:
        raise ValueError(f"Unknown hub {hub}")
    repo = repo_id(model_key, hub)
    callback = callback or (lambda _event: None)
    callback(
        {
            "type": "download_start",
            "model_key": model_key,
            "hub": hub,
            "repo": repo,
            "approx_gb": MODELS[model_key]["approx_gb"],
        }
    )

    existing = model_local_path(model_key, hub)
    if existing:
        callback(
            {
                "type": "download_complete",
                "model_key": model_key,
                "hub": hub,
                "repo": repo,
                "path": str(existing),
                "already_present": True,
            }
        )
        return existing

    if hub == "huggingface":
        path = _download_huggingface(repo, token, callback, model_key, hub)
    else:
        path = _download_modelscope(repo, token, callback, model_key, hub)

    resolved = resolve_snapshot_dir(Path(path))
    if not _looks_complete(resolved):
        raise RuntimeError(
            f"Download finished but model_index.json was not found in {resolved}. "
            "The snapshot may be incomplete — retry the download."
        )
    callback(
        {
            "type": "download_complete",
            "model_key": model_key,
            "hub": hub,
            "repo": repo,
            "path": str(resolved),
            "already_present": False,
        }
    )
    return resolved


def _download_huggingface(repo: str, token: str, callback: ProgressFn, model_key: str, hub: str) -> Path:
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo,
        "tqdm_class": _progress_tqdm(callback, model_key, hub, repo),
    }
    if token:
        kwargs["token"] = token
    return Path(snapshot_download(**kwargs))


def _download_modelscope(repo: str, token: str, callback: ProgressFn, model_key: str, hub: str) -> Path:
    if token:
        os.environ["MODELSCOPE_API_TOKEN"] = token
        os.environ.setdefault("MODELSCOPE_SDK_TOKEN", token)

    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
    except Exception as exc:  # pragma: no cover - import surface
        raise RuntimeError(
            "ModelScope is not installed. `pip install modelscope` and retry."
        ) from exc

    stop = threading.Event()

    def _watch() -> None:
        cache = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
        last = -1
        while not stop.wait(1.2):
            size = _dir_size(cache)
            if size != last:
                last = size
                callback(
                    {
                        "type": "download_progress",
                        "model_key": model_key,
                        "hub": hub,
                        "repo": repo,
                        "file": "modelscope cache",
                        "n": size,
                        "total": None,
                        "unit": "B",
                    }
                )

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    started = time.time()
    try:
        path = ms_download(repo)
    finally:
        stop.set()
        watcher.join(timeout=2)
    callback(
        {
            "type": "log",
            "message": f"ModelScope snapshot finished in {time.time() - started:.1f}s",
        }
    )
    return Path(path)


def model_status(hub: str) -> list[dict]:
    rows = []
    for key, spec in MODELS.items():
        path = model_local_path(key, hub)
        size = _dir_size(path) if path else 0
        rows.append(
            {
                "key": key,
                "label": spec["label"],
                "precision": spec["precision"],
                "repo": repo_id(key, hub),
                "approx_gb": spec["approx_gb"],
                "downloaded": bool(path),
                "path": str(path) if path else None,
                "size_bytes": size,
                "size_gb": round(size / (1024**3), 2) if size else 0,
                "requires_cuda": spec["requires_cuda"],
                "notes_zh": spec["notes_zh"],
                "notes_en": spec["notes_en"],
                "urls": spec["urls"],
            }
        )
    return rows
