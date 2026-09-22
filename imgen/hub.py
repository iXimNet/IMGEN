"""Download Qwen-Image-2.1 / Image21-INT8 from Hugging Face or ModelScope."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

from .constants import MODELS, repo_id

ProgressFn = Callable[[dict], None]

# Image21-INT8 ships a large evaluation gallery. Skip it so weight shards
# are the files that actually get downloaded.
HF_IGNORE_PATTERNS = [
    "evaluation/**",
    "benchmarks/**",
    "tests/**",
    "cards/**",
    "scripts/**",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.webp",
    "**/*.gif",
    "**/*.md",
]
MS_IGNORE_PATTERNS = [
    r"evaluation/.*",
    r"benchmarks/.*",
    r"tests/.*",
    r"cards/.*",
    r"scripts/.*",
    r".*\.png$",
    r".*\.jpg$",
    r".*\.jpeg$",
    r".*\.webp$",
    r".*\.gif$",
    r".*\.md$",
]
MIN_WEIGHT_BYTES = 1024


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


def _file_ok(path: Path, min_bytes: int = MIN_WEIGHT_BYTES) -> bool:
    try:
        return path.exists() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def missing_weight_files(root: Path) -> list[str]:
    """Return relative paths of weight shards that are absent or empty."""
    if not root.exists():
        return ["(snapshot missing)"]
    missing: list[str] = []
    if not (root / "model_index.json").exists():
        missing.append("model_index.json")
        return missing

    indexes = list(root.rglob("*.safetensors.index.json"))
    for index_path in indexes:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append(str(index_path.relative_to(root)).replace("\\", "/"))
            continue
        shards = sorted(set((payload.get("weight_map") or {}).values()))
        if not shards:
            missing.append(str(index_path.relative_to(root)).replace("\\", "/"))
            continue
        for name in shards:
            shard = index_path.parent / name
            if not _file_ok(shard):
                missing.append(str(shard.relative_to(root)).replace("\\", "/"))

    for folder in ("transformer", "text_encoder", "vae"):
        directory = root / folder
        if not directory.is_dir():
            missing.append(f"{folder}/")
            continue
        has_index = any(directory.glob("*.safetensors.index.json"))
        singles = list(directory.glob("*.safetensors"))
        if not has_index and not any(_file_ok(item) for item in singles):
            missing.append(f"{folder}/*.safetensors")

    # Unique, stable order.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in missing:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def snapshot_is_complete(path: Path) -> bool:
    return path.exists() and not missing_weight_files(path)


def resolve_snapshot_dir(path: Path) -> Path:
    if (path / "model_index.json").exists():
        return path
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / "model_index.json").exists():
                return child
    return path


def _snapshot_candidates(model_key: str, hub: str, cache_root: Path | None = None) -> list[Path]:
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
    return candidates


def find_snapshot_dir(model_key: str, hub: str, cache_root: Path | None = None) -> Path | None:
    """Return a snapshot that has model_index.json, even if weight shards are missing."""
    for candidate in _snapshot_candidates(model_key, hub, cache_root):
        resolved = resolve_snapshot_dir(candidate)
        if (resolved / "model_index.json").exists():
            return resolved
    return None


def model_local_path(model_key: str, hub: str, cache_root: Path | None = None) -> Path | None:
    """Lookup a snapshot that has every required weight shard on disk."""
    found = find_snapshot_dir(model_key, hub, cache_root)
    if found and snapshot_is_complete(found):
        return found
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
    force: bool = False,
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
    if existing and not force:
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

    incomplete = find_snapshot_dir(model_key, hub)
    if incomplete:
        missing = missing_weight_files(incomplete)
        callback(
            {
                "type": "log",
                "message": "Resuming incomplete snapshot; missing "
                + ", ".join(missing[:6])
                + ("…" if len(missing) > 6 else ""),
            }
        )

    if hub == "huggingface":
        path = _download_huggingface(repo, token, callback, model_key, hub)
    else:
        path = _download_modelscope(repo, token, callback, model_key, hub)

    resolved = resolve_snapshot_dir(Path(path))
    still_missing = missing_weight_files(resolved)
    if still_missing:
        preview = ", ".join(still_missing[:6])
        extra = "…" if len(still_missing) > 6 else ""
        raise RuntimeError(
            f"Download finished but weight shards are still missing: {preview}{extra}. "
            "Retry the download to resume the remaining files."
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
        "ignore_patterns": HF_IGNORE_PATTERNS,
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
        try:
            path = ms_download(repo, ignore_file_pattern=MS_IGNORE_PATTERNS)
        except TypeError:
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
        complete = model_local_path(key, hub)
        found = find_snapshot_dir(key, hub)
        missing = missing_weight_files(found) if found and not complete else []
        path = complete or found
        size = _dir_size(path) if path else 0
        rows.append(
            {
                "key": key,
                "label": spec["label"],
                "precision": spec["precision"],
                "repo": repo_id(key, hub),
                "approx_gb": spec["approx_gb"],
                "downloaded": bool(complete),
                "incomplete": bool(found) and not bool(complete),
                "missing_files": missing[:12],
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
