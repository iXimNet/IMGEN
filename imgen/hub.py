"""Download BF16 / Image21-INT8 / Image21-INT4 from Hugging Face or ModelScope."""

from __future__ import annotations

import fnmatch
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

from .constants import HUBS, MODELS, repo_id

ProgressFn = Callable[[dict], None]

# The community checkpoints ship large evaluation galleries. Skip them so weight shards
# are the files that actually get downloaded.
#
# One pattern list serves both hubs, because both matchers agree that `*` spans
# `/` (Hugging Face's `fnmatchcase`, ModelScope's `_matches_patterns`). ModelScope
# used to be handed regex-style patterns (`.*\.png$`) which it matches with
# `fnmatch` — `$` is a literal there, so nothing was ever skipped and README.md
# plus every gallery image came down with the weights.
_GALLERY_DIRS = ("evaluation", "benchmarks", "tests", "cards", "scripts")
_GALLERY_SUFFIXES = ("png", "jpg", "jpeg", "webp", "gif", "md")

IGNORE_PATTERNS = [
    *(f"{folder}/*" for folder in _GALLERY_DIRS),
    *(f"*.{suffix}" for suffix in _GALLERY_SUFFIXES),
]
# Kept as named aliases: the two hubs take the same list, and callers/tests
# refer to them by hub.
HF_IGNORE_PATTERNS = IGNORE_PATTERNS
MS_IGNORE_PATTERNS = IGNORE_PATTERNS
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
    """The directory that actually holds a snapshot's files.

    A downloader may hand back the repo root instead of the revision folder
    (ModelScope nests files under ``snapshots/<revision>/``), so descend until
    ``model_index.json`` shows up.
    """
    if (path / "model_index.json").exists():
        return path
    if not path.is_dir():
        return path
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "model_index.json").exists():
            return child
    # One more level: repo root -> snapshots/<revision>.
    for child in sorted(path.iterdir(), reverse=True):
        if child.is_dir() and child.name == "snapshots":
            for revision in sorted(
                (item for item in child.iterdir() if item.is_dir()),
                key=lambda item: item.name,
                reverse=True,
            ):
                if (revision / "model_index.json").exists():
                    return revision
    return path


HF_ENV_ORDER = ("HF_HUB_CACHE", "HF_HOME")
MS_ENV_ORDER = ("MODELSCOPE_CACHE",)


def _hub_cache_roots(hub: str) -> list[Path]:
    """Candidate cache roots for a hub, most authoritative first.

    These mirror what the download clients themselves read, so the studio
    writes weights where the rest of the toolchain already looks for them.
    """
    if hub == "huggingface":
        roots: list[Path] = []
        hub_cache = os.environ.get("HF_HUB_CACHE")
        if hub_cache:
            roots.append(Path(hub_cache))
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        roots.extend([hf_home / "hub", hf_home])
        return roots
    if hub == "modelscope":
        return [Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))]
    raise ValueError(f"Unknown hub {hub}")


def hub_cache_root(hub: str) -> Path:
    """The directory this hub keeps weights in, after environment overrides."""
    return _hub_cache_roots(hub)[0]


def hub_storage(hub: str) -> dict:
    """Where a hub reads and writes weights, and which setting chose the path."""
    env_vars = HF_ENV_ORDER if hub == "huggingface" else MS_ENV_ORDER
    env_var = next((name for name in env_vars if os.environ.get(name)), None)
    if hub == "huggingface":
        default = Path.home() / ".cache" / "huggingface" / "hub"
    else:
        default = Path.home() / ".cache" / "modelscope"
    root = hub_cache_root(hub)
    return {
        "hub": hub,
        "path": str(root),
        "env_var": env_var,
        "default_path": str(default),
        "exists": root.exists(),
    }


def _modelscope_repo_dirs(root: Path, repo: str) -> list[Path]:
    """Every on-disk directory a ModelScope client may have put ``repo`` in.

    Two generations of layout exist, and both are still found in the wild:

    * current (SDK >= 1.38, i.e. ``modelscope_hub``) —
      ``{root}/models/{owner}--{name}/snapshots/{revision}/``
    * older — the repo directly under ``{root}/hub/models/``, ``{root}/models/``,
      ``{root}/hub/`` or ``{root}/``, with the owner as a sub-directory and
      dots in the name optionally encoded as ``___``.

    The list mirrors what the SDKs themselves probe, so a snapshot is found
    wherever the downloader decided to leave it.
    """
    owner, _, name = repo.partition("/")
    safe_name = name.replace(".", "___")
    flat_id = repo.replace("/", "--")
    ordered = [
        root / "models" / flat_id,
        root / "models" / owner / safe_name,
        root / "hub" / "models" / owner / safe_name,
        root / "hub" / "models" / owner / name,
        root / "models" / owner / name,
        root / "hub" / owner / safe_name,
        root / "hub" / owner / name,
        root / owner / safe_name,
        root / owner / name,
    ]
    # `safe_name` equals `name` for repos without dots, so drop the repeats.
    seen: set[str] = set()
    unique: list[Path] = []
    for path in ordered:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _snapshot_children(repo_dir: Path) -> list[Path]:
    """The directory holding a repo's files, newest revision first.

    Current ModelScope clients add a ``snapshots/<revision>`` layer; older
    ones wrote the files straight into the repo directory. Probe both so a
    snapshot is found wherever it landed.
    """
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        revisions = sorted(
            (item for item in snapshots.iterdir() if item.is_dir()),
            key=lambda item: item.name,
            reverse=True,
        )
        return revisions or [repo_dir]
    return [repo_dir]


def _snapshot_candidates(model_key: str, hub: str, cache_root: Path | None = None) -> list[Path]:
    repo = repo_id(model_key, hub)
    candidates: list[Path] = []
    if hub == "huggingface":
        dir_name = "models--" + repo.replace("/", "--")
        for root in _hub_cache_roots("huggingface"):
            hub_dir = root / dir_name
            snapshots = hub_dir / "snapshots"
            if snapshots.exists():
                for snap in sorted(snapshots.iterdir(), reverse=True):
                    candidates.append(snap)
            candidates.append(hub_dir)
    else:
        # ModelScope moved snapshots under `hub/models/` and then again under
        # `models/<owner>--<name>/snapshots/`; every layout is still reachable.
        for root in _hub_cache_roots("modelscope"):
            for repo_dir in _modelscope_repo_dirs(root, repo):
                candidates.extend(_snapshot_children(repo_dir))
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


def hub_snapshot_state(model_key: str, hub: str, cache_root: Path | None = None) -> dict:
    """What one hub's cache holds for a model: nothing, a partial snapshot, or weights."""
    complete = model_local_path(model_key, hub, cache_root)
    found = complete or find_snapshot_dir(model_key, hub, cache_root)
    return {
        "hub": hub,
        "complete": bool(complete),
        "incomplete": bool(found) and not complete,
        "path": complete or found,
        "missing": missing_weight_files(found) if found and not complete else [],
    }


def resolve_local_hub(model_key: str, hub: str, cache_root: Path | None = None) -> str | None:
    """The hub that actually holds this model on disk, preferring the requested one.

    A user can have the weights from ModelScope while the studio is set to
    Hugging Face (or the reverse). Generating should use what is on disk
    instead of failing, so callers resolve the hub before loading.
    """
    if hub_snapshot_state(model_key, hub, cache_root)["complete"]:
        return hub
    for other in HUBS:
        if other != hub and hub_snapshot_state(model_key, other, cache_root)["complete"]:
            return other
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


def _ms_matches_any(path: str, patterns: list[str]) -> bool:
    """Mirror ``modelscope_hub._download._matches_patterns``.

    Totals have to be filtered exactly the way the download filters, or the
    percentage drifts away from 100%.
    """
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
        # The SDK also accepts legacy regex-style patterns.
        if ".*" in pattern and fnmatch.fnmatch(path, pattern.replace(".*", "*")):
            return True
    return False


def _modelscope_remote_total(repo: str, token: str = "") -> int | None:
    """Bytes the download will actually fetch, or ``None`` if it cannot be known.

    A percentage needs a denominator, and the only trustworthy one is the
    remote file list run through the same ignore patterns the download uses.
    Any failure here is non-fatal: the watcher still reports raw bytes.
    """
    try:
        from modelscope_hub import HubApi
    except Exception:
        return None
    try:
        api = HubApi(token=token or None)
        files = api.list_repo_files(repo, "model")
    except Exception:
        return None
    total = 0
    for item in files:
        path = getattr(item, "path", "") or ""
        if not path or getattr(item, "type", "blob") == "tree":
            continue
        if _ms_matches_any(path, MS_IGNORE_PATTERNS):
            continue
        total += int(getattr(item, "size", 0) or 0)
    return total or None


def _download_modelscope(
    repo: str, token: str, callback: ProgressFn, model_key: str, hub: str
) -> Path:
    if token:
        os.environ["MODELSCOPE_API_TOKEN"] = token
        os.environ.setdefault("MODELSCOPE_SDK_TOKEN", token)

    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
    except Exception as exc:  # pragma: no cover - import surface
        raise RuntimeError(
            "ModelScope is not installed. `pip install modelscope` and retry."
        ) from exc

    # Watch the repo's own directory, not the whole cache root: the cache holds
    # every other model too, so its size says nothing about this download. The
    # SDK writes to the current `models/<owner>--<name>` layout, but may reuse a
    # legacy directory, so all candidates are sampled and the largest wins —
    # monotonic, and correct whichever one is being written.
    watch_dirs = _modelscope_repo_dirs(hub_cache_root("modelscope"), repo)
    total_bytes = _modelscope_remote_total(repo, token)
    stop = threading.Event()

    def _watch() -> None:
        last = -1
        while not stop.wait(1.2):
            size = max((_dir_size(folder) for folder in watch_dirs), default=0)
            if size != last:
                last = size
                callback(
                    {
                        "type": "download_progress",
                        "model_key": model_key,
                        "hub": hub,
                        "repo": repo,
                        "file": "modelscope",
                        "n": size,
                        "total": total_bytes,
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
    """Per-model presence report, told apart by which hub holds the weights.

    `downloaded` answers "is it in the selected source"; `downloaded_any`
    answers "is it on disk at all". The studio only ever needs the second one
    to decide whether a model is usable, which is why both are reported.
    """
    order = [hub] + [key for key in HUBS if key != hub]
    rows = []
    for key, spec in MODELS.items():
        states = {name: hub_snapshot_state(key, name) for name in order}
        selected = states[hub]
        complete_hubs = [name for name in order if states[name]["complete"]]
        incomplete_hubs = [name for name in order if states[name]["incomplete"]]
        local_hub = complete_hubs[0] if complete_hubs else None

        if complete_hubs:
            preferred = states[local_hub]["path"]
        else:
            preferred = selected["path"] or next(
                (states[name]["path"] for name in incomplete_hubs), None
            )
        missing = selected["missing"] or next(
            (states[name]["missing"] for name in incomplete_hubs), []
        )
        size = _dir_size(preferred) if preferred else 0

        rows.append(
            {
                "key": key,
                "label": spec["label"],
                "precision": spec["precision"],
                "repo": repo_id(key, hub),
                "approx_gb": spec["approx_gb"],
                "downloaded": bool(selected["complete"]),
                "downloaded_any": bool(complete_hubs),
                "available_hubs": complete_hubs,
                "local_hub": local_hub,
                "incomplete": bool(selected["incomplete"]),
                "incomplete_any": bool(incomplete_hubs),
                "incomplete_hubs": incomplete_hubs,
                "missing_files": missing[:12],
                "missing_count": len(missing),
                "path": str(preferred) if preferred else None,
                "size_bytes": size,
                "size_gb": round(size / (1024**3), 2) if size else 0,
                "requires_cuda": spec["requires_cuda"],
                "notes_zh": spec["notes_zh"],
                "notes_en": spec["notes_en"],
                "urls": spec["urls"],
            }
        )
    return rows
