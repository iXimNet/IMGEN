"""Persistent user settings stored under IMGEN_HOME."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .paths import AppPaths

DEFAULTS: dict[str, Any] = {
    "language": "zh",
    "hub": "huggingface",
    "model_key": "qwen-image-2.1",
    "setup_completed": False,
    "hf_token": "",
    "ms_token": "",
    "open_browser": True,
    "cpu_offload": "auto",
    "vae_tiling": "auto",
    "last_params": {},
}


class ConfigStore:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._data = deepcopy(DEFAULTS)

    def load(self) -> dict[str, Any]:
        self.paths.ensure()
        if self.paths.config_file.exists():
            try:
                loaded = json.loads(self.paths.config_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    merged = deepcopy(DEFAULTS)
                    merged.update({k: v for k, v in loaded.items() if k in DEFAULTS or k == "last_params"})
                    self._data = merged
            except (OSError, json.JSONDecodeError):
                self._data = deepcopy(DEFAULTS)
        return deepcopy(self._data)

    def save(self, patch: dict[str, Any] | None = None) -> dict[str, Any]:
        if patch:
            for key, value in patch.items():
                self._data[key] = value
        self.paths.ensure()
        payload = deepcopy(self._data)
        self.paths.config_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            self.paths.config_file.chmod(0o600)
        except OSError:
            pass
        return deepcopy(self._data)

    def public_view(self) -> dict[str, Any]:
        data = deepcopy(self._data)
        data["hf_token_set"] = bool(str(data.get("hf_token") or "").strip())
        data["ms_token_set"] = bool(str(data.get("ms_token") or "").strip())
        data.pop("hf_token", None)
        data.pop("ms_token", None)
        return data

    def token_for(self, hub: str) -> str:
        if hub == "huggingface":
            return str(self._data.get("hf_token") or "").strip()
        if hub == "modelscope":
            return str(self._data.get("ms_token") or "").strip()
        return ""


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
