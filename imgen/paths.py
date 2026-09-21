"""Filesystem layout for config, history, and generated images."""

from __future__ import annotations

import os
from pathlib import Path


def default_home() -> Path:
    override = os.environ.get("IMGEN_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".imgen"


class AppPaths:
    def __init__(self, home: Path | None = None) -> None:
        self.home = Path(home) if home else default_home()
        self.config_file = self.home / "config.json"
        self.db_file = self.home / "history.sqlite"
        self.outputs = self.home / "outputs"
        self.thumbs = self.home / "thumbs"
        self.refs = self.home / "refs"
        self.logs = self.home / "logs"

    def ensure(self) -> None:
        for folder in (self.home, self.outputs, self.thumbs, self.refs, self.logs):
            folder.mkdir(parents=True, exist_ok=True)
