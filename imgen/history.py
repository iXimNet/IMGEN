"""SQLite history of generation and edit jobs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .paths import AppPaths


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    model_key TEXT NOT NULL,
    hub TEXT NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    params_json TEXT NOT NULL,
    seed INTEGER,
    width INTEGER,
    height INTEGER,
    duration_ms INTEGER,
    image_path TEXT,
    thumb_path TEXT,
    ref_paths_json TEXT,
    status TEXT NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class History:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        paths.ensure()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.paths.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, record: dict[str, Any]) -> dict[str, Any]:
        job_id = record.get("id") or new_id()
        row = {
            "id": job_id,
            "created_at": record.get("created_at") or utc_now(),
            "mode": record["mode"],
            "model_key": record["model_key"],
            "hub": record["hub"],
            "prompt": record.get("prompt") or "",
            "negative_prompt": record.get("negative_prompt") or "",
            "params_json": json.dumps(record.get("params") or {}, ensure_ascii=False),
            "seed": record.get("seed"),
            "width": record.get("width"),
            "height": record.get("height"),
            "duration_ms": record.get("duration_ms"),
            "image_path": record.get("image_path"),
            "thumb_path": record.get("thumb_path"),
            "ref_paths_json": json.dumps(record.get("ref_paths") or [], ensure_ascii=False),
            "status": record.get("status") or "running",
            "error": record.get("error"),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, created_at, mode, model_key, hub, prompt, negative_prompt,
                    params_json, seed, width, height, duration_ms, image_path,
                    thumb_path, ref_paths_json, status, error
                ) VALUES (
                    :id, :created_at, :mode, :model_key, :hub, :prompt, :negative_prompt,
                    :params_json, :seed, :width, :height, :duration_ms, :image_path,
                    :thumb_path, :ref_paths_json, :status, :error
                )
                """,
                row,
            )
        return self.get(job_id)

    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        if "params" in fields:
            fields["params_json"] = json.dumps(fields.pop("params"), ensure_ascii=False)
        if "ref_paths" in fields:
            fields["ref_paths_json"] = json.dumps(fields.pop("ref_paths"), ensure_ascii=False)
        if not fields:
            return self.get(job_id)
        assignments = ", ".join(f"{key} = :{key}" for key in fields)
        fields["id"] = job_id
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = :id", fields)
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._to_dict(row) if row else None

    def list(self, limit: int = 80, offset: int = 0, query: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM jobs WHERE status != 'deleted'"
        params: list[Any] = []
        if query:
            sql += " AND (prompt LIKE ? OR mode LIKE ? OR model_key LIKE ?)"
            like = f"%{query}%"
            params.extend([like, like, like])
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._to_dict(row) for row in rows]

    def delete(self, job_id: str) -> bool:
        record = self.get(job_id)
        if not record:
            return False
        for key in ("image_path", "thumb_path"):
            path = record.get(key)
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
        for ref in record.get("ref_paths") or []:
            try:
                Path(ref).unlink(missing_ok=True)
            except OSError:
                pass
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return True

    def save_image(self, job_id: str, image: Image.Image) -> tuple[str, str]:
        self.paths.ensure()
        image_path = self.paths.outputs / f"{job_id}.png"
        thumb_path = self.paths.thumbs / f"{job_id}.jpg"
        image.save(image_path, format="PNG")
        thumb = image.convert("RGB")
        thumb.thumbnail((640, 640))
        thumb.save(thumb_path, format="JPEG", quality=86, optimize=True)
        return str(image_path), str(thumb_path)

    def save_refs(self, job_id: str, images: list[Image.Image]) -> list[str]:
        folder = self.paths.refs / job_id
        folder.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for index, image in enumerate(images):
            path = folder / f"ref_{index:02d}.png"
            image.save(path, format="PNG")
            paths.append(str(path))
        return paths

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["params"] = json.loads(data.pop("params_json") or "{}")
        data["ref_paths"] = json.loads(data.pop("ref_paths_json") or "[]")
        return data
