from PIL import Image

from imgen.history import History
from imgen.paths import AppPaths


def test_history_roundtrip(tmp_path):
    paths = AppPaths(tmp_path)
    hist = History(paths)
    record = hist.create(
        {
            "mode": "generate",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
            "prompt": "a lamp",
            "params": {"steps": 40, "true_cfg_scale": 1.0},
            "seed": 42,
            "width": 1024,
            "height": 1024,
            "status": "running",
        }
    )
    job_id = record["id"]
    image = Image.new("RGBA", (64, 64), (20, 30, 40, 255))
    image_path, thumb_path = hist.save_image(job_id, image)
    hist.update(
        job_id,
        status="succeeded",
        image_path=image_path,
        thumb_path=thumb_path,
        duration_ms=1234,
    )
    loaded = hist.get(job_id)
    assert loaded["prompt"] == "a lamp"
    assert loaded["params"]["steps"] == 40
    assert loaded["status"] == "succeeded"
    listed = hist.list()
    assert listed[0]["id"] == job_id
    assert hist.delete(job_id) is True
    assert hist.get(job_id) is None


def _make(hist: History, prompt: str) -> dict:
    return hist.create(
        {
            "mode": "generate",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
            "prompt": prompt,
            "params": {"steps": 40},
            "status": "succeeded",
        }
    )


def test_a_favorite_is_a_mark_on_a_record_the_list_can_filter_on(tmp_path):
    """Starring must not move or copy the record — only mark it.

    A copy would drift the moment the original was edited or deleted, and the
    two views would then disagree about the same picture.
    """
    hist = History(AppPaths(tmp_path))
    kept = _make(hist, "kept")
    other = _make(hist, "other")

    assert kept["favorite"] is False
    assert hist.set_favorite(kept["id"], True)["favorite"] is True

    # Marked in place: same row, same id, still in the unfiltered list.
    assert hist.get(kept["id"])["favorite"] is True
    assert {row["id"] for row in hist.list()} == {kept["id"], other["id"]}

    only = hist.list(favorite=True)
    assert [row["id"] for row in only] == [kept["id"]]
    assert hist.count(favorite=True) == 1

    # Unstarring is the same call with the other value.
    hist.set_favorite(kept["id"], False)
    assert hist.count(favorite=True) == 0
    assert hist.get(kept["id"]) is not None

    # An unknown id changes nothing and creates nothing.
    assert hist.set_favorite("nosuchrecord", True) is None
    assert hist.count() == 2


def test_favorites_narrow_a_search_rather_than_replacing_it(tmp_path):
    """Both conditions apply, because the filter is one more WHERE term."""
    hist = History(AppPaths(tmp_path))
    keep = _make(hist, "needle keep")
    _make(hist, "needle drop")
    hist.set_favorite(keep["id"], True)

    assert [row["id"] for row in hist.list(query="needle", favorite=True)] == [keep["id"]]
    assert hist.count(query="needle", favorite=True) == 1
    assert hist.count(query="absent", favorite=True) == 0
    # The query alone still reaches both.
    assert hist.count(query="needle") == 2


def test_a_database_written_before_favorites_gains_the_column(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` will not reshape a table it finds.

    A studio that has been running since before this column existed keeps the
    old schema on disk, so startup has to add the column in place — and every
    record written before it must read as "not starred", not as NULL.
    """
    import sqlite3

    paths = AppPaths(tmp_path)
    paths.ensure()
    conn = sqlite3.connect(paths.db_file)
    conn.executescript(
        """
        CREATE TABLE jobs (
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
        )
        """
    )
    conn.execute(
        "INSERT INTO jobs (id, created_at, mode, model_key, hub, prompt, params_json, status)"
        " VALUES ('old1', '2026-01-01T00:00:00+00:00', 'generate', 'm', 'h', 'old', '{}', 'succeeded')"
    )
    conn.commit()
    conn.close()

    hist = History(paths)
    old = hist.get("old1")
    assert old is not None
    assert old["favorite"] is False
    # And the new column is writable on the migrated table.
    assert hist.set_favorite("old1", True)["favorite"] is True
