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
