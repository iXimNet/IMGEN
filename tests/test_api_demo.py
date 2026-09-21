from io import BytesIO

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from imgen.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGEN_DEMO", "1")
    app = create_app(demo=True, home=tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def test_index_page(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "IMGEN" in page.text
    css = client.get("/css/app.css")
    assert css.status_code == 200


def test_health_and_bootstrap(client):
    health = client.get("/api/health").json()
    assert health["ok"] is True
    assert health["demo"] is True
    boot = client.get("/api/bootstrap").json()
    assert boot["prompts"]["counts"]["generate_prompts"] >= 70
    assert boot["sizes"]["2k"]["1:1"] == [2048, 2048]
    assert {row["key"] for row in boot["models"]} == {"qwen-image-2.1", "image21-int8"}


def test_demo_generate_and_history(client):
    response = client.post(
        "/api/jobs",
        data={
            "mode": "generate",
            "prompt": "a ceramic teapot on a wooden table",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
            "scale": "1k",
            "aspect": "1:1",
            "steps": "8",
            "true_cfg_scale": "1.0",
            "seed": "42",
        },
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    # Demo generation is a background thread; poll briefly.
    item = None
    for _ in range(40):
        item = client.get(f"/api/jobs/{job_id}").json()
        if item["status"] in {"succeeded", "failed"}:
            break
        import time

        time.sleep(0.05)
    assert item["status"] == "succeeded"
    assert item["seed"] == 42
    image = client.get(f"/api/outputs/{job_id}")
    assert image.status_code == 200
    listed = client.get("/api/jobs").json()["items"]
    assert listed[0]["id"] == job_id


def test_demo_edit_requires_image(client):
    response = client.post(
        "/api/jobs",
        data={
            "mode": "edit",
            "prompt": "change the background",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
        },
    )
    assert response.status_code == 200
    job_id = response.json()["id"]
    import time

    item = None
    for _ in range(40):
        item = client.get(f"/api/jobs/{job_id}").json()
        if item["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert item["status"] == "failed"


def test_demo_edit_with_reference(client):
    buf = BytesIO()
    Image.new("RGB", (64, 64), (12, 80, 160)).save(buf, format="PNG")
    buf.seek(0)
    response = client.post(
        "/api/jobs",
        data={
            "mode": "edit",
            "prompt": "Change the background to a sunset beach",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
            "scale": "1k",
            "steps": "8",
        },
        files=[("files", ("ref.png", buf, "image/png"))],
    )
    assert response.status_code == 200
    job_id = response.json()["id"]
    import time

    item = None
    for _ in range(50):
        item = client.get(f"/api/jobs/{job_id}").json()
        if item["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert item["status"] == "succeeded"
    assert item["ref_paths"]
    assert item["params"]["follow_ref_aspect"] is True
