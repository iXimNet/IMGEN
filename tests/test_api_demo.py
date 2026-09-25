import time
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
    # Settings is the model / download sheet; history uses a merged detail overlay.
    assert 'id="settingsClose"' in page.text
    assert 'id="detail"' in page.text
    assert 'id="dRefList"' in page.text
    assert 'id="composerScroll"' in page.text
    assert 'id="canvasFields"' in page.text
    assert 'id="refSizeBlock"' in page.text
    assert 'id="sizeVal"' in page.text
    assert 'id="popPresets"' in page.text
    assert 'id="btnPresets"' in page.text
    assert 'id="zoombar"' in page.text
    # Fonts must come from the system stack, not a CDN that is unreachable on
    # a typical China network.
    assert "fonts.googleapis" not in page.text
    css = client.get("/css/app.css")
    assert css.status_code == 200
    assert "--accent" in css.text
    assert ".viewport" in css.text
    assert ".zoomwrap" in css.text
    assert ".detail" in css.text
    js = client.get("/js/app.js")
    assert js.status_code == 200
    assert "openDetail" in js.text
    assert "makeZoom" in js.text
    assert ".kv" in css.text


def test_health_and_bootstrap(client):
    health = client.get("/api/health").json()
    assert health["ok"] is True
    assert health["demo"] is True
    boot = client.get("/api/bootstrap").json()
    assert boot["prompts"]["counts"]["generate_prompts"] >= 70
    assert boot["sizes"]["2k"]["1:1"] == [2048, 2048]
    assert {row["key"] for row in boot["models"]} == {
        "qwen-image-2.1", "image21-int8", "image21-int4",
    }


def test_default_scale_is_1k(client):
    """The Image21-INT8 card withdrew its 2048px recommendation."""
    boot = client.get("/api/bootstrap").json()
    assert boot["sizes"]["default_scale"] == "1k"


def test_bootstrap_reports_storage_dirs(client):
    """The storage block must describe real directories, per source."""
    storage = client.get("/api/bootstrap").json()["storage"]
    hub_dirs = storage["hub_dirs"]

    assert set(hub_dirs) == {"huggingface", "modelscope"}
    for row in hub_dirs.values():
        assert row["path"] and row["default_path"]
        # Either an environment variable overrode the path, or the default stands.
        assert (row["env_var"] is None) == (row["path"] == row["default_path"])
    # `~/.imgen/models` never existed — the two sources point somewhere different.
    assert hub_dirs["huggingface"]["path"] != hub_dirs["modelscope"]["path"]
    assert storage["outputs"].endswith("outputs")
    assert storage["home"] == str(client.app.state.paths.home)


def test_vae_tiling_defaults_to_off(client):
    """Tiled VAE decoding is what stained the 2K edits, so it is opt-in."""
    assert client.get("/api/bootstrap").json()["config"]["vae_tiling"] == "off"


def test_generate_without_sizes_follows_the_default_scale(client):
    """No scale, no width/height in the request: the server's default decides."""
    response = client.post(
        "/api/jobs",
        data={"mode": "generate", "prompt": "default size probe", "steps": "8"},
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    item = None
    for _ in range(60):
        item = client.get(f"/api/jobs/{job_id}").json()
        if item["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert item and item["status"] == "succeeded", item
    assert (item["width"], item["height"]) == (1024, 1024)
    assert item["params"]["scale"] == "1k"


def test_parameter_copy_makes_no_authority_claim(client):
    """Image21-INT8 is a community conversion whose 2048px + tiling recipe was
    corrected, so nothing in the panel may be sold as an official recommendation."""
    page = client.get("/").text
    for blob in (page, client.get("/js/app.js").text, client.get("/js/i18n.js").text):
        assert "官方推荐" not in blob
        assert "官方默认" not in blob
        assert "Official default" not in blob
    # The select offers on / off only; `auto` is a legacy value, never a choice.
    assert 'value="auto"' not in page
    assert 'data-i="vaeAuto"' not in page


def test_scale_labels_and_weight_notes_are_descriptive(client):
    boot = client.get("/api/bootstrap").json()

    for spec in boot["sizes"]["scales"].values():
        assert "官方" not in spec["label_zh"]
        assert "official" not in spec["label_en"].lower()

    int8 = next(row for row in boot["models"] if row["key"] == "image21-int8")
    # The withdrawn advice was "edit at 2048 with VAE tiling on".
    assert "2048" not in int8["notes_zh"]
    assert "2048" not in int8["notes_en"]
    assert "VAE Tiling 保持关闭" in int8["notes_zh"]
    assert "leave VAE tiling off" in int8["notes_en"]


@pytest.mark.parametrize("model_key", ["qwen-image-2.1", "image21-int4"])
def test_demo_generate_and_history(client, model_key):
    response = client.post(
        "/api/jobs",
        data={
            "mode": "generate",
            "prompt": "a ceramic teapot on a wooden table",
            "model_key": model_key,
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
    assert item["model_key"] == model_key
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


@pytest.mark.parametrize("model_key", ["qwen-image-2.1", "image21-int4"])
def test_demo_edit_with_reference(client, model_key):
    buf = BytesIO()
    Image.new("RGB", (64, 64), (12, 80, 160)).save(buf, format="PNG")
    buf.seek(0)
    response = client.post(
        "/api/jobs",
        data={
            "mode": "edit",
            "prompt": "Change the background to a sunset beach",
            "model_key": model_key,
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
    assert item["ref_count"] == 1
    assert item["ref_urls"] == [f"/api/refs/{job_id}/0"]
    assert item["params"]["follow_ref_aspect"] is True


def test_jobs_never_expose_local_paths(client):
    """The detail view needs URLs, not the absolute paths on disk."""
    response = client.post(
        "/api/jobs",
        data={
            "mode": "generate",
            "prompt": "a quiet lighthouse",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
            "scale": "1k",
            "steps": "8",
        },
    )
    assert response.status_code == 200
    job_id = response.json()["id"]
    import time

    items = []
    for _ in range(50):
        items = client.get("/api/jobs?limit=5").json()["items"]
        if items and items[0]["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert items
    for item in items:
        for leaked in ("image_path", "thumb_path", "ref_paths"):
            assert leaked not in item, f"{leaked} must not reach the browser"
        assert isinstance(item["ref_count"], int)
        assert len(item["ref_urls"]) == item["ref_count"]
    assert client.get(f"/api/jobs/{job_id}").json()["image_url"] == f"/api/outputs/{job_id}"


def test_bootstrap_reports_engine_health(client):
    """The pill dot reports health; residency is a separate field."""
    engine = client.get("/api/bootstrap").json()["engine"]
    assert "last_error" in engine, "health needs its own field, not loaded/not-loaded"
    assert engine["last_error"] is None, "a fresh demo engine has not failed"


def test_demo_engine_loads_without_error(client):
    """A successful run must not leave a stale failure behind."""
    response = client.post(
        "/api/jobs",
        data={"mode": "generate", "prompt": "health probe", "steps": "4"},
    )
    job_id = response.json()["id"]
    item = None
    for _ in range(60):
        item = client.get(f"/api/jobs/{job_id}").json()
        if item["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert item and item["status"] == "succeeded", item

    engine = client.get("/api/bootstrap").json()["engine"]
    assert engine["last_error"] is None
    # Demo renders placeholder images without touching a pipeline, so `loaded`
    # stays None — and that is not a failure. Residency and health are separate.
    assert engine["loaded"] is None


def test_bootstrap_reports_engine_health(client):
    """The pill dot reports health; residency is a separate field."""
    engine = client.get("/api/bootstrap").json()["engine"]
    assert "last_error" in engine, "health needs its own field, not loaded/not-loaded"
    assert engine["last_error"] is None, "a fresh demo engine has not failed"


def _wait_idle(client, tries: int = 120) -> None:
    """Wait until the engine accepts another job. `busy` is the real gate."""
    for _ in range(tries):
        if not client.get("/api/bootstrap").json()["engine"]["busy"]:
            return
        time.sleep(0.05)


def _make_jobs(client, count: int, prefix: str) -> None:
    """Submit jobs one at a time — a second POST while one runs gets a 409."""
    for i in range(count):
        _wait_idle(client)
        resp = client.post("/api/jobs", data={"mode": "generate", "prompt": f"{prefix} {i}", "steps": "4"})
        assert resp.status_code == 200, resp.text
    _wait_idle(client)


def test_jobs_paging_reports_total(client):
    """`total` is what lets the list decide whether another page exists."""
    _make_jobs(client, 3, "page probe")
    total = client.get("/api/jobs?limit=1").json()["total"]
    assert total >= 3, total

    first = client.get("/api/jobs?limit=2&offset=0").json()
    assert len(first["items"]) == 2, first
    assert first["total"] == total, first
    assert first["offset"] == 0 and first["limit"] == 2

    second = client.get("/api/jobs?limit=2&offset=2").json()
    assert second["offset"] == 2
    assert second["total"] == total, "total must not depend on the page"
    ids_first = {i["id"] for i in first["items"]}
    ids_second = {i["id"] for i in second["items"]}
    assert not (ids_first & ids_second), "pages must not overlap"


def test_jobs_search_is_server_side(client):
    """Searching must reach records beyond the first page."""
    _wait_idle(client)
    resp = client.post("/api/jobs", data={"mode": "generate", "prompt": "zzunique needle", "steps": "4"})
    assert resp.status_code == 200, resp.text
    _wait_idle(client)

    hit = client.get("/api/jobs?q=zzunique").json()
    assert hit["total"] >= 1
    assert all("zzunique" in (i["prompt"] or "") for i in hit["items"])

    miss = client.get("/api/jobs?q=zznotpresent").json()
    assert miss["total"] == 0 and miss["items"] == []


def test_history_count_matches_list_filter(client):
    """count() and list() share a WHERE clause; they must agree."""
    total = client.get("/api/jobs?limit=1").json()["total"]
    everything = client.get(f"/api/jobs?limit={max(total, 1)}").json()
    assert len(everything["items"]) == total


def test_apply_i18n_reruns_every_language_dependent_renderer(client):
    """A renderer that reads `S.lang` must be re-run on a language switch.

    Missing one leaves stale copy on screen until the next unrelated repaint —
    that is exactly how the scale chips got stuck in the previous language.
    """
    import re

    js = client.get("/js/app.js").text
    body = js[js.index("function applyI18n"):]
    body = body[: body.index("\n  }")]

    # Every renderer whose body mentions S.lang (directly, or via tr/tfx).
    lang_sensitive = set()
    for m in re.finditer(r"^  function (render\w+)\(", js, re.M):
        name = m.group(1)
        start = m.end()
        end = js.index("\n  }\n", start)
        fn_body = js[start:end]
        if "S.lang" in fn_body or re.search(r"\b(tr|tfx)\(", fn_body):
            lang_sensitive.add(name)

    assert lang_sensitive, "expected to find language-sensitive renderers"

    # Renderers that take a record cannot be called bare; `refreshOpenDetail`
    # re-runs those for whichever record is open.
    takes_args = {
        m.group(1)
        for m in re.finditer(r"^  function (render\w+)\((\w)", js, re.M)
    }
    refresh = js[js.index("function refreshOpenDetail"):]
    refresh = refresh[: refresh.index("\n  }")]

    missing = []
    for name in sorted(lang_sensitive):
        if f"{name}()" in body:
            continue
        if name in takes_args and f"{name}(" in refresh:
            continue
        missing.append(name)
    assert not missing, f"applyI18n does not re-run: {missing}"


def test_topbar_links_to_the_repository(client):
    """The GitHub entry must be a real anchor, not a scripted button.

    An anchor is what gives users middle-click / open-in-new-tab for free, and
    `rel="noopener"` is what stops the new tab from reaching back into this one.
    """
    html = client.get("/").text

    assert 'id="githubLink"' in html
    assert 'href="https://github.com/iXimNet/IMGEN"' in html
    anchor = html[html.index('id="githubLink"') - 200: html.index('id="githubLink"') + 200]
    assert 'target="_blank"' in anchor
    assert "noopener" in anchor
    # The icon lives in the sprite like every other top-bar glyph.
    assert 'symbol id="i-github"' in html


def test_topbar_links_to_the_repository(client):
    """The GitHub entry must be a real anchor, not a scripted button.

    An anchor is what gives users middle-click / open-in-new-tab for free, and
    `rel="noopener"` is what stops the new tab from reaching back into this one.
    """
    html = client.get("/").text

    assert 'id="githubLink"' in html
    assert 'href="https://github.com/iXimNet/IMGEN"' in html
    anchor = html[html.index('id="githubLink"') - 200: html.index('id="githubLink"') + 200]
    assert 'target="_blank"' in anchor
    assert "noopener" in anchor
    # The icon lives in the sprite like every other top-bar glyph.
    assert 'symbol id="i-github"' in html


def test_brand_mark_and_favicon_share_one_symbol(client):
    """The header mark and the favicon must stay the same artwork.

    They are drawn twice on purpose — the favicon is a standalone file the
    browser fetches, and the header needs it inline to inherit nothing — so the
    shared 48-unit grid and the safelight accent are the contract between them.
    """
    html = client.get("/").text
    favicon = client.get("/favicon.svg").text

    assert 'symbol id="i-brand"' in html
    assert 'class="brand-mark"' in html and "<use href=\"#i-brand\"/>" in html
    # Both use the same coordinate system and the same single accent colour.
    assert 'viewBox="0 0 48 48"' in favicon
    for colour in ("#f0a043", "#3a4048", "#111316"):
        assert colour in html, f"{colour} missing from the header mark"
        assert colour in favicon, f"{colour} missing from the favicon"
    # The old mark was two CSS pseudo-elements; make sure it is really gone.
    assert "brand-mark::after" not in client.get("/css/app.css").text


def test_brand_mark_and_favicon_share_one_symbol(client):
    """The header mark and the favicon must stay the same artwork.

    They are drawn twice on purpose — the favicon is a standalone file the
    browser fetches, and the header needs it inline to inherit nothing — so the
    shared 48-unit grid and the safelight accent are the contract between them.
    """
    html = client.get("/").text
    favicon = client.get("/favicon.svg").text

    assert 'symbol id="i-brand"' in html
    assert 'class="brand-mark"' in html and "<use href=\"#i-brand\"/>" in html
    # Both use the same coordinate system and the same single accent colour.
    assert 'viewBox="0 0 48 48"' in favicon
    for colour in ("#f0a043", "#3a4048", "#111316"):
        assert colour in html, f"{colour} missing from the header mark"
        assert colour in favicon, f"{colour} missing from the favicon"
    # The old mark was two CSS pseudo-elements; make sure it is really gone.
    assert "brand-mark::after" not in client.get("/css/app.css").text
