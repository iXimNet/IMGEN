import json
import re
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


def test_a_generate_run_ignores_reference_uploads(client):
    """A 生图 record must not carry references, even if the request sends them.

    The panel keeps `S.refs` while switching modes (a reference the user just
    picked should survive a trip through 生图, and `sendToEdit` sets the mode
    *before* adding the file), but the reference section is hidden in 生图 — so
    the files used to be posted by a run that could not show them. The engine
    ignored them all along (`images=refs if mode == "edit" else None`), which is
    what made it insidious: the picture was generated correctly, while the record
    saved the upload and the detail overlay listed a 参考图 that had no part in
    the result.

    Two layers are checked here — the server refuses to use them, and the public
    view refuses to publish them — plus the third (the client not sending them)
    by inspecting the source.
    """
    buf = BytesIO()
    Image.new("RGB", (64, 64), (200, 40, 40)).save(buf, format="PNG")
    buf.seek(0)
    response = client.post(
        "/api/jobs",
        data={
            "mode": "generate",
            "prompt": "a red apple on a wooden table",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
            "scale": "1k",
            "steps": "8",
        },
        # A stray reference on a generate job — exactly what a mode switch used
        # to produce.
        files=[("files", ("stale-ref.png", buf, "image/png"))],
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
    assert item["mode"] == "generate"

    # Nothing is published, so the overlay has no 参考图 group to render.
    assert item["ref_count"] == 0
    assert item["ref_urls"] == []
    # `ref_paths` is the stored form and is never sent to the browser, which is
    # also why the count above is the only thing a caller can see. Fetching a
    # reference URL must 404: the run has none.
    assert client.get(f"/api/refs/{job_id}/0").status_code == 404

    # The client does not send them in this mode to begin with.
    js = client.get("/js/app.js").text
    assert 'if (S.mode === "edit") {' in js, "the upload must be gated on the mode"
    gated = js[js.index('if (S.mode === "edit") {'):]
    gated = gated[: gated.index("return fd;")]
    assert 'fd.append("files"' in gated, "the append must sit inside the guard"


def test_send_to_edit_adds_a_reference_and_leaves_the_prompt_alone(client):
    """送到改图 must send a picture, not a prompt.

    The button used to run `if (prompt) $("prompt").value = prompt` after adding
    the reference, so opening a record's detail and sending its picture silently
    replaced whatever the user had typed — no confirmation, no undo, and the
    toast never mentioned it. It read "已作为参考图送到改图 / 在提示词里用「第一张图」
    指代它", i.e. it told the user to name the image themselves while the box had
    already been filled in.

    Restoring a record's prompt is what 复用参数 (`applyItemParams`) is for, and
    that one restores the negative prompt too. So the rule enforced here is that
    the box is not touched at all — the wrong half of a settings restore is worse
    than none of it.

    The behaviour lives in the DOM, which these tests do not drive; the contract
    is asserted against the source, the same way the mode gate above is.
    """
    js = client.get("/js/app.js").text

    assert "async function sendToEdit(url) {" in js, (
        "sendToEdit must take the picture url and nothing else"
    )

    body = js[js.index("async function sendToEdit(url) {"):]
    body = body[: body.index('toast("ok", tr("toastToEdit")')]
    assert '$("prompt")' not in body, "sendToEdit must not write the prompt box"
    assert "$(\"negative\")" not in body, "nor the negative prompt"
    # It still has to do the one thing it does.
    assert "addFiles(" in body, "the reference image is the whole point"


def test_send_to_edit_callers_pass_only_the_url(client):
    """Both call sites agree on the new signature.

    They did not agree before: the detail footer passed `item.prompt` while the
    stage toolbar passed `null`, so the same button meant two different things
    depending on where it was clicked. That asymmetry is what surfaced the bug,
    and it must not come back.
    """
    js = client.get("/js/app.js").text

    call = js[js.index("sendToEdit(detailFrame()"):]
    assert call.startswith("sendToEdit(detailFrame().url || item.image_url);"), (
        "the detail footer sends the picture on screen, and nothing else"
    )
    assert "sendToEdit(S.currentImage);" in js, (
        "the stage toolbar sends the canvas picture, and nothing else"
    )
    # Nothing anywhere may hand it a second argument again.
    calls = re.findall(r"sendToEdit\(([^;]*?)\);", js)
    assert calls, "the call sites must still be findable"
    offenders = [c for c in calls if "," in c]
    assert not offenders, f"sendToEdit takes one argument; got {offenders}"


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
    # Normalise line endings: this scans the source for `\n  }` block ends, and
    # a Windows checkout (or an editor that writes CRLF) would otherwise make
    # every block look unterminated.
    js = js.replace("\r\n", "\n").replace("\r", "\n")
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


def test_status_pill_is_not_width_capped(client):
    """The pill must size to its content, not to a number someone guessed.

    It names three things that vary per machine — device, model, weight hub — and
    a fixed cap therefore guarantees truncation on some configuration. A 340px cap
    cut "ModelScope 魔搭" down to "Model…" on the machine this was written on, and
    a machine with a longer card name loses more.

    The cap is replaced by content sizing plus `min-width`, so the pill only gives
    ground when the bar genuinely runs out of room, and by `overflow: hidden` so
    that when it does, the contents are clipped to the rounded edge instead of
    spilling past it.
    """
    css = client.get("/css/app.css").text
    pill = css[css.index(".status-pill {"):]
    pill = pill[: pill.index("}")]

    # No fixed ceiling: that is the actual defect.
    assert "max-width" not in pill, "a max-width cap is what truncated the text"
    # A floor so a squeezed pill still shows its dot and chevron, and clipping so
    # nothing escapes the rounded border.
    assert "min-width: 56px" in pill
    assert "overflow: hidden" in pill

    # Both text halves must be able to ellipsise. A `<b>` with `nowrap` and
    # default overflow is simply guillotined by the pill's clip.
    device = css[css.index(".status-pill b {"):]
    device = device[: device.index("}")]
    assert "text-overflow: ellipsis" in device
    assert "overflow: hidden" in device


def test_the_status_pill_does_not_claim_a_weight_source(client):
    """The pill must not print the hub as if it were where the weights came from.

    `S.hub` is the *preference* — the last source picked in the model sheet, or
    whatever is in the config. It is not the origin: `resolve_local_hub` on the
    server silently falls back to whichever source actually holds a complete
    snapshot, and `POST /api/jobs` logs "using weights from X (Y has no local
    copy)" while the pill would still have shown Y. So the one configuration the
    pill appeared to describe was exactly the one it could get wrong.

    Nothing is lost by dropping it: the sheet behind the pill renders
    `model.local_hub` (the real location), and every history record stores the
    resolved `hub` from the job request. Guard the removal, and guard the fact
    that those two places still carry the truth.
    """
    js = client.get("/js/app.js").text

    # The pill renders device + model, and nothing about the hub.
    pins = js[js.index("function renderPins"):]
    pins = pins[: pins.index("\n  }\n")]
    assert "$(\"pillDevice\").textContent" in pins
    assert "$(\"pillModel\").textContent" in pins
    assert "hubLabel" not in pins, "the pill must not name a weight source"

    # The real source is still reported by the model sheet …
    models = js[js.index("function renderModels"):]
    models = models[: models.index("\n  }\n")]
    assert "model.local_hub" in models, "the sheet must name where the weights are"

    # … and by the backend, which resolves the hub before writing the record.
    server = open("imgen/app.py", encoding="utf-8").read()
    assert "resolve_local_hub(model_key, requested_hub)" in server
    assert '"hub": hub,' in server, "the resolved hub should be stored on the job"


def test_the_narrow_bar_does_not_repeat_the_mode_switch(client):
    """<=900px has two 生图/改图 switches; one is a duplicate.

    The bottom bar carries its own switch next to the run button and is the only
    one of the two in reach there. Keeping the top bar's copy spent ~166px of the
    narrowest budget restating it, and was the reason the status pill had to
    shrink at all.
    """
    css = client.get("/css/app.css").text

    # The rule lives in the <=900px block, where the bottom bar appears.
    narrow = css[css.index("@media (max-width: 900px)"):]
    narrow = narrow[: narrow.index("@media (max-width: 560px)")]
    assert ".topbar .mode-switch { display: none; }" in narrow

    # And the bottom bar's copy — the one that stays — is wired up in the client.
    js = client.get("/js/app.js").text
    assert '$("modeGen2").onclick = () => setMode("generate");' in js
    assert '$("modeEdit2").onclick = () => setMode("edit");' in js


def test_preset_triggers_share_one_handler(client):
    """Both the toolbar button and the empty-stage card must open the popover.

    The card used to forward a synthetic `.click()` to the button. That opened
    the popover during the target phase, but the *original* click kept bubbling to
    `document`, whose outside-click rule did not treat the card as a trigger, so
    it closed the popover again inside the same event — the card looked dead.

    Forwarding is what breaks it, so the guard is that the card does not forward:
    both elements are wired to the same function.
    """
    js = client.get("/js/app.js").text

    assert "$(\"btnPresets\").onclick = togglePresets;" in js
    assert "$(\"startPresets\").onclick = togglePresets;" in js
    # No synthesised click anywhere — that is the shape of the bug.
    assert "$(\"startPresets\").onclick = () => $(\"btnPresets\").click()" not in js
    assert 'startPresets").onclick = () =>' not in js

    # The shared handler stops the event, which is what keeps `document`'s
    # outside-click rule from seeing the card at all.
    handler = js[js.index("const togglePresets"): js.index("$(\"btnPresets\").onclick")]
    assert "stopPropagation" in handler, "the shared handler must stop propagation"

    # And the outside-click rule is unchanged: it still excused the popover and
    # the two toolbar triggers, and still closes on anything else.
    doc = js[js.index('document.addEventListener("click"'):]
    guard = doc[:doc.index("closePops();")]
    for selector in (".pop", "#btnPresets", "#statusBtn"):
        assert f'closest("{selector}")' in guard, f"{selector} should stay excused"


def test_brand_wordmark_is_served_transparent(client):
    """The wordmark must be transparent artwork, and reversed for the dark bar.

    Both properties are easy to lose silently — re-exporting the file from the
    original (which has a flat light-grey plate) would put a pale rectangle on
    the header, and shipping the near-black original would make the name
    invisible on it (1.0:1 contrast).
    """
    import io

    from PIL import Image

    page = client.get("/").text
    assert 'class="brand-word"' in page
    assert "/brand/imgen-wordmark-dark.png" in page
    # The accessible name has to survive the swap from text to image.
    assert 'alt="IMGEN"' in page and 'data-i-alt="app"' in page

    for name in ("imgen-wordmark", "imgen-wordmark-dark"):
        res = client.get(f"/brand/{name}.png")
        assert res.status_code == 200, name
        im = Image.open(io.BytesIO(res.content))
        assert im.mode == "RGBA", f"{name} is {im.mode}, so it has no transparency"
        alpha = im.getchannel("A")
        lo, hi = alpha.getextrema()
        assert lo == 0, f"{name}: nothing is fully transparent — a plate is baked in"
        assert hi > 240, f"{name}: nothing is fully opaque — the strokes are washed out"

    # The header version is the light one, not the near-black original. Compare
    # only where the ink is actually opaque: transparent pixels carry whatever
    # colour the source had and would drag the average anywhere.
    import numpy as np

    dark = Image.open(io.BytesIO(client.get("/brand/imgen-wordmark-dark.png").content))
    arr = np.array(dark)
    ink = arr[arr[:, :, 3] > 200][:, :3]
    assert len(ink), "the dark variant has no visible ink"
    avg = ink.mean(axis=0)
    assert avg.min() > 200, f"the dark variant is not light enough for the header ({avg})"

    plain = Image.open(io.BytesIO(client.get("/brand/imgen-wordmark.png").content))
    plain_ink = np.array(plain)[np.array(plain)[:, :, 3] > 200][:, :3]
    assert plain_ink.mean(axis=0).max() < 60, "the README variant should stay near-black"


def test_weight_dirs_normalisation(tmp_path):
    from imgen.config import ConfigStore, normalize_weight_dirs
    from imgen.paths import AppPaths

    assert normalize_weight_dirs(None) == []
    assert normalize_weight_dirs("D:/weights") == ["D:/weights"]
    assert normalize_weight_dirs(["a", "a", "  ", "b", 5]) == ["a", "b", "5"]

    store = ConfigStore(AppPaths(tmp_path))
    store.save({"extra_weight_dirs": ["a", "a"]})
    assert store.load()["extra_weight_dirs"] == ["a"]
    # A hand-edited config file cannot wedge the store into a non-list.
    store.paths.config_file.write_text(
        '{"extra_weight_dirs": "just-a-string"}', encoding="utf-8"
    )
    assert store.load()["extra_weight_dirs"] == ["just-a-string"]


def test_extra_weight_dirs_roundtrip(client, tmp_path):
    """check-dir guards the gate; PUT /api/config persists and echoes the list."""
    folder = tmp_path / "my-weights"
    folder.mkdir()

    bad = client.post("/api/weights/check-dir", json={"path": str(tmp_path / "nope")}).json()
    assert bad["ok"] is False
    assert bad["reason"] == "missing"

    good = client.post("/api/weights/check-dir", json={"path": str(folder)}).json()
    assert good["ok"] is True
    assert good["path"] == str(folder)
    assert good["models"] == []

    saved = client.put(
        "/api/config", json={"extra_weight_dirs": [str(folder), str(folder)]}
    ).json()
    assert saved["extra_weight_dirs"] == [str(folder)]

    extras = client.get("/api/bootstrap").json()["storage"]["extra_dirs"]
    assert [item["path"] for item in extras] == [str(folder)]
    assert extras[0]["ok"] is True

    # Removing goes through the same endpoint and clears the search layer.
    saved = client.put("/api/config", json={"extra_weight_dirs": []}).json()
    assert saved["extra_weight_dirs"] == []
    assert client.get("/api/bootstrap").json()["storage"]["extra_dirs"] == []



def _run_generate(client, **overrides) -> dict:
    """Submit one generation and wait for a terminal state, returning the row."""
    data = {"mode": "generate", "prompt": "multi frame probe", "steps": "4"}
    data.update(overrides)
    job_id = client.post("/api/jobs", data=data).json()["id"]
    item = None
    for _ in range(120):
        item = client.get(f"/api/jobs/{job_id}").json()
        if item["status"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert item and item["status"] == "succeeded", item
    return item


def test_multi_image_run_exposes_every_frame(client):
    """`num_images > 1` produces N pictures; all N must be reachable by URL.

    The extra frames have no row of their own, so the only way the browser can
    learn about them is this contract. Before it existed the studio showed one
    picture out of four while the other three sat in `outputs/` unreferenced.
    """
    item = _run_generate(client, num_images="3")
    assert item["params"]["num_images"] == 3
    assert item["output_count"] == 3
    assert len(item["image_urls"]) == 3
    assert len(item["thumb_urls"]) == 3
    # The first frame keeps the plain id, so older callers still work.
    assert item["image_urls"][0] == item["image_url"] == f"/api/outputs/{item['id']}"
    assert item["image_urls"][1:] == [
        f"/api/outputs/{item['id']}_1",
        f"/api/outputs/{item['id']}_2",
    ]
    for url in item["image_urls"] + item["thumb_urls"]:
        assert client.get(url).status_code == 200, url
    # Each frame is its own picture, not the same one served three times.
    bodies = {client.get(url).content for url in item["image_urls"]}
    assert len(bodies) == 3


def test_extra_frame_paths_never_reach_the_browser(client):
    """The contract carries URLs; absolute paths stay on the server."""
    item = _run_generate(client, num_images="2")
    assert "extra_images" not in item["params"], "paths must be stripped"
    blob = json.dumps(item)
    assert "extra_images" not in blob
    assert "image_path" not in blob and "thumb_path" not in blob
    assert ":\\" not in blob and ":/" not in blob, "no filesystem paths may leak"


def test_single_image_run_still_reports_one(client):
    """The default path keeps its old shape: one frame, one URL."""
    item = _run_generate(client)
    assert item["output_count"] == 1
    assert item["image_urls"] == [f"/api/outputs/{item['id']}"]
    assert item["image_url"] == f"/api/outputs/{item['id']}"


def test_output_route_refuses_names_it_did_not_make(client, tmp_path):
    """The extra-frame fallback joins a name onto a folder, so it must be strict.

    Without the shape check `..\\secret` walks out of `outputs/` and reads any
    `.png` next to it — reachable the moment the server is bound to a LAN
    address.
    """
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-image")

    for probe in (
        "..%5Csecret",
        "..%2Fsecret",
        f"..%5C{outside.name}",
        "0e275912caf441ea_0",  # frame numbers start at 1
        "0e275912caf441ea_1_2",
        "not-a-job-id",
        "0E275912CAF441EA",  # ids are lower-case hex
    ):
        assert client.get(f"/api/outputs/{probe}").status_code == 404, probe
        assert client.get(f"/api/thumbs/{probe}").status_code == 404, probe

    # A well-formed extra-frame name that simply does not exist is still a 404,
    # not a 500 — the shape check must not turn a miss into a crash.
    assert client.get("/api/outputs/0123456789abcdef_7").status_code == 404


def test_deleting_a_multi_image_run_removes_every_frame(client, tmp_path):
    """Delete has to reach the frames it cannot see in a column.

    `params.extra_images` is the only record of them; a delete that ignores it
    leaves orphan files behind forever.
    """
    item = _run_generate(client, num_images="3")
    job_id = item["id"]
    outputs = tmp_path / "outputs"
    thumbs = tmp_path / "thumbs"

    made = sorted(p.name for p in outputs.glob(f"{job_id}*"))
    assert len(made) == 3, made
    assert len(list(thumbs.glob(f"{job_id}*"))) == 3

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert list(outputs.glob(f"{job_id}*")) == []
    assert list(thumbs.glob(f"{job_id}*")) == []


def test_older_records_gain_frames_without_a_migration(client, tmp_path):
    """Rows written before the contract existed must light up as they are.

    The frames were always in `params.extra_images`; the API just never read
    them. Nothing on disk changes, so no migration is needed.
    """
    from imgen.history import History
    from imgen.paths import AppPaths

    paths = AppPaths(tmp_path)
    hist = History(paths)
    job_id = "aaaabbbbccccdddd"
    image = Image.new("RGBA", (32, 32), (10, 20, 30, 255))
    image_path, thumb_path = hist.save_image(job_id, image)
    extras = []
    for index in (1, 2):
        extra_id = f"{job_id}_{index}"
        path, thumb = hist.save_image(extra_id, image)
        extras.append({"id": extra_id, "image_path": path, "thumb_path": thumb})
    hist.create(
        {
            "id": job_id,
            "mode": "generate",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
            "prompt": "legacy row",
            "params": {"num_images": 3, "extra_images": extras},
            "status": "succeeded",
            "image_path": image_path,
            "thumb_path": thumb_path,
        }
    )

    row = client.get(f"/api/jobs/{job_id}").json()
    assert row["output_count"] == 3
    assert row["image_urls"][2] == f"/api/outputs/{job_id}_2"
    assert "extra_images" not in row["params"]


def test_count_stepper_shares_the_engines_bound(client):
    """The panel's 1..4 stepper and the engine's clamp must agree.

    The bound lives in MAX_NUM_IMAGES so the two cannot drift: a UI that offers
    a number the engine silently reduces is a lie about what will happen.
    """
    from imgen.constants import MAX_NUM_IMAGES

    page = client.get("/").text
    js = client.get("/js/app.js").text

    # The control exists and publishes the same bound the engine clamps to.
    assert 'id="nImages"' in page
    assert 'id="nImagesMinus"' in page and 'id="nImagesPlus"' in page
    # It is a stepper, not a free text field: `readonly` is what makes the two
    # buttons and the arrow keys the only ways in.
    assert 'readonly' in page[page.index('id="nImages"'): page.index('id="nImages"') + 160]

    # The client reads the bound from bootstrap instead of hardcoding it.
    assert client.get("/api/bootstrap").json()["defaults"]["max_images"] == MAX_NUM_IMAGES
    assert "maxImages()" in js
    assert "MAX_NUM_IMAGES" not in js  # server-side only; the client uses the payload

    # And the engine's clamp is the shared constant, not a literal 4.
    engine = open("imgen/engine.py", encoding="utf-8").read()
    assert "min(MAX_NUM_IMAGES" in engine
    assert "min(4," not in engine


def test_bootstrap_reports_the_sampling_defaults(client):
    """The panel labels values as "the default", so it reads them from the
    engine rather than repeating the numbers in two places."""
    from imgen.constants import DEFAULT_CFG, DEFAULT_STEPS

    defaults = client.get("/api/bootstrap").json()["defaults"]
    assert defaults["steps"] == DEFAULT_STEPS
    assert defaults["cfg"] == DEFAULT_CFG
    assert defaults["max_images"] == 4


def test_the_seed_well_states_its_own_contents(client):
    """The seed field is the one control whose *emptiness* is a valid, meaningful
    setting, so the panel has to say which of the two states it is in.

    It used to carry a static "blank = random" caption pinned to the right edge:
    it stayed there over a pinned seed, claiming the field was empty while a
    number sat in it, and it explained nothing about how to get back. The two
    states are now exclusive and each owns one edge — the hint stands in for the
    value, the clear button replaces it — and the guard is that a single
    `has-value` class on the wrapper drives both, so no combination of them can
    be on screen at once.
    """
    page = client.get("/").text
    css = client.get("/css/app.css").text.replace("\r\n", "\n")
    js = client.get("/js/app.js").text.replace("\r\n", "\n")
    i18n = client.get("/js/i18n.js").text

    assert 'id="seedWrap"' in page and 'class="numwrap seedwrap"' in page
    assert 'id="seedHint"' in page and 'data-i="seedHint"' in page
    assert 'id="seedClear"' in page
    # The hint is announced as the field's description, so the meaning reaches a
    # screen reader and not only the eye.
    assert 'aria-describedby="seedHint"' in page

    # One class, both halves: nothing else may decide what the field looks like.
    assert ".seedwrap.has-value .seed-hint { display: none; }" in css
    assert ".seedwrap.has-value .seed-clear { display: grid; }" in css
    # `display`, not `opacity` — a clear button that is only invisible is still in
    # the tab order, and reachable with nothing on screen to explain it.
    assert "opacity" not in css[css.index(".seed-clear {"): css.index(".seed-clear {") + 400]

    # JS writes the value in three places; only one of them is typing, so all of
    # them have to re-read the state or the panel contradicts itself.
    assert js.count('$("seed").value =') == 3
    assert js.count("renderSeed();") == 4  # input, clear click, reuse, snapshot
    assert "$(\"seedClear\").onclick" in js
    # Clearing has to hand the caret back: the click took the focus.
    assert "$(\"seed\").focus();" in js
    # The clear button is a real control, named in both languages.
    assert "seedClear" in i18n and i18n.count("seedClear:") == 2
    assert "$(\"seedClear\").setAttribute(\"aria-label\", tr(\"seedClear\"))" in js
