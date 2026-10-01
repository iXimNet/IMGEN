"""Freeing VRAM: the one action in the studio that really gives memory back.

Stopping a run and handing the card back are two different things, and the app
used to announce the second whenever it did the first. A cancel keeps the
weights where they are on purpose — that is what makes the next run start in a
second — and frees nothing at all, yet the studio toasted "VRAM from this run is
released" every time. These tests pin the replacement: an endpoint whose answer
is honest about whether the memory actually came back, a wait that gives up
rather than pretending, and a cancellation that no longer makes that claim.
"""

import gc
import re
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import imgen.engine as engine_module  # noqa: E402
from imgen.app import create_app  # noqa: E402
from imgen.engine import Engine, EngineError  # noqa: E402


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGEN_DEMO", "1")
    # The default wait is 15s, sized for a real card. Nothing here waits on a
    # GPU, so every test that would sit through it shortens it instead.
    monkeypatch.setattr(engine_module, "RELEASE_WAIT_S", 0.1)
    return create_app(demo=True, home=tmp_path)


def test_release_does_not_claim_memory_the_run_is_still_using():
    """A stage does not check the stop flag while it runs — one CUDA op cannot
    be interrupted from Python — so a release that arrives mid-stage has freed
    nothing yet and must say so. The old cancel path said it had anyway."""
    engine = Engine(demo=True)
    engine._busy = True
    engine._pipe = object()
    engine._loaded = {"model_key": "qwen-image-2.1", "loader": "bf16"}

    out = engine.release(timeout=0.05)

    assert out["released"] is False
    assert out["stopping"] is True
    assert out["freed_gb"] is None
    # Nothing was unloaded either: dropping the pipeline here would free nothing
    # (the worker holds its own reference) and would leave the studio reporting
    # a loaded model next to a job that is still going.
    assert engine._pipe is not None
    assert engine._loaded is not None
    # The stop was still requested — that is the half that does take effect.
    assert engine._cancel.is_set() is True

    engine._busy = False


def test_release_unloads_once_the_run_has_actually_ended():
    engine = Engine(demo=True)
    engine._busy = True
    engine._pipe = object()
    engine._loaded = {"model_key": "qwen-image-2.1", "loader": "bf16"}

    def finish():
        time.sleep(0.1)
        engine._busy = False

    threading.Thread(target=finish, daemon=True).start()
    out = engine.release(timeout=5.0)

    assert out["released"] is True
    assert out["stopping"] is False
    assert engine._pipe is None
    assert engine._loaded is None


def test_release_frees_an_idle_engine_too():
    """Freeing the card is not only something you do about a run: nothing is
    running, the weights are resident, and the card is wanted elsewhere."""
    engine = Engine(demo=True)
    engine._pipe = object()
    engine._loaded = {"model_key": "qwen-image-2.1", "loader": "bf16"}

    out = engine.release(timeout=0.05)

    assert out["released"] is True
    assert engine._loaded is None


def test_release_reports_what_it_actually_gave_back(monkeypatch):
    """The number in the toast is measured, not assumed."""
    engine = Engine(demo=True)
    engine._pipe = object()
    engine._loaded = {"model_key": "qwen-image-2.1", "loader": "bf16"}
    readings = iter(
        [
            {"used_gb": 20.5, "total_gb": 31.8},
            {"used_gb": 1.6, "total_gb": 31.8},
        ]
    )
    monkeypatch.setattr(engine, "vram_usage", lambda: next(readings, {"used_gb": 1.6, "total_gb": 31.8}))

    out = engine.release(timeout=0.05)

    assert out["freed_gb"] == pytest.approx(18.9, abs=0.01)


def test_dropping_the_reference_is_not_enough_to_return_the_memory(monkeypatch):
    """`empty_cache()` only returns blocks the allocator still holds; it cannot
    free memory live tensors are using, and a diffusers pipeline is a web of
    reference cycles that stays alive until it is collected. Without the
    collection step the cache is emptied and nothing is given back."""
    collected = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: collected.append(1))

    Engine(demo=True).unload()

    assert collected, "unload must collect cycles before emptying the cache"


def test_a_release_during_a_long_stage_answers_stopping_over_http(app, monkeypatch):
    """The HTTP answer carries the honest half so the studio can keep asking
    instead of claiming a release that has not happened — ten minutes inside a
    2K conditioning pass is exactly this case."""
    gate = threading.Event()
    engine = app.state.engine

    def stuck(request, callback=None):
        # Mirrors the real contract: `generate` owns the busy flag, and the
        # stage in flight does not check the cancel flag — which is the whole
        # reason a release can come back "stopping".
        engine._busy = True
        try:
            gate.wait(10)
            raise EngineError("Cancelled.", "CANCELLED")
        finally:
            engine._busy = False

    monkeypatch.setattr(engine, "generate", stuck)

    with TestClient(app) as client:
        job = client.post("/api/jobs", data={"mode": "generate", "prompt": "stuck"}).json()
        for _ in range(60):
            if engine.busy:
                break
            time.sleep(0.05)
        assert engine.busy is True

        out = client.post("/api/engine/release").json()

        assert out["ok"] is True
        assert out["released"] is False
        assert out["stopping"] is True
        assert "vram" in out and "waited_ms" in out

        gate.set()
        for _ in range(60):
            if client.get(f"/api/jobs/{job['id']}").json()["status"] != "running":
                break
            time.sleep(0.05)

    # Once the stage returned, the second ask gets the memory.
    out = client.post("/api/engine/release").json()
    assert out["released"] is True


def test_release_endpoint_works_when_nothing_was_ever_loaded(app):
    with TestClient(app) as client:
        out = client.post("/api/engine/release").json()

    assert out["ok"] is True
    assert out["released"] is True
    assert out["stopping"] is False


def test_the_cancel_toast_no_longer_claims_the_memory_was_freed(app):
    """The false line, in both languages. A cancel keeps the weights resident —
    that is the point of a resident load — so it freed nothing, and saying it
    did is what sent people to the console to restart the server."""
    with TestClient(app) as client:
        i18n = client.get("/js/i18n.js").text

    assert "这一轮的显存已经释放" not in i18n
    assert "VRAM from this run is released" not in i18n
    # And the replacement points at the action that does free it.
    assert "releaseVram" in i18n


def test_release_is_offered_where_a_stalled_run_is_watched(app):
    """Two hosts, one action: the progress block, where someone looks when a run
    has gone quiet, and the environment popover, for when nothing is running and
    the card itself is the thing to hand back."""
    with TestClient(app) as client:
        page = client.get("/").text

    assert 'id="releaseRow"' in page
    assert 'id="releaseEnvRow"' in page
    # The popover row sits under the device readout it belongs to.
    assert page.index('id="releaseEnvRow"') > page.index('id="envDevice"')


def test_the_studio_only_claims_a_release_when_the_server_says_so(app):
    """The front end acts on `released`, not on the fact that it sent a request.
    A source scan, because the honesty lives in the shape of the branch."""
    with TestClient(app) as client:
        js = client.get("/js/app.js").text

    body = js[js.index("async function releaseVram"):]
    body = body[: body.index("\n  }\n")]
    assert re.search(r"if \(out\.released\)", body), body[:400]
    # The success toast sits inside that branch: a "stopping" answer must not
    # reach it.
    success = body.index("if (out.released)")
    assert body.index("toastReleased") > success
    assert body.index("await new Promise") > success
