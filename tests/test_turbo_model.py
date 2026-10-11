"""Qwen-Image-2.1-Turbo — same pipeline, a recipe of its own.

The first catalog entries that are not simply "the base weights at another
precision". They share the pipeline class and the call signature, so the loader
is untouched, but they are distilled: eight steps and a schedule the checkpoint
carries inside its own ``model_index.json``. Two releases carry it — the BF16
checkpoint and the community INT8 conversion of it — so the recipe is declared
twice and has to survive both a loader switch and a disk sniff.

Two things have to hold, and the second is the reason most of this file exists:

* a run has to *use* that recipe — the installed diffusers does not read the
  checkpoint's schedule key, so a run that trusted the step field alone would be
  a 40-step run wearing an 8-step label;
* a caller asking for something else has to be unable to override it — the panel
  is one of three callers, and it is the only one that can be seen. Locking in
  the client alone would leave the API and a restored history record free to ask
  for a recipe the checkpoint was not trained for.
"""

import json
import time

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from imgen.app import create_app  # noqa: E402
from imgen.constants import MODELS, TURBO_SIGMAS  # noqa: E402
from imgen.engine import Engine, model_sigmas  # noqa: E402
from imgen.hub import sniff_snapshot_model  # noqa: E402

TURBO = "qwen-image-2.1-turbo"
TURBO_INT8 = "image21-turbo-int8"
# Both carry the distilled recipe; nothing else in the catalog is allowed to.
DISTILLED = (TURBO, TURBO_INT8)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGEN_DEMO", "1")
    app = create_app(demo=True, home=tmp_path)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def cuda_client(tmp_path, monkeypatch):
    """The same demo client, on an engine that believes it has an NVIDIA card.

    The CUDA gate runs before the demo branch, so an INT8 entry cannot be
    exercised without it — and it must not depend on the machine running the
    tests actually owning a GPU.
    """
    monkeypatch.setenv("IMGEN_DEMO", "1")
    monkeypatch.setattr(
        "imgen.engine.probe",
        lambda: {"device": "cuda", "cuda": True, "vram_gb": 32.0},
    )
    app = create_app(demo=True, home=tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def _generate(client, **form):
    payload = {"mode": "generate", "prompt": "turbo probe"}
    payload.update(form)
    response = client.post("/api/jobs", data=payload)
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    item = None
    for _ in range(80):
        item = client.get(f"/api/jobs/{job_id}").json()
        if item["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert item and item["status"] == "succeeded", item
    return item


# ---------------------------------------------------------------------------
# The catalog entry
# ---------------------------------------------------------------------------


def test_turbo_reuses_the_bf16_loader_and_declares_its_recipe():
    """No new loader, no new pipeline class — the difference is the recipe."""
    spec = MODELS[TURBO]
    assert spec["loader"] == "bf16"
    assert spec["precision"] == "BF16"
    assert spec["locked_params"] == {"steps": 8, "true_cfg_scale": 1.0}
    assert spec["sigmas_from"] == "sample_sigmas"
    assert spec["sigmas_fallback"] == TURBO_SIGMAS
    # Eight values, and a schedule that actually descends to noise.
    assert len(spec["sigmas_fallback"]) == 8
    assert spec["sigmas_fallback"][0] == pytest.approx(1.0)
    assert 0 < spec["sigmas_fallback"][-1] < spec["sigmas_fallback"][-2]
    # The licence is a compliance item, so it belongs in the copy the studio
    # shows, not only in a model card someone has to go and read.
    assert "非商用" in spec["notes_zh"]
    assert "non-commercial" in spec["notes_en"]


def test_the_catalog_reaches_the_panel_with_its_locks(client):
    """`model_catalog` is the panel's only source for the locked values."""
    boot = client.get("/api/bootstrap").json()
    catalog = boot["model_catalog"]
    assert catalog[TURBO]["locked_params"] == {"steps": 8, "true_cfg_scale": 1.0}
    # Every entry that is not distilled must stay lock-free, or the panel would
    # freeze controls that the engine still honours.
    for key, spec in catalog.items():
        if key not in DISTILLED:
            assert "locked_params" not in spec, key


def test_the_quantised_turbo_declares_both_halves():
    """Turbo's recipe reachable through the INT8 loader — the entry is the
    intersection of two families, and neither half may be dropped.

    Losing the recipe would run the distilled weights on the base schedule;
    losing the loader would hand quantised shards to the BF16 path.
    """
    spec = MODELS[TURBO_INT8]
    assert spec["loader"] == "int8"
    assert spec["precision"] == "INT8"
    assert spec["int8"] is True
    assert spec["requires_cuda"] is True
    # Declared exactly as the BF16 Turbo declares it — same literals, so the two
    # cannot drift into two different recipes.
    assert spec["locked_params"] == MODELS[TURBO]["locked_params"]
    assert spec["sigmas_from"] == MODELS[TURBO]["sigmas_from"]
    assert spec["sigmas_fallback"] == MODELS[TURBO]["sigmas_fallback"] == TURBO_SIGMAS
    assert "非商用" in spec["notes_zh"]
    assert "non-commercial" in spec["notes_en"]
    # The two sources do not share a namespace for this owner, so both ids are
    # spelled out rather than assumed from the BF16 entry.
    assert spec["repos"] == {
        "huggingface": "ixim/Image21-Turbo-INT8",
        "modelscope": "iximbox/Image21-Turbo-INT8",
    }


def test_the_catalog_quotes_the_same_size_convention_for_every_model():
    """`approx_gb` is decimal GB throughout, so the entries are comparable."""
    assert MODELS["qwen-image-2.1"]["approx_gb"] == pytest.approx(33.1, abs=0.05)
    # 30.21 GiB of weight files on the hub, on the same convention.
    assert MODELS[TURBO]["approx_gb"] == pytest.approx(32.4, abs=0.05)


# ---------------------------------------------------------------------------
# The lock holds against every caller
# ---------------------------------------------------------------------------


def test_a_request_for_more_steps_cannot_talk_turbo_out_of_its_recipe(client):
    """The panel, a reused history record and a direct API call all land here."""
    item = _generate(
        client,
        model_key=TURBO,
        steps="40",
        true_cfg_scale="4.0",
        use_kv_cache="false",
    )
    params = item["params"]
    assert params["steps"] == 8
    assert params["true_cfg_scale"] == pytest.approx(1.0)
    # Only the recipe is locked. KV cache is a memory/speed choice the run still
    # owns, and the caller's answer is kept.
    assert params["use_kv_cache"] is False


def test_other_models_still_honour_the_fields_they_are_given(client):
    """The lock is per model; the base model must be untouched by it."""
    item = _generate(client, model_key="qwen-image-2.1", steps="23", true_cfg_scale="3.5")
    assert item["params"]["steps"] == 23
    assert item["params"]["true_cfg_scale"] == pytest.approx(3.5)


def test_the_quantised_turbo_is_locked_against_every_caller(cuda_client):
    """The lock belongs to the recipe, so adding a loader did not lose it.

    The same request that the BF16 entry refuses has to be refused here, or the
    INT8 release would run its distilled weights on the base schedule.
    """
    item = _generate(
        cuda_client,
        model_key=TURBO_INT8,
        steps="40",
        true_cfg_scale="4.0",
        use_kv_cache="false",
    )
    params = item["params"]
    assert params["model_key"] == TURBO_INT8
    assert params["steps"] == 8
    assert params["true_cfg_scale"] == pytest.approx(1.0)
    # Still only the recipe: the memory/speed choice stays the caller's.
    assert params["use_kv_cache"] is False


# ---------------------------------------------------------------------------
# Reading the schedule off the checkpoint
# ---------------------------------------------------------------------------


def _snapshot(root, sigmas=...):
    root.mkdir(parents=True, exist_ok=True)
    index = {"_class_name": "QwenImage21Pipeline"}
    if sigmas is not ...:
        index["sample_sigmas"] = sigmas
    (root / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_the_checkpoints_own_schedule_wins(tmp_path):
    """A checkpoint that ships a recipe is read, not second-guessed."""
    snapshot = _snapshot(tmp_path / "turbo", [1.0, 0.5, 0.1])
    assert model_sigmas(str(snapshot), MODELS[TURBO]) == [1.0, 0.5, 0.1]


@pytest.mark.parametrize(
    "sigmas",
    [
        ...,  # the key is absent altogether
        [],  # present but empty
        [1.0],  # a single value is not a schedule
        "1.0, 0.5",  # a string, not a list
        [1.0, "noise"],  # a list that is not all numbers
    ],
)
def test_an_unusable_schedule_falls_back_to_the_declared_list(tmp_path, sigmas):
    """A conversion that dropped or mangled the key must not fail a render."""
    snapshot = _snapshot(tmp_path / "broken", sigmas)
    assert model_sigmas(str(snapshot), MODELS[TURBO]) == TURBO_SIGMAS


def test_a_missing_or_corrupt_index_still_yields_the_declared_list(tmp_path):
    absent = tmp_path / "absent"
    assert model_sigmas(str(absent), MODELS[TURBO]) == TURBO_SIGMAS

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "model_index.json").write_text("{not json", encoding="utf-8")
    assert model_sigmas(str(corrupt), MODELS[TURBO]) == TURBO_SIGMAS


def test_models_without_a_recipe_ask_for_nothing(tmp_path):
    """Every other model returns `None`, which is what keeps them unchanged."""
    snapshot = _snapshot(tmp_path / "base", [1.0, 0.5])
    for key in ("qwen-image-2.1", "image21-int8", "image21-int4"):
        assert model_sigmas(str(snapshot), MODELS[key]) is None


def test_both_distilled_entries_read_one_recipe_from_one_checkpoint(tmp_path):
    """One reader, two entries: whatever the file says, both halves agree.

    They are the same weights at two precisions, so a schedule read differently
    by the two entries would mean the entry, not the checkpoint, is deciding.
    """
    snapshot = _snapshot(tmp_path / "turbo", [1.0, 0.7, 0.3])
    bf16 = model_sigmas(str(snapshot), MODELS[TURBO])
    int8 = model_sigmas(str(snapshot), MODELS[TURBO_INT8])
    assert bf16 == int8 == [1.0, 0.7, 0.3]


# ---------------------------------------------------------------------------
# Handing the schedule to a pipeline
# ---------------------------------------------------------------------------


class _RecipePipe:
    """Shaped like `QwenImage21Pipeline.__call__`: it declares `sigmas`.

    It also honours the step hook the way the real sampler does — one call per
    step — so the progress stream can be checked against the schedule rather
    than merely against the field.
    """

    def __init__(self):
        self.vae = None
        self.sigmas = "unset"
        self.steps = None

    def __call__(
        self,
        prompt,
        num_inference_steps,
        true_cfg_scale,
        num_images_per_prompt,
        generator,
        sigmas=None,
        width=None,
        height=None,
        callback_on_step_end=None,
    ):
        self.sigmas = sigmas
        self.steps = num_inference_steps
        if callback_on_step_end is not None:
            payload = {}
            for step in range(num_inference_steps):
                payload = callback_on_step_end(self, step, 0.0, payload) or {}
        return type("Result", (), {"images": [Image.new("RGB", (8, 8))]})()


class _PlainPipe:
    """A pipeline that never heard of `sigmas` — the guard has to hold."""

    def __init__(self):
        self.vae = None
        self.seen = None

    def __call__(
        self,
        prompt,
        num_inference_steps,
        true_cfg_scale,
        num_images_per_prompt,
        generator,
        width=None,
        height=None,
    ):
        self.seen = {"prompt": prompt, "num_inference_steps": num_inference_steps}
        return type("Result", (), {"images": [Image.new("RGB", (8, 8))]})()


def _run(engine, pipe, sigmas=None):
    progress = []
    images = engine._run_pipe(
        pipe,
        prompt="p",
        negative_prompt=None,
        images=None,
        width=64,
        height=64,
        output_resolution=1024,
        steps=8,
        cfg=1.0,
        seed=1,
        kv_cache=True,
        n_images=1,
        follow_ref=False,
        callback=progress.append,
        sigmas=sigmas,
    )
    return images, progress


def test_the_schedule_is_passed_along_with_its_own_length():
    """The scheduler takes its step count from the array, so both travel."""
    engine = Engine(demo=True)
    pipe = _RecipePipe()
    _run(engine, pipe, [1.0, 0.5, 0.1])
    assert pipe.sigmas == [1.0, 0.5, 0.1]
    assert pipe.steps == 3


def test_the_progress_total_follows_the_schedule():
    """The bar must count the steps that ran, not the ones the field asked for.

    A run told `steps=8` but handed a three-value schedule executes three
    steps; a bar still counting to eight would crawl to 37% and then finish.
    """
    engine = Engine(demo=True)
    pipe = _RecipePipe()
    _images, progress = _run(engine, pipe, [1.0, 0.5, 0.1])

    steps = [event for event in progress if event["type"] == "generate_progress"]
    assert len(steps) == 3
    assert {event["total"] for event in steps} == {3}


def test_without_a_schedule_the_total_is_still_the_step_field():
    """The other half of the same rule: nothing changes for models without one."""
    engine = Engine(demo=True)
    pipe = _RecipePipe()
    _images, progress = _run(engine, pipe, None)

    steps = [event for event in progress if event["type"] == "generate_progress"]
    assert len(steps) == 8
    assert {event["total"] for event in steps} == {8}


def test_a_pipeline_without_the_parameter_is_never_handed_one():
    """The same signature probe every other optional kwarg goes through."""
    engine = Engine(demo=True)
    pipe = _PlainPipe()
    _run(engine, pipe, [1.0, 0.5, 0.1])
    assert pipe.seen == {"prompt": "p", "num_inference_steps": 8}


def test_no_schedule_leaves_the_step_field_in_charge():
    """Every model that declares no recipe keeps the old behaviour exactly."""
    engine = Engine(demo=True)
    pipe = _RecipePipe()
    _run(engine, pipe, None)
    assert pipe.sigmas is None
    assert pipe.steps == 8


# ---------------------------------------------------------------------------
# Telling the distilled checkpoints apart on disk
# ---------------------------------------------------------------------------


def test_a_distilled_snapshot_is_not_claimed_as_the_base_model(tmp_path):
    """Both are unquantised, so the only marker is the recipe key.

    Without this, a Turbo folder added by hand would be reported as
    Qwen-Image-2.1 and run on a 40-step schedule — the wrong model, silently.
    """
    def snapshot(root, payload):
        root.mkdir(parents=True, exist_ok=True)
        (root / "model_index.json").write_text(json.dumps(payload), encoding="utf-8")
        return root

    base = snapshot(tmp_path / "base", {"_class_name": "QwenImage21Pipeline"})
    assert sniff_snapshot_model(base) == "qwen-image-2.1"

    distilled = snapshot(
        tmp_path / "turbo",
        {"_class_name": "QwenImage21Pipeline", "sample_sigmas": TURBO_SIGMAS},
    )
    assert sniff_snapshot_model(distilled) == TURBO


def test_an_unusable_recipe_key_does_not_claim_the_distilled_model(tmp_path):
    """A stray single value is not evidence of a distilled checkpoint."""
    root = tmp_path / "odd"
    root.mkdir()
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImage21Pipeline", "sample_sigmas": [1.0]}),
        encoding="utf-8",
    )
    assert sniff_snapshot_model(root) == "qwen-image-2.1"


def test_a_quantised_turbo_snapshot_is_not_claimed_as_the_base_int8(tmp_path):
    """The quantised pair needs the same question the unquantised pair does.

    Both folders hold INT8 shards and neither name says which weights they are,
    so a Turbo conversion read as Image21-INT8 would run the distilled model on
    the 40-step base schedule: wrong pictures, no error, nothing in the log.
    """

    def snapshot(root, *, base_model=None, sigmas=...):
        root.mkdir(parents=True, exist_ok=True)
        meta = {"method": "bitsandbytes LLM.int8", "threshold": 6.0}
        if base_model is not None:
            meta["base_model"] = base_model
        (root / "conversion.json").write_text(json.dumps(meta), encoding="utf-8")
        index = {"_class_name": "QwenImage21Pipeline"}
        if sigmas is not ...:
            index["sample_sigmas"] = sigmas
        (root / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
        return root

    # What the published release carries: both witnesses agree.
    released = snapshot(
        tmp_path / "released",
        base_model="Qwen/Qwen-Image-2.1-Turbo",
        sigmas=TURBO_SIGMAS,
    )
    assert sniff_snapshot_model(released) == TURBO_INT8

    # Either witness on its own is enough, for a conversion that records one.
    assert (
        sniff_snapshot_model(
            snapshot(tmp_path / "only-base", base_model="Qwen/Qwen-Image-2.1-Turbo")
        )
        == TURBO_INT8
    )
    assert (
        sniff_snapshot_model(snapshot(tmp_path / "only-recipe", sigmas=TURBO_SIGMAS))
        == TURBO_INT8
    )

    # And the base INT8 release keeps its own answer, whichever way it is dressed.
    assert (
        sniff_snapshot_model(
            snapshot(tmp_path / "base", base_model="Qwen/Qwen-Image-2.1")
        )
        == "image21-int8"
    )
    assert sniff_snapshot_model(snapshot(tmp_path / "bare")) == "image21-int8"


# ---------------------------------------------------------------------------
# The panel mirrors what the engine will do
# ---------------------------------------------------------------------------


def test_the_panel_reads_the_locks_and_closes_the_controls(client):
    """A control that still moves while the run ignores it is the whole risk.

    The value has to be shown and the control has to stop taking input; either
    one alone leaves the panel describing a run that will not happen.
    """
    js = client.get("/js/app.js").text
    assert "model_catalog" in js, "the panel must read the catalog it is sent"
    assert "locked_params" in js
    assert "function applyLocks()" in js
    # Applied where state is written, not only where it is painted.
    assert js.count("applyLocks()") >= 3
    assert '$("steps").disabled = stepsLocked' in js
    assert '$("stepQuick").classList.toggle("hidden", stepsLocked)' in js
    assert '$("cfg").disabled = cfgLocked' in js
    # The value written into the panel comes from the catalog, never from a
    # number typed into the source.
    assert "locked.steps" in js
    # The form posts what the inputs hold, so every path that changes these
    # numbers has to reach the fields — one definition and at least two call
    # sites (applying a lock, and releasing one on a model switch).
    assert js.count("paintSamplingFields()") >= 3


def test_both_languages_can_explain_a_locked_control(client):
    """The note next to a disabled slider is the only thing that says why."""
    js = client.get("/js/i18n.js").text
    for key in ("stepsLocked", "cfgLocked", "cfgTipLocked"):
        assert js.count(f"{key}:") == 2, key
    # The tip has to name the consequence a reader would otherwise discover by
    # getting nothing back: a negative prompt that does not apply.
    assert "负向提示词不参与" in js
    assert "negative prompt has no effect" in js
