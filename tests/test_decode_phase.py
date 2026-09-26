"""VAE decode phase events + startup reconciliation of stale running jobs.

The decode step of a pipeline run used to be a silent stretch: the studio
froze on the last sampler update ("100%, ~0s left") while small-GPU INT4 runs
decoded on the CPU for minutes. These tests pin the pieces that fix that: the
engine announces the decode phase exactly once (flagging the CPU-decode case),
the wrapper never damages the VAE it borrows, the demo path keeps the same
event shape, and a restart never leaves phantom "running" rows behind.
"""

import time

import pytest
import torch
from PIL import Image

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from imgen.app import create_app  # noqa: E402
from imgen.engine import Engine, EngineError  # noqa: E402
from imgen.history import History, new_id  # noqa: E402
from imgen.int4_runtime import attach_split_vae  # noqa: E402
from imgen.paths import AppPaths  # noqa: E402


class _ClassVae:
    """decode is a plain class method — the common BF16/INT8 shape."""

    def decode(self, latents, *args, **kwargs):
        return "decoded"


class _InstanceVae:
    """decode is an instance attribute — the INT4 group-offload shape, where
    apply_offload() installs a CPU-decode wrapper onto the module."""


class _FakePipe:
    """Callable good enough for inspect.signature and the decode hook."""

    def __init__(self, vae):
        self.vae = vae
        self.decode_calls = 0

    def __call__(self, **kwargs):
        # Two decode invocations: the phase must be announced only once.
        self.vae.decode("latent-a")
        self.vae.decode("latent-b")
        self.decode_calls = 2
        return type("Result", (), {"images": [Image.new("RGB", (8, 8))]})()


class _FullPipe:
    """A pipe with all three silent stages, shaped like the real one.

    ``encode_prompt`` is a class method (so the shim must shadow it and then
    step aside), and it is called twice — once for the prompt and once for the
    negative prompt — which must still read as one stage.
    """

    def __init__(self, vae):
        self.vae = vae
        self.encode_calls = 0
        self.encode_image_calls = 0

    def encode_prompt(self, prompt, **kwargs):
        self.encode_calls += 1
        return f"embeds:{prompt}"

    def _encode_vae_image(self, image, generator):
        self.encode_image_calls += 1
        return f"latents:{image}"

    def __call__(self, **kwargs):
        self.encode_prompt("sunset beach")
        self.encode_prompt("")
        self._encode_vae_image("reference", None)
        self.vae.decode("latents")
        return type("Result", (), {"images": [Image.new("RGB", (8, 8))]})()


def _run(engine, pipe, events):
    def callback(event):
        events.append(event)

    return engine._run_pipe(
        pipe,
        prompt="p",
        negative_prompt=None,
        images=None,
        width=64,
        height=64,
        output_resolution=0,
        steps=2,
        cfg=1.0,
        seed=1,
        kv_cache=False,
        n_images=1,
        follow_ref=False,
        callback=callback,
    )


def test_run_pipe_announces_decode_once_and_restores_bound_decode():
    engine = Engine(demo=True)
    events = []
    pipe = _FakePipe(_ClassVae())

    result = _run(engine, pipe, events)

    phases = [e for e in events if e["type"] == "generate_phase"]
    assert len(phases) == 1
    assert phases[0] == {"type": "generate_phase", "phase": "decode", "slow": False}
    assert pipe.decode_calls == 2
    assert len(result) == 1
    # The shadowing instance attribute is gone; the class method shows through.
    assert "decode" not in vars(pipe.vae)
    assert pipe.vae.decode("x") == "decoded"


def test_run_pipe_flags_cpu_decode_and_preserves_instance_override():
    engine = Engine(demo=True)
    # INT4 group offload: the load path left its own instance-level decode.
    engine._loaded = {"loader": "int4", "runtime": {"offload": "group"}}

    def cpu_decode(latents, *args, **kwargs):
        return "cpu-decoded"

    vae = _InstanceVae()
    vae.decode = cpu_decode
    events = []
    pipe = _FakePipe(vae)

    _run(engine, pipe, events)

    phases = [e for e in events if e["type"] == "generate_phase"]
    assert len(phases) == 1
    assert phases[0]["slow"] is True
    # The borrowed instance-level wrapper survived the run untouched.
    assert vars(vae)["decode"] is cpu_decode
    assert vae.decode("x") == "cpu-decoded"


def test_demo_engine_walks_every_stage_in_order():
    """Demo mirrors a real run: conditioning and VAE encode land before the
    first sampler step, the decode after the last one."""
    engine = Engine(demo=True)
    events = []

    def callback(event):
        events.append(event)

    engine.generate(
        {
            "mode": "generate",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
            "prompt": "decode phase probe",
            "steps": 2,
        },
        callback=callback,
    )

    kinds = [e["type"] for e in events]
    phases = [e["phase"] for e in events if e["type"] == "generate_phase"]
    assert phases == ["condition", "encode", "decode"]
    first_progress = kinds.index("generate_progress")
    decode_at = kinds.index("generate_phase", kinds.index("generate_progress"))
    assert kinds.index("generate_phase") < first_progress
    assert decode_at > max(i for i, kind in enumerate(kinds) if kind == "generate_progress")
    progress_steps = [e["step"] for e in events if e["type"] == "generate_progress"]
    assert progress_steps == [1, 2]


def test_every_silent_stage_is_announced_once_and_restored():
    """Each stage is announced the first time it is really entered, and the
    hooks the pipeline owns come back exactly as they were."""
    engine = Engine(demo=True)
    events = []
    pipe = _FullPipe(_ClassVae())

    _run(engine, pipe, events)

    phases = [e for e in events if e["type"] == "generate_phase"]
    assert [p["phase"] for p in phases] == ["condition", "encode", "decode"]
    assert all(p["slow"] is False for p in phases)
    # Two encode_prompt calls are still one stage, announced once.
    assert pipe.encode_calls == 2
    assert len([p for p in phases if p["phase"] == "condition"]) == 1
    # The class methods show through again.
    assert "encode_prompt" not in vars(pipe)
    assert "_encode_vae_image" not in vars(pipe)
    assert pipe.encode_prompt("x") == "embeds:x"


def test_every_silent_stage_is_timed_in_the_console(capsys):
    engine = Engine(demo=True)
    _run(engine, _FullPipe(_ClassVae()), [])

    out = capsys.readouterr().out
    assert "prompt and reference encoding started on" in out
    assert "prompt and reference encoding finished in" in out
    assert "VAE encode started on" in out
    assert "VAE encode finished in" in out
    assert "VAE decode started on" in out
    assert "VAE decode finished in" in out
    # Two conditioning calls are reported as such rather than looking like two
    # separate stages.
    assert "(2 calls)" in out


def test_phase_status_reports_the_live_stage():
    engine = Engine(demo=True)
    seen = {}

    class _ProbeVae(_ClassVae):
        def decode(self, latents, *args, **kwargs):
            seen.update(engine.phase_status() or {})
            return "decoded"

    _run(engine, _FakePipe(_ProbeVae()), [])

    assert seen["name"] == "decode"
    assert seen["slow"] is False
    assert seen["elapsed_ms"] >= 0


def test_phase_status_is_cleared_when_the_run_ends():
    engine = Engine(demo=True)
    engine.generate(
        {
            "mode": "generate",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
            "prompt": "phase probe",
            "steps": 2,
        },
        callback=lambda _e: None,
    )
    assert engine.phase_status() is None


def test_a_stop_at_a_stage_boundary_ends_the_run_as_cancelled():
    """Nothing checks the flag inside a minutes-long encode, so the boundary
    check is what makes "stop" land in seconds."""
    engine = Engine(demo=True)

    class _CancellingVae(_ClassVae):
        def decode(self, latents, *args, **kwargs):
            engine.cancel()
            return "decoded"

    with pytest.raises(EngineError) as raised:
        _run(engine, _FakePipe(_CancellingVae()), [])
    assert raised.value.code == "CANCELLED"


def test_a_stop_pressed_before_the_run_starts_never_enters_the_pipeline():
    engine = Engine(demo=True)
    engine.cancel()
    pipe = _FullPipe(_ClassVae())

    with pytest.raises(EngineError) as raised:
        _run(engine, pipe, [])

    assert raised.value.code == "CANCELLED"
    assert pipe.encode_calls == 0


def test_a_stop_inside_the_last_stage_is_not_reported_as_a_finished_run():
    """The sampler honours the flag between steps, but a run stopped during the
    decode would otherwise be saved as a success nobody asked for."""
    engine = Engine(demo=True)

    class _LateVae:
        def __init__(self):
            self.calls = 0

        def decode(self, latents, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                engine.cancel()
            return "decoded"

    with pytest.raises(EngineError) as raised:
        _run(engine, _FakePipe(_LateVae()), [])
    assert raised.value.code == "CANCELLED"


def _seed_job(history: History, status: str) -> str:
    record = history.create(
        {
            "id": new_id(),
            "mode": "generate",
            "model_key": "qwen-image-2.1",
            "hub": "huggingface",
            "prompt": "stale row probe",
            "status": status,
        }
    )
    return record["id"]


def test_reconcile_running_flips_only_running_rows(tmp_path):
    history = History(AppPaths(tmp_path))
    stale = _seed_job(history, "running")
    done = _seed_job(history, "succeeded")

    flipped = history.reconcile_running()

    assert flipped == 1
    assert history.get(stale)["status"] == "failed"
    assert history.get(stale)["error"]  # readable reason, not a silent flip
    assert history.get(done)["status"] == "succeeded"
    # A second pass has nothing left to do.
    assert history.reconcile_running() == 0


def test_startup_reconciles_leftover_running_rows(tmp_path, monkeypatch):
    """Closing the console mid-decode must not leave a job running forever."""
    paths = AppPaths(tmp_path)
    history = History(paths)
    stale = _seed_job(history, "running")

    monkeypatch.setenv("IMGEN_DEMO", "1")
    app = create_app(demo=True, home=tmp_path)
    with TestClient(app) as client:
        rows = client.get("/api/jobs").json()["items"]

    row = next(item for item in rows if item["id"] == stale)
    assert row["status"] == "failed"
    assert row["error"]


def test_decode_time_is_printed_in_the_console(capsys):
    """The console has to show where a run's time went — that is how a
    minutes-long decode stops looking like a hang."""
    engine = Engine(demo=True)
    events = []
    _run(engine, _FakePipe(_ClassVae()), events)

    out = capsys.readouterr().out
    assert "VAE decode started on" in out
    assert "VAE decode finished in" in out


def test_failed_decode_is_printed_in_the_console(capsys):
    class _BoomVae:
        def decode(self, latents, *args, **kwargs):
            raise RuntimeError("decode exploded")

    engine = Engine(demo=True)
    events = []
    with pytest.raises(RuntimeError):
        _run(engine, _FakePipe(_BoomVae()), events)

    assert "VAE decode failed after" in capsys.readouterr().out


class _RecordingVae(torch.nn.Module):
    """Stand-in for the pipeline VAE: records device moves and can fail the way
    a real GPU decode does when VRAM runs out."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.moves = []
        self.gpu_error = None

    def to(self, *args, **kwargs):
        if args:
            self.moves.append(torch.device(args[0]).type)
        return super().to(*args, **kwargs)

    def decode(self, latents, *args, **kwargs):
        if self.gpu_error and self.weight.device.type == "cuda":
            raise RuntimeError(self.gpu_error)
        return latents


class _VaeHost:
    """Minimal pipe: what attach_split_vae touches and nothing more."""

    def __init__(self, vae, encode_error=None):
        self.vae = vae
        self.encode_error = encode_error
        self.encode_calls = 0

    def _encode_vae_image(self, image, generator):
        self.encode_calls += 1
        if self.encode_error and image.device.type == "cuda" and self.encode_calls == 1:
            raise RuntimeError(self.encode_error)
        return image


def test_split_vae_reports_the_devices_and_keeps_a_cpu_only_run_on_the_cpu():
    spec = {"offload": "group"}
    vae = _RecordingVae()
    pipe = _VaeHost(vae)

    attach_split_vae(pipe, spec, torch.device("cpu"))

    assert spec["vae_decode"] == "cpu" and spec["vae_encode"] == "cpu"
    assert vae.moves[0] == "cpu"  # the resident copy never leaves the CPU
    decoded = pipe.vae.decode(torch.zeros(1, 4, 4, 4))
    assert decoded.device.type == "cpu"
    encoded = pipe._encode_vae_image(torch.zeros(1, 3, 8, 8), None)
    assert encoded.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_split_vae_lends_the_gpu_to_both_directions_and_takes_it_back():
    spec = {"offload": "group"}
    vae = _RecordingVae()
    pipe = _VaeHost(vae)

    attach_split_vae(pipe, spec, torch.device("cuda", 0))
    assert spec["vae_decode"] == "cuda" and spec["vae_encode"] == "cuda"

    decoded = pipe.vae.decode(torch.zeros(1, 4, 4, 4))
    assert decoded.device.type == "cuda"
    # Handed back for the sampler — that is the whole point.
    assert vae.weight.device.type == "cpu"
    assert vae.moves[-1] == "cpu"

    encoded = pipe._encode_vae_image(torch.zeros(1, 3, 8, 8, device="cuda"), None)
    assert encoded.device.type == "cuda"
    assert vae.weight.device.type == "cpu"
    assert vae.moves[-1] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_encode_out_of_memory_latches_back_to_the_cpu():
    """An edit run must still finish when the references do not fit: the encode
    re-runs on the CPU and the runtime stops claiming the GPU."""
    spec = {"offload": "group"}
    vae = _RecordingVae()
    pipe = _VaeHost(vae, encode_error="CUDA out of memory. Tried to allocate 3.00 GiB")
    attach_split_vae(pipe, spec, torch.device("cuda", 0))

    encoded = pipe._encode_vae_image(torch.zeros(1, 3, 8, 8, device="cuda"), None)

    assert pipe.encode_calls == 2  # the CPU retry really ran
    assert encoded.device.type == "cuda"  # returned on the caller's device
    assert spec["vae_encode"] == "cpu"
    assert vae.weight.device.type == "cpu"
    moves_after_fallback = len(vae.moves)
    pipe._encode_vae_image(torch.zeros(1, 3, 8, 8, device="cuda"), None)
    assert len(vae.moves) == moves_after_fallback  # latched: no more GPU trips


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_decode_out_of_memory_latches_back_to_the_cpu():
    spec = {"offload": "group"}
    vae = _RecordingVae()
    vae.gpu_error = "CUDA out of memory. Tried to allocate 2.00 GiB"
    pipe = _VaeHost(vae)
    attach_split_vae(pipe, spec, torch.device("cuda", 0))

    decoded = pipe.vae.decode(torch.zeros(1, 4, 4, 4))

    assert decoded.device.type == "cpu"
    assert spec["vae_decode"] == "cpu"  # reported, so the studio says CPU again
    assert vae.weight.device.type == "cpu"
    moves_after_fallback = len(vae.moves)
    pipe.vae.decode(torch.zeros(1, 4, 4, 4))
    assert len(vae.moves) == moves_after_fallback  # latched: no more GPU trips


def test_a_running_job_reports_the_live_stage(tmp_path, monkeypatch):
    """The poll carries the stage, so "still encoding, be patient" and "nothing
    is happening" stop looking the same from the browser."""
    monkeypatch.setenv("IMGEN_DEMO", "1")
    app = create_app(demo=True, home=tmp_path)
    engine = app.state.engine
    stale = _seed_job(app.state.history, "running")
    engine._mark_phase("encode", True)

    with TestClient(app) as client:
        live = client.get(f"/api/jobs/{stale}").json()
        listed = client.get("/api/jobs").json()["items"]

    assert live["live_phase"]["name"] == "encode"
    assert live["live_phase"]["slow"] is True
    assert isinstance(live["live_phase"]["elapsed_ms"], int)
    assert listed[0]["live_phase"]["name"] == "encode"


def test_a_finished_job_carries_no_live_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGEN_DEMO", "1")
    app = create_app(demo=True, home=tmp_path)
    done = _seed_job(app.state.history, "succeeded")
    app.state.engine._mark_phase("decode", False)

    with TestClient(app) as client:
        got = client.get(f"/api/jobs/{done}").json()

    assert "live_phase" not in got


def test_a_cancelled_run_is_recorded_as_cancelled_not_failed(tmp_path, monkeypatch):
    """A stop request that lands at a stage boundary must read as cancelled —
    a red "failed" toast is the wrong story."""
    monkeypatch.setenv("IMGEN_DEMO", "1")
    app = create_app(demo=True, home=tmp_path)

    def cancelled(request, callback=None):
        raise EngineError("Cancelled.", "CANCELLED")

    monkeypatch.setattr(app.state.engine, "generate", cancelled)

    with TestClient(app) as client:
        job = client.post("/api/jobs", data={"mode": "generate", "prompt": "cancel probe"}).json()
        row = {"status": "running", "error": None}
        for _ in range(60):
            row = client.get(f"/api/jobs/{job['id']}").json()
            if row["status"] != "running":
                break
            time.sleep(0.05)

    assert row["status"] == "cancelled"
    assert row["error"] and "Cancelled" in row["error"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_decode_other_errors_still_raise():
    spec = {"offload": "group"}
    vae = _RecordingVae()
    vae.gpu_error = "device-side assert triggered"
    pipe = _VaeHost(vae)
    attach_split_vae(pipe, spec, torch.device("cuda", 0))

    with pytest.raises(RuntimeError, match="device-side assert"):
        pipe.vae.decode(torch.zeros(1, 4, 4, 4))
    # Nothing was masked as a memory fallback.
    assert spec["vae_decode"] == "cuda"
