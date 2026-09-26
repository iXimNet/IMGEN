"""VAE decode phase events + startup reconciliation of stale running jobs.

The decode step of a pipeline run used to be a silent stretch: the studio
froze on the last sampler update ("100%, ~0s left") while small-GPU INT4 runs
decoded on the CPU for minutes. These tests pin the pieces that fix that: the
engine announces the decode phase exactly once (flagging the CPU-decode case),
the wrapper never damages the VAE it borrows, the demo path keeps the same
event shape, and a restart never leaves phantom "running" rows behind.
"""

import pytest
import torch
from PIL import Image

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from imgen.app import create_app  # noqa: E402
from imgen.engine import Engine  # noqa: E402
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


def test_demo_engine_emits_decode_phase_after_last_sampler_step():
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
    assert "generate_phase" in kinds
    phase = events[kinds.index("generate_phase")]
    assert phase["phase"] == "decode"
    assert phase["slow"] is False
    # It comes after every sampler step, never before.
    assert kinds.index("generate_phase") > kinds.index("generate_progress")
    progress_steps = [e["step"] for e in events if e["type"] == "generate_progress"]
    assert progress_steps == [1, 2]


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

    def __init__(self, vae):
        self.vae = vae

    def _encode_vae_image(self, image, generator):
        return image


def test_split_vae_reports_the_decode_device_and_keeps_encode_on_cpu():
    spec = {"offload": "group"}
    vae = _RecordingVae()
    pipe = _VaeHost(vae)

    attach_split_vae(pipe, spec, torch.device("cpu"))

    assert spec["vae_decode"] == "cpu"
    assert vae.moves[0] == "cpu"  # the resident copy never leaves the CPU
    decoded = pipe.vae.decode(torch.zeros(1, 4, 4, 4))
    assert decoded.device.type == "cpu"
    encoded = pipe._encode_vae_image(torch.zeros(1, 3, 8, 8), None)
    assert encoded.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_split_vae_borrows_the_gpu_for_decode_and_gives_it_back():
    spec = {"offload": "group"}
    vae = _RecordingVae()
    pipe = _VaeHost(vae)

    attach_split_vae(pipe, spec, torch.device("cuda", 0))
    assert spec["vae_decode"] == "cuda"

    decoded = pipe.vae.decode(torch.zeros(1, 4, 4, 4))
    assert decoded.device.type == "cuda"
    # Handed back for the next sampling run — that is the whole point.
    assert vae.weight.device.type == "cpu"
    assert vae.moves[-1] == "cpu"
    # Encode still never touches the GPU: it runs while the transformer owns it.
    moves_before_encode = len(vae.moves)
    pipe._encode_vae_image(torch.zeros(1, 3, 8, 8), None)
    assert len(vae.moves) == moves_before_encode


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
