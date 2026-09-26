"""VAE decode phase events + startup reconciliation of stale running jobs.

The decode step of a pipeline run used to be a silent stretch: the studio
froze on the last sampler update ("100%, ~0s left") while small-GPU INT4 runs
decoded on the CPU for minutes. These tests pin the pieces that fix that: the
engine announces the decode phase exactly once (flagging the CPU-decode case),
the wrapper never damages the VAE it borrows, the demo path keeps the same
event shape, and a restart never leaves phantom "running" rows behind.
"""

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from imgen.app import create_app  # noqa: E402
from imgen.engine import Engine  # noqa: E402
from imgen.history import History, new_id  # noqa: E402
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
