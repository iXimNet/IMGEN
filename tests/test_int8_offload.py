"""Image21-INT8 placement: resident when the weights fit, streamed when they do not.

INT8 used to force ``enable_model_cpu_offload()`` on every machine, so a 32 GB
card copied every module onto the accelerator for its turn and back again while
the weights would have fit outright. The decision then went the other way and
lost track of two things: it measured the weights by their size on disk (18.6 GB)
rather than what they occupy once loaded (29.83 GiB, measured), and it compared
that against the device *total*, so memory another process was already holding
counted as free. A 32 GB card read "31.82 >= 25.6", kept ~30 GB resident, and
died in the VAE decode at 2K.

These tests pin the corrected rule: the line is the resident weights plus room
for the run, and it is drawn against what is actually free right now.
"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from imgen import int8_runtime as runtime
from imgen.engine import INT8_RESIDENT_HEADROOM_GB, Engine, int8_headroom_gb


def _component():
    return SimpleNamespace(is_loaded_in_8bit=True, to=Mock(), modules=lambda: [])


@pytest.fixture
def stack(monkeypatch):
    """Exercise the loader wiring without bitsandbytes, diffusers or weights."""
    transformer = _component()
    text_encoder = _component()
    pipe = SimpleNamespace(
        transformer=transformer,
        text_encoder=text_encoder,
        to=Mock(),
        enable_model_cpu_offload=Mock(),
    )
    loader = Mock(return_value=pipe)
    torch = SimpleNamespace(
        bfloat16="bfloat16",
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=Mock()),
    )
    # The device-move patch is the subject of its own module; wiring is not.
    monkeypatch.setattr(runtime, "patch_int8_device_moves", Mock())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            QwenImage21Pipeline=SimpleNamespace(from_pretrained=loader),
            QwenImage21Transformer2DModel=SimpleNamespace(
                from_pretrained=Mock(return_value=transformer)
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Qwen3VLForConditionalGeneration=SimpleNamespace(
                from_pretrained=Mock(return_value=text_encoder)
            )
        ),
    )
    return SimpleNamespace(pipe=pipe, loader=loader)


def _cuda_engine(monkeypatch, *, vram_gb=32, free_gb=1.5):
    """An engine on a fake CUDA card with a pinned free-memory reading.

    The reading is pinned rather than measured because the real one depends on
    the machine running the tests: on a card that already holds a model it is a
    few hundred MB, which would make every case below answer "stream them".
    `free_gb=None` stands for a reading that failed.
    """
    monkeypatch.setattr(
        "imgen.engine.probe",
        lambda: {"device": "cuda", "cuda": True, "vram_gb": vram_gb},
    )
    engine = Engine()
    monkeypatch.setattr(engine, "_available_gb", lambda: free_gb)
    return engine


def test_offload_streams_each_module(stack):
    runtime.load_int8_pipeline("/snapshot", device="cuda", offload=True, local_files_only=True)

    stack.pipe.enable_model_cpu_offload.assert_called_once_with()
    stack.pipe.to.assert_not_called()


def test_resident_keeps_the_weights_on_the_card(stack):
    runtime.load_int8_pipeline("/snapshot", device="cuda", offload=False, local_files_only=True)

    stack.pipe.to.assert_called_once_with("cuda")
    stack.pipe.enable_model_cpu_offload.assert_not_called()


def test_loading_stays_local(stack):
    """Either placement loads with local_files_only and no network fallback."""
    runtime.load_int8_pipeline("/snapshot", device="cuda", offload=False, local_files_only=True)

    kwargs = stack.loader.call_args.kwargs
    assert kwargs["local_files_only"] is True
    assert kwargs["dtype"] == "bfloat16"


def test_int8_headroom_shrinks_with_the_output_size():
    """The decode allocates the largest single block of the run and it grows
    with the picture, so a 1K run asks for about a quarter of the 2K figure.

    Demanding the 2K number at 1K would stream the weights off a card that can
    hold them.
    """
    assert int8_headroom_gb(2048) == INT8_RESIDENT_HEADROOM_GB
    assert int8_headroom_gb(4096) == INT8_RESIDENT_HEADROOM_GB  # capped
    assert int8_headroom_gb(1024) == pytest.approx(INT8_RESIDENT_HEADROOM_GB / 4)
    # No size known — a bare model load — assumes the worst case.
    assert int8_headroom_gb(None) == INT8_RESIDENT_HEADROOM_GB
    assert int8_headroom_gb(0) == INT8_RESIDENT_HEADROOM_GB


@pytest.mark.parametrize(
    "free_gb,resolution,expected",
    [
        # Floor = 16.7 GiB of weights + headroom. Headroom is the measured peak:
        # 27.0 GiB at 2K, a quarter of that at 1K.
        (1.5, 2048, True),  # a card that already holds something
        (30.26, 2048, True),  # measured free on this machine, card idle
        (43.6, 2048, True),
        (43.8, 2048, False),
        (23.4, 1024, True),  # 16.7 + 6.75
        (23.5, 1024, False),
        # The point of the split: 1K fits resident on this card, 2K does not.
        (30.26, 1024, False),
        (5.0, None, True),  # no size: worst case
    ],
)
def test_int8_offload_is_drawn_against_what_is_free(
    monkeypatch, free_gb, resolution, expected
):
    """Below the line the components stream, above it they stay put.

    What made a 32 GB card OOM at 2K was never the weights — 16.7 GiB fits
    easily. It is the 27 GiB of activations a 2K run needs beside them, against
    a card that only has 31.8 GiB in total.
    """
    engine = _cuda_engine(monkeypatch, free_gb=free_gb)
    assert engine._should_offload_int8(resolution) is expected


def test_int8_offload_falls_back_to_the_card_total_when_the_reading_fails(monkeypatch):
    """No reading, no better number: fall back to the device total, which is
    what this rule always used. Optimistic, but it is the only one left."""
    assert _cuda_engine(monkeypatch, vram_gb=48, free_gb=None)._should_offload_int8(2048) is False
    assert _cuda_engine(monkeypatch, vram_gb=24, free_gb=None)._should_offload_int8(2048) is True
    # And a missing total at all reads as zero, which streams.
    assert _cuda_engine(monkeypatch, vram_gb=None, free_gb=None)._should_offload_int8(2048) is True


def test_int8_offload_is_a_cuda_question(monkeypatch):
    monkeypatch.setattr("imgen.engine.probe", lambda: {"device": "cpu", "cuda": False})
    assert Engine()._should_offload_int8() is False


def test_engine_reports_the_int8_placement_it_chose(monkeypatch, tmp_path):
    engine = _cuda_engine(monkeypatch, free_gb=50.0)
    monkeypatch.setattr("imgen.engine.model_local_path", lambda *args: tmp_path)
    seen = {}
    monkeypatch.setattr(
        "imgen.int8_runtime.load_int8_pipeline",
        lambda *args, **kwargs: seen.update(kwargs) or SimpleNamespace(),
    )

    events = []
    loaded = engine.load("image21-int8", "huggingface", events.append)

    assert seen["offload"] is False  # 50 GB free holds 16.7 GB plus a 2K run
    assert seen["device"] == "cuda"
    assert loaded["cpu_offload"] is False
    assert loaded["loader"] == "int8"
    assert not any(e.get("stage") == "cpu_offload" for e in events)
    assert events[-1]["type"] == "load_complete"


def test_engine_asks_for_offload_on_a_small_card(monkeypatch, tmp_path):
    engine = _cuda_engine(monkeypatch, vram_gb=16, free_gb=None)
    monkeypatch.setattr("imgen.engine.model_local_path", lambda *args: tmp_path)
    seen = {}
    monkeypatch.setattr(
        "imgen.int8_runtime.load_int8_pipeline",
        lambda *args, **kwargs: seen.update(kwargs) or SimpleNamespace(),
    )

    events = []
    loaded = engine.load("image21-int8", "modelscope", events.append)

    assert seen["offload"] is True
    assert loaded["cpu_offload"] is True
    assert any(e.get("stage") == "cpu_offload" for e in events)


def test_streaming_says_why(monkeypatch, tmp_path):
    """The studio gets slower at the larger sizes when the weights stream, and
    nothing else about the run looks different. The reason has to travel."""
    engine = _cuda_engine(monkeypatch, free_gb=1.5)
    monkeypatch.setattr("imgen.engine.model_local_path", lambda *args: tmp_path)
    monkeypatch.setattr(
        "imgen.int8_runtime.load_int8_pipeline", lambda *args, **kwargs: SimpleNamespace()
    )

    events = []
    engine.load("image21-int8", "huggingface", events.append, output_resolution=2048)

    notices = [e for e in events if e.get("type") == "notice"]
    assert notices, "expected a notice explaining the placement"
    assert notices[0]["code"] == "int8_streamed"
    assert notices[0]["free_gb"] == 1.5


def test_a_larger_run_reloads_a_pipeline_that_no_longer_fits(monkeypatch, tmp_path):
    """Placement is part of the answer, so it is re-asked per run.

    A pipeline kept resident for a 1K run is the wrong shape for a 2K one, and
    reusing it as-is is exactly how the decode ran out of room.
    """
    engine = _cuda_engine(monkeypatch, free_gb=32.0)
    monkeypatch.setattr("imgen.engine.model_local_path", lambda *args: tmp_path)
    calls: list[dict] = []
    monkeypatch.setattr(
        "imgen.int8_runtime.load_int8_pipeline",
        lambda *args, **kwargs: calls.append(kwargs) or SimpleNamespace(),
    )

    # 32 GB free covers 29.8 + 1.75, so the weights stay put for a 1K run.
    engine.load("image21-int8", "huggingface", output_resolution=1024)
    assert calls[-1]["offload"] is False

    # The same pipeline does not fit once the run is 2K: reload, streamed.
    engine.load("image21-int8", "huggingface", output_resolution=2048)
    assert calls[-1]["offload"] is True
    assert engine.loaded["cpu_offload"] is True

    # Asking for the same shape again is a no-op, not another reload.
    engine.load("image21-int8", "huggingface", output_resolution=2048)
    assert len(calls) == 2


def test_the_same_run_does_not_reload_an_already_correct_pipeline(monkeypatch, tmp_path):
    engine = _cuda_engine(monkeypatch, free_gb=1.5)
    monkeypatch.setattr("imgen.engine.model_local_path", lambda *args: tmp_path)
    calls: list[dict] = []
    monkeypatch.setattr(
        "imgen.int8_runtime.load_int8_pipeline",
        lambda *args, **kwargs: calls.append(kwargs) or SimpleNamespace(),
    )

    engine.load("image21-int8", "huggingface", output_resolution=2048)
    engine.load("image21-int8", "huggingface", output_resolution=2048)

    assert len(calls) == 1
