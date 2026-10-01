"""Image21-INT8 placement: resident when the weights fit, streamed when they do not.

INT8 used to force ``enable_model_cpu_offload()`` on every machine, so a 32 GB
card copied every module onto the accelerator for its turn and back again while
the ~18.6 GB of weights would have fit outright. These tests pin the size-based
choice, and the loader wiring that carries it through.
"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from imgen import int8_runtime as runtime
from imgen.engine import Engine


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


@pytest.mark.parametrize(
    "vram,expected",
    [(16, True), (24, True), (32, False), (48, False), (None, True)],
)
def test_int8_offload_follows_the_card(monkeypatch, vram, expected):
    """~18.6 GB of weights plus room for a 2K step is the line: below it the
    components stream, above it they stay put."""
    info = {"device": "cuda", "cuda": True}
    if vram is not None:
        info["vram_gb"] = vram
    monkeypatch.setattr("imgen.engine.probe", lambda: info)

    assert Engine()._should_offload_int8() is expected


def test_int8_offload_is_a_cuda_question(monkeypatch):
    monkeypatch.setattr("imgen.engine.probe", lambda: {"device": "cpu", "cuda": False})
    assert Engine()._should_offload_int8() is False


def test_engine_reports_the_int8_placement_it_chose(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "imgen.engine.probe", lambda: {"device": "cuda", "cuda": True, "vram_gb": 32}
    )
    monkeypatch.setattr("imgen.engine.model_local_path", lambda *args: tmp_path)
    seen = {}
    monkeypatch.setattr(
        "imgen.int8_runtime.load_int8_pipeline",
        lambda *args, **kwargs: seen.update(kwargs) or SimpleNamespace(),
    )

    events = []
    loaded = Engine().load("image21-int8", "huggingface", events.append)

    assert seen["offload"] is False  # a 32 GB card holds the weights
    assert seen["device"] == "cuda"
    assert loaded["cpu_offload"] is False
    assert loaded["loader"] == "int8"
    assert not any(e.get("stage") == "cpu_offload" for e in events)
    assert events[-1]["type"] == "load_complete"


def test_engine_asks_for_offload_on_a_small_card(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "imgen.engine.probe", lambda: {"device": "cuda", "cuda": True, "vram_gb": 16}
    )
    monkeypatch.setattr("imgen.engine.model_local_path", lambda *args: tmp_path)
    seen = {}
    monkeypatch.setattr(
        "imgen.int8_runtime.load_int8_pipeline",
        lambda *args, **kwargs: seen.update(kwargs) or SimpleNamespace(),
    )

    events = []
    loaded = Engine().load("image21-int8", "modelscope", events.append)

    assert seen["offload"] is True
    assert loaded["cpu_offload"] is True
    assert any(e.get("stage") == "cpu_offload" for e in events)
