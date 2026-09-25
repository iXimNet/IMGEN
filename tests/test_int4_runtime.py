import os
import sys
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from imgen import int4_runtime as runtime
from imgen.engine import Engine, EngineError


class Device:
    def __init__(self, value):
        parts = str(value).split(":")
        self.type = parts[0]
        self.index = int(parts[1]) if len(parts) > 1 else None

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


class Tensor:
    def __init__(self, device, dtype="bfloat16"):
        self.device = device
        self.dtype = dtype

    def detach(self):
        return self

    def to(self, device, dtype=None):
        return Tensor(str(device), dtype or self.dtype)


def component(quantized=True):
    quant = {"quant_method": "sdnq", "weights_dtype": "uint4", "use_quantized_matmul": False}
    return SimpleNamespace(config=SimpleNamespace(quantization_config=quant if quantized else None))


@pytest.fixture
def stack(monkeypatch):
    """Exercise loader wiring without requiring GPU packages or model weights."""
    torch = SimpleNamespace(
        device=Device,
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
        is_tensor=lambda value: isinstance(value, Tensor),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda device: SimpleNamespace(total_memory=8 * 1024**3),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    pipe = SimpleNamespace(
        transformer=component(),
        text_encoder=component(),
        vae=component(False),
        to=Mock(),
        enable_model_cpu_offload=Mock(),
    )
    pipe.transformer.enable_group_offload = Mock()
    pipe.vae.to = Mock()
    pipe.vae.decode = Mock(side_effect=lambda latents, **kw: latents)
    pipe._encode_vae_image = Mock(side_effect=lambda image, generator: image)
    group = Mock()

    def load(*args, **kwargs):
        assert os.environ["DIFFUSERS_SDNQ_TRANSFORMERS"] == "1"
        return pipe

    loader = Mock(side_effect=load)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sdnq", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        QwenImage21Pipeline=SimpleNamespace(from_pretrained=loader),
    ))
    monkeypatch.setitem(sys.modules, "diffusers.hooks.group_offloading", SimpleNamespace(
        apply_group_offloading=group,
    ))
    monkeypatch.delenv("IMAGE21_DTYPE", raising=False)
    monkeypatch.delenv("DIFFUSERS_SDNQ_TRANSFORMERS", raising=False)
    return SimpleNamespace(torch=torch, pipe=pipe, loader=loader, group=group)


@pytest.mark.parametrize("device,memory,mode", [
    ("cuda", 8, "group"), ("cuda", 10, "group"), ("cuda", 12, "model"),
    ("cuda", 32, "model"), ("mps", 24, "resident"), ("cpu", 64, "resident"),
])
def test_runtime_placement(device, memory, mode):
    assert runtime.choose_runtime(device, memory * 1024**3)["offload"] == mode


def test_small_cuda_offloads_both_vae_directions(stack):
    pipe = runtime.load_int4_pipeline("/snapshot", device="cuda:1", local_files_only=True)
    stack.loader.assert_called_once_with("/snapshot", dtype="bfloat16", local_files_only=True)
    block = pipe.transformer.enable_group_offload.call_args.kwargs
    assert block["offload_type"] == "block_level" and block["num_blocks_per_group"] == 1
    assert str(block["onload_device"]) == "cuda:1"
    assert block["use_stream"] is False and block["non_blocking"] is False
    assert stack.group.call_args.args == (pipe.text_encoder,)
    assert stack.group.call_args.kwargs["offload_type"] == "leaf_level"
    pipe.vae.to.assert_called_once_with("cpu")
    decoded = pipe.vae.decode(Tensor("cuda:1"), return_dict=False)
    assert decoded.device == "cpu"
    encoded = pipe._encode_vae_image(Tensor("cuda:1", "float16"), "cpu-generator")
    assert encoded.device == "cuda:1" and encoded.dtype == "float16"
    # The closure captured the original encode function; inspect its call via a separate test below.
    assert pipe.image21_runtime["offload"] == "group"
    pipe.to.assert_not_called()
    pipe.enable_model_cpu_offload.assert_not_called()


def test_group_encode_actually_runs_on_cpu(stack):
    original = stack.pipe._encode_vae_image
    pipe = runtime.load_int4_pipeline("/snapshot", device="cuda")
    pipe._encode_vae_image(Tensor("cuda"), "cpu-generator")
    args = original.call_args.args
    assert args[0].device == "cpu" and args[1] == "cpu-generator"


def test_large_cuda_uses_model_offload(stack):
    stack.torch.cuda.get_device_properties = lambda _: SimpleNamespace(total_memory=24 * 1024**3)
    pipe = runtime.load_int4_pipeline("/snapshot", device="cuda:1")
    pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=1)
    pipe.to.assert_not_called()
    stack.group.assert_not_called()


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_resident_devices_and_dtype_override(stack, monkeypatch, device):
    monkeypatch.setenv("IMAGE21_DTYPE", "float16")
    pipe = runtime.load_int4_pipeline("/snapshot", device=device, local_files_only=True)
    assert str(pipe.to.call_args.args[0]) == device
    assert stack.loader.call_args.kwargs["dtype"] == "float16"
    assert pipe.image21_runtime == {
        "offload": "resident", "dtype": "float16", "device": device, "use_stream": False,
    }
    pipe.enable_model_cpu_offload.assert_not_called()


def test_invalid_dtype_fails_before_loading(stack, monkeypatch):
    monkeypatch.setenv("IMAGE21_DTYPE", "int8")
    with pytest.raises(ValueError, match="IMAGE21_DTYPE"):
        runtime.load_int4_pipeline("/snapshot", device="cpu")
    stack.loader.assert_not_called()


@pytest.mark.parametrize("target,change", [
    ("transformer", {"quant_method": "bitsandbytes"}),
    ("text_encoder", {"weights_dtype": "nf4"}),
    ("transformer", {"use_quantized_matmul": True}),
])
def test_wrong_quantization_is_rejected(stack, target, change):
    getattr(stack.pipe, target).config.quantization_config.update(change)
    with pytest.raises(ValueError, match=target):
        runtime.load_int4_pipeline("/snapshot", device="cpu")
    stack.pipe.to.assert_not_called()


def test_vae_must_remain_float(stack):
    stack.pipe.vae.config.quantization_config = {"quant_method": "sdnq"}
    with pytest.raises(ValueError, match="VAE"):
        runtime.load_int4_pipeline("/snapshot", device="cpu")


def test_quantization_config_accepts_library_enums():
    class Method(Enum):
        SDNQ = "sdnq"

    module = component()
    module.config.quantization_config = SimpleNamespace(to_dict=lambda: {
        "quant_method": Method.SDNQ, "weights_dtype": "uint4",
    })
    runtime._require_int4(module, "transformer")


def test_missing_sdnq_has_install_instructions(stack, monkeypatch):
    monkeypatch.setitem(sys.modules, "sdnq", None)
    with pytest.raises(RuntimeError, match="requirements-int4.txt"):
        runtime.load_int4_pipeline("/snapshot", device="cpu")


@pytest.mark.parametrize("device", ["mps", "cpu", "cuda"])
def test_engine_routes_int4_and_reports_runtime(stack, monkeypatch, tmp_path, device):
    monkeypatch.setattr("imgen.engine.probe", lambda: {"device": device, "cuda": device == "cuda"})
    monkeypatch.setattr("imgen.engine.model_local_path", lambda *args: tmp_path)
    engine = Engine()
    events = []
    loaded = engine.load("image21-int4", "modelscope", events.append)
    assert loaded["loader"] == "int4"
    assert loaded["cpu_offload"] is (device == "cuda")
    assert loaded["runtime"]["device"] == device
    assert stack.loader.call_args.kwargs["local_files_only"] is True
    assert engine.load("image21-int4", "modelscope") is loaded
    assert stack.loader.call_count == 1
    assert events[-1]["type"] == "load_complete"
    if device != "cuda":
        with pytest.raises(EngineError, match="requires an NVIDIA CUDA"):
            engine.load("image21-int8", "modelscope")
