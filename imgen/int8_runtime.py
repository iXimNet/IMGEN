"""Device-only offload workaround for the pinned bitsandbytes 0.50.2 stack.

Adapted from ixim/Image21-INT8 `scripts/runtime.py`
https://huggingface.co/ixim/Image21-INT8

Parent Module.to() recursively invokes _apply(), not each child's to().
Linear8bitLt.to() moves auxiliary tensors, but this path is bypassed by a
parent move. Keep CB aliases and SCB/outlier state colocated with the weight.
This is instance-local; checkpoint tensors and installed packages are unchanged.
"""

import logging
import warnings
from types import MethodType

_SILENCED = False
_ORIG_WARN = warnings.warn
_BF16_CAST_FILTER = r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization"


def _is_matmul8bit_cast_warning(message) -> bool:
    text = message.args[0] if isinstance(message, BaseException) else str(message)
    return "MatMul8bitLt:" in text and "float16" in text


def _quiet_warn(*args, **kwargs):
    message = args[0] if args else kwargs.get("message", "")
    if _is_matmul8bit_cast_warning(message):
        return None
    return _ORIG_WARN(*args, **kwargs)


class _DropMatMul8bitCast(logging.Filter):
    """bitsandbytes 0.50+ logs the bf16→fp16 cast via logging, not warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "MatMul8bitLt:" not in record.getMessage()


def silence_int8_bf16_cast_warnings() -> None:
    """bitsandbytes INT8 kernels take fp16 activations.

    Image21-INT8 is loaded in bfloat16. Older bitsandbytes calls warnings.warn
    on every layer; 0.50+ uses logger.warning. Both paths are silenced here.
    """
    global _SILENCED
    warnings.filterwarnings("ignore", message=_BF16_CAST_FILTER)
    warnings.filterwarnings("ignore", message=r"MatMul8bitLt:")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"bitsandbytes(\..*)?",
    )
    warnings.warn = _quiet_warn
    drop = _DropMatMul8bitCast()
    for name in ("bitsandbytes", "bitsandbytes.autograd", "bitsandbytes.autograd._functions", "py.warnings"):
        logger = logging.getLogger(name)
        if not any(isinstance(item, _DropMatMul8bitCast) for item in logger.filters):
            logger.addFilter(drop)
    try:
        import bitsandbytes.autograd._functions as bnb_fn

        bnb_fn.warn = _quiet_warn
        bnb_fn.warnings.warn = _quiet_warn
        if not any(isinstance(item, _DropMatMul8bitCast) for item in bnb_fn.logger.filters):
            bnb_fn.logger.addFilter(drop)
    except Exception:
        pass
    try:
        import bitsandbytes.research.autograd._functions as bnb_research

        bnb_research.warnings.warn = _quiet_warn
    except Exception:
        pass
    _SILENCED = True


def patch_int8_device_moves(model):
    import torch
    from bitsandbytes.nn import Linear8bitLt

    for layer in model.modules():
        if not isinstance(layer, Linear8bitLt) or getattr(layer, "_int8_move_patched", False):
            continue
        if layer.state.has_fp16_weights:
            raise ValueError("Offload helper only supports inference INT8 weights")
        original = layer._apply

        def apply(self, fn, recurse=True, _original=original):
            result = _original(fn, recurse=recurse)
            device = self.weight.device
            # Parameter auxiliaries are used before init_8bit_state(); state
            # auxiliaries are used afterwards. Never duplicate the CB storage.
            if self.weight.CB is not None:
                self.weight.CB = self.weight.data
            if self.weight.SCB is not None:
                self.weight.SCB = self.weight.SCB.to(device)
            for name, value in vars(self.state).items():
                if isinstance(value, torch.Tensor):
                    setattr(
                        self.state,
                        name,
                        self.weight.data if name == "CB" else value.to(device),
                    )
            return result

        layer._apply = MethodType(apply, layer)
        layer._int8_move_patched = True


def enable_int8_cpu_offload(pipe):
    """Install auxiliary-tensor movement before standard model CPU offload."""
    for name in ("text_encoder", "transformer"):
        patch_int8_device_moves(getattr(pipe, name))
    pipe.enable_model_cpu_offload()


def load_int8_pipeline(model, local_files_only=False):
    """Load/offload components sequentially to avoid a combined CUDA load peak."""
    import torch
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration

    silence_int8_bf16_cast_warnings()

    components = {}
    for name, cls in [
        ("transformer", QwenImage21Transformer2DModel),
        ("text_encoder", Qwen3VLForConditionalGeneration),
    ]:
        component = cls.from_pretrained(
            model,
            subfolder=name,
            dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        if not getattr(component, "is_loaded_in_8bit", False):
            raise ValueError("Expected saved bitsandbytes INT8 component: " + name)
        patch_int8_device_moves(component)
        component.to("cpu")
        torch.cuda.empty_cache()
        components[name] = component
    pipe = QwenImage21Pipeline.from_pretrained(
        model,
        dtype=torch.bfloat16,
        local_files_only=local_files_only,
        **components,
    )
    enable_int8_cpu_offload(pipe)
    return pipe
