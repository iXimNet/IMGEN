"""Device-only offload workaround for the pinned bitsandbytes 0.50.2 stack.

Adapted from ixim/Image21-INT8 `scripts/runtime.py`
https://huggingface.co/ixim/Image21-INT8

Parent Module.to() recursively invokes _apply(), not each child's to().
Linear8bitLt.to() moves auxiliary tensors, but this path is bypassed by a
parent move. Keep CB aliases and SCB/outlier state colocated with the weight.
This is instance-local; checkpoint tensors and installed packages are unchanged.
"""

from types import MethodType

import torch


def patch_int8_device_moves(model):
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
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration

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
