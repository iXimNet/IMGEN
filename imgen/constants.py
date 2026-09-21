"""Catalog of supported models, hubs, and official sampling defaults."""

from __future__ import annotations

APP_NAME = "IMGEN"
APP_TAGLINE_ZH = "Qwen-Image-2.1 本地生图 / 改图工作室"
APP_TAGLINE_EN = "Local studio for Qwen-Image-2.1 generation and editing"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100
MAX_REFERENCE_IMAGES = 10
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Official Qwen-Image-2.1 sampling defaults:
# https://github.com/QwenLM/Qwen-Image-2.1
# https://huggingface.co/Qwen/Qwen-Image-2.1
# Diffusers QwenImage21Pipeline: 40 steps, true_cfg_scale=1.0, use_kv_cache=True
DEFAULT_STEPS = 40
DEFAULT_CFG = 1.0
DEFAULT_KV_CACHE = True
DEFAULT_OUTPUT_RESOLUTION_2K = 2048
DEFAULT_OUTPUT_RESOLUTION_1K = 1024
DEFAULT_NUM_IMAGES = 1

# Native 2K sizes published by Qwen.
ASPECT_RATIOS_2K: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

def _snap32(value: int) -> int:
    return max(32, int(round(value / 32) * 32))


# Half of the official 2K table, snapped to multiples of 32 (VAE constraint).
ASPECT_RATIOS_1K: dict[str, tuple[int, int]] = {
    key: (_snap32(w // 2), _snap32(h // 2)) for key, (w, h) in ASPECT_RATIOS_2K.items()
}

RGBA_PREFIX = "This is an RGBA image with transparency."
RGBA_SUFFIX = "The image has alpha channel and the background is transparent."

NEGATIVE_PROMPT_PLACEHOLDER = (
    "blurry, low quality, distorted, deformed, watermark, extra limbs, extra text"
)

MODELS: dict[str, dict] = {
    "qwen-image-2.1": {
        "key": "qwen-image-2.1",
        "label": "Qwen-Image-2.1",
        "precision": "BF16",
        "loader": "bf16",
        "approx_gb": 33.1,
        "repos": {
            "huggingface": "Qwen/Qwen-Image-2.1",
            "modelscope": "Qwen/Qwen-Image-2.1",
        },
        "urls": {
            "huggingface": "https://huggingface.co/Qwen/Qwen-Image-2.1",
            "modelscope": "https://modelscope.cn/models/Qwen/Qwen-Image-2.1",
            "github": "https://github.com/QwenLM/Qwen-Image-2.1",
            "blog": "https://qwen.ai/blog?id=qwen-image-2.1",
        },
        "notes_zh": "官方 BF16 权重。原生 2K，统一生图与改图，最多 10 张参考图，支持 RGBA。",
        "notes_en": "Official BF16 weights. Native 2K, unified generate/edit, up to 10 refs, RGBA.",
        "requires_cuda": False,
        "int8": False,
    },
    "image21-int8": {
        "key": "image21-int8",
        "label": "Image21-INT8",
        "precision": "INT8",
        "loader": "int8",
        "approx_gb": 18.6,
        "repos": {
            "huggingface": "ixim/Image21-INT8",
            "modelscope": "iximbox/Image21-INT8",
        },
        "urls": {
            "huggingface": "https://huggingface.co/ixim/Image21-INT8",
            "modelscope": "https://modelscope.cn/models/iximbox/Image21-INT8",
        },
        "notes_zh": "ixim 社区 bitsandbytes INT8 转换。仅 NVIDIA CUDA。改图请使用 2048 并开启 VAE Tiling。",
        "notes_en": "Community bitsandbytes INT8 conversion by ixim. NVIDIA CUDA only. Edit at 2048 with VAE tiling.",
        "requires_cuda": True,
        "int8": True,
    },
}

HUBS = {
    "huggingface": {
        "key": "huggingface",
        "label": "Hugging Face",
        "hint_zh": "全球节点，适合大多数地区。可填写 HF Token。",
        "hint_en": "Global CDN. Optional Hugging Face access token.",
        "token_env": "HF_TOKEN",
        "url": "https://huggingface.co",
    },
    "modelscope": {
        "key": "modelscope",
        "label": "ModelScope",
        "hint_zh": "魔搭社区，中国大陆下载通常更快。可填写 ModelScope Token。",
        "hint_en": "Usually faster in mainland China. Optional ModelScope token.",
        "token_env": "MODELSCOPE_API_TOKEN",
        "url": "https://modelscope.cn",
    },
}

RESOLUTION_SCALES = {
    "2k": {
        "key": "2k",
        "label_zh": "原生 2K（官方推荐）",
        "label_en": "Native 2K (official)",
        "output_resolution": DEFAULT_OUTPUT_RESOLUTION_2K,
        "table": ASPECT_RATIOS_2K,
    },
    "1k": {
        "key": "1k",
        "label_zh": "1K（更省显存）",
        "label_en": "1K (lower VRAM)",
        "output_resolution": DEFAULT_OUTPUT_RESOLUTION_1K,
        "table": ASPECT_RATIOS_1K,
    },
}


def repo_id(model_key: str, hub: str) -> str:
    model = MODELS[model_key]
    try:
        return model["repos"][hub]
    except KeyError as exc:
        raise ValueError(f"Unknown model/hub combination: {model_key}/{hub}") from exc
