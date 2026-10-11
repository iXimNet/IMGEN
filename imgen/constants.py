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
# The engine clamps a run to this many pictures; the panel's stepper shows the
# same bound, so the two cannot drift apart.
MAX_NUM_IMAGES = 4

# Image21-INT8 errata (2026-09-24): its earlier general 2048px recommendation was
# withdrawn in favour of a 1024px area scale, so 1K is what the studio opens on.
DEFAULT_SCALE = "1k"

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

# The sampling schedule Qwen-Image-2.1-Turbo ships in its own `model_index.json`
# under this key. The installed diffusers build does not read the key, so a run
# that only trusted the step field would fall back to a 40-step linear schedule
# and quietly miss the recipe the checkpoint was distilled for. Read from the
# checkpoint when it is there; this list is what the studio falls back to when a
# conversion drops it.
TURBO_SIGMAS_KEY = "sample_sigmas"
TURBO_SIGMAS = [
    1.0,
    0.978453,
    0.95418,
    0.926626,
    0.89508,
    0.845148,
    0.704534,
    0.414568,
]

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
    # The distilled sibling of the base weights. It is the same pipeline class
    # with the same call signature and a full checkpoint — not a LoRA — so it
    # reuses the bf16 loader untouched. The one real difference is the recipe:
    # eight steps and a schedule the checkpoint carries itself, which is what
    # `locked_params` and `sigmas_from` below declare.
    "qwen-image-2.1-turbo": {
        "key": "qwen-image-2.1-turbo",
        "label": "Qwen-Image-2.1-Turbo",
        "precision": "BF16",
        "loader": "bf16",
        # Measured from the hub's own file table: 30.21 GiB of weights, and the
        # catalogue quotes decimal GB (the base entry's 33.1 is its 30.84 GiB).
        "approx_gb": 32.4,
        "repos": {
            "huggingface": "Qwen/Qwen-Image-2.1-Turbo",
            "modelscope": "Qwen/Qwen-Image-2.1-Turbo",
        },
        "urls": {
            "huggingface": "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo",
            "modelscope": "https://modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo",
            "github": "https://github.com/QwenLM/Qwen-Image-2.1",
        },
        "notes_zh": (
            "BF16 8 步蒸馏加速版，采样配方随权重内置。CFG 固定为 1，负向提示词不参与。"
            "许可证为 Qwen Research，仅限非商用。"
        ),
        "notes_en": (
            "BF16 8-step distilled build. The sampling schedule ships with the weights. "
            "CFG is fixed at 1, so negative prompts do not apply. "
            "Qwen Research licence — non-commercial use only."
        ),
        # Values the run will use whatever the request asked for. They are not
        # preferences: a distilled checkpoint is only correct on its own recipe,
        # so the engine re-applies these after every normalisation. A model
        # without this key behaves exactly as it did before.
        "locked_params": {"steps": 8, "true_cfg_scale": 1.0},
        "sigmas_from": TURBO_SIGMAS_KEY,
        "sigmas_fallback": TURBO_SIGMAS,
        "requires_cuda": False,
        "int8": False,
    },
    # The INT8 conversion of the distilled checkpoint, so it carries both
    # halves: the eight-step recipe (declared exactly as the BF16 Turbo above,
    # and the converter did keep `sample_sigmas` in the release) and the
    # quantised loader. Everything the studio does with the recipe is keyed off
    # those two declarations, not off the model key, so nothing else has to
    # learn about a fifth entry.
    #
    # The release asks for a pinned diffusers commit that reads
    # `sample_sigmas` on its own. This project does not need it and does not
    # upgrade for it: the schedule is passed in explicitly, which arrives at the
    # same eight values on the build already installed.
    "image21-turbo-int8": {
        "key": "image21-turbo-int8",
        "label": "Image21-Turbo-INT8",
        "precision": "INT8",
        "loader": "int8",
        # 17.25 GiB of weights, and the release's own table quotes decimal GB
        # (18.527), the same unit the other two entries already use.
        "approx_gb": 18.5,
        "repos": {
            "huggingface": "ixim/Image21-Turbo-INT8",
            "modelscope": "iximbox/Image21-Turbo-INT8",
        },
        "urls": {
            "huggingface": "https://huggingface.co/ixim/Image21-Turbo-INT8",
            "modelscope": "https://modelscope.cn/models/iximbox/Image21-Turbo-INT8",
        },
        "notes_zh": (
            "ixim 社区 INT8 转换的 Turbo 版，8 步采样配方随权重内置。CFG 固定为 1，负向提示词不参与。"
            "仅 NVIDIA CUDA。许可证为 Qwen Research，仅限非商用。"
        ),
        "notes_en": (
            "Community INT8 conversion of Turbo by ixim. The eight-step recipe ships with the "
            "weights. CFG is fixed at 1, so negative prompts do not apply. NVIDIA CUDA only. "
            "Qwen Research licence — non-commercial use only."
        ),
        "locked_params": {"steps": 8, "true_cfg_scale": 1.0},
        "sigmas_from": TURBO_SIGMAS_KEY,
        "sigmas_fallback": TURBO_SIGMAS,
        "requires_cuda": True,
        "int8": True,
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
        "notes_zh": "ixim 社区 bitsandbytes INT8 转换。仅 NVIDIA CUDA。改图建议从 1024px 起，VAE Tiling 保持关闭。",
        "notes_en": "Community bitsandbytes INT8 conversion by ixim. NVIDIA CUDA only. Start edits at 1024px and leave VAE tiling off.",
        "requires_cuda": True,
        "int8": True,
    },
    "image21-int4": {
        "key": "image21-int4",
        "label": "Image21-INT4",
        "precision": "INT4",
        "loader": "int4",
        "approx_gb": 13.4,
        "repos": {
            "huggingface": "ixim/Image21-INT4",
            "modelscope": "iximbox/Image21-INT4",
        },
        "urls": {
            "huggingface": "https://huggingface.co/ixim/Image21-INT4",
            "modelscope": "https://modelscope.cn/models/iximbox/Image21-INT4",
        },
        "notes_zh": (
            "ixim 社区 SDNQ UINT4 转换。支持 CUDA、Apple Silicon MPS 和 CPU。"
            "小显存 CUDA 自动分组卸载；MPS 建议 24GB 统一内存。"
        ),
        "notes_en": (
            "Community SDNQ UINT4 conversion by ixim. CUDA, Apple Silicon MPS, or CPU. "
            "Small CUDA GPUs use group offload; 24GB unified memory suggested for MPS."
        ),
        "requires_cuda": False,
        "int8": False,
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

# Labels stay descriptive, never advisory: the Image21-INT8 card withdrew its
# earlier general 2048px recommendation, so the studio does not endorse a size.
RESOLUTION_SCALES = {
    "2k": {
        "key": "2k",
        "label_zh": "原生 2K",
        "label_en": "Native 2K",
        "note_zh": "",
        "note_en": "",
        "output_resolution": DEFAULT_OUTPUT_RESOLUTION_2K,
        "table": ASPECT_RATIOS_2K,
    },
    "1k": {
        "key": "1k",
        "label_zh": "1K",
        "label_en": "1K",
        "note_zh": "",
        "note_en": "",
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
