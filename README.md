# <img src="imgen/static/brand/imgen-wordmark.png" alt="IMGEN" width="360" height="88">

Local visual studio for **Qwen-Image-2.1** — generate from text, edit with up to **10 reference images**, and keep every run in a searchable history. Supports the official BF16 weights, the official **8-step Turbo** build, and three community quantisations.

Not Gradio. A FastAPI backend with a purpose-built interface. Runs on **Windows** and **macOS**.

[Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) · [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-2.1) · [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-2.1) · [Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) · [Blog](https://qwen.ai/blog?id=qwen-image-2.1)

> **Weights are not MIT.** Every model — Qwen-Image-2.1, Qwen-Image-2.1-Turbo, Image21-INT8, Image21-Turbo-INT8 and Image21-INT4 — is released under the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1), which is **non-commercial**. Read it before you download or publish results. This repository only ships the studio application (MIT).

---

## Features

- **Generate and edit in one pipeline.** Qwen-Image-2.1 is a unified model (`QwenImage21Pipeline`). Switching modes does not switch checkpoints.
- **Five weights, two hubs.** Choose **Qwen-Image-2.1** (official BF16), **Qwen-Image-2.1-Turbo** (official 8-step distilled BF16), **Image21-INT8** (bitsandbytes), **Image21-Turbo-INT8** (bitsandbytes), or **Image21-INT4** (SDNQ UINT4). The quantized models are community conversions by [ixim](https://huggingface.co/ixim). Download from **Hugging Face** or **ModelScope** on first run.
- **Sampling defaults.** 40 steps, `true_cfg_scale=1.0` (guidance off), prefix KV cache on, RGBA prompt template — plus a 1024px default area and VAE tiling off, following the Image21-INT8 errata. The 2K aspect table is one click away.
- **The Turbo builds pin their own recipe.** Both Turbo entries run **8 steps** on the schedule the checkpoint carries in its own `model_index.json`. The panel shows the number, disables the step and CFG controls and drops the step presets; the engine re-applies the values after every request, so neither a restored history record nor a direct API call can ask a distilled checkpoint for a schedule it was not trained on.
- **Multi-reference editing.** Up to 10 images, ordered the way the model reads them. Optional “follow last reference aspect”.
- **Preset library.** 14 generate categories and 5 edit categories — portraits, landscapes, product, Chinese/English lettering, stickers, interiors, multi-ref composites, and more.
- **History with full parameters.** Prompt, seed, size, steps, CFG, model, hub, duration, and reference images. Restore any run.
- **Windows and macOS.** CUDA on NVIDIA; Apple Silicon via MPS for BF16 and INT4. The two INT8 entries (Image21-INT8 and Image21-Turbo-INT8) are CUDA-only.
- **Demo mode.** `python -m imgen --demo` opens the full UI without downloading 30 GB of weights.

## Models

| Key | Label | Hub IDs | Approx. size | Notes |
|---|---|---|---|---|
| `qwen-image-2.1` | Qwen-Image-2.1 | HF / ModelScope `Qwen/Qwen-Image-2.1` | ~33 GB | Official BF16. Native 2K, RGBA, 10 refs. CUDA, MPS, or CPU. |
| `qwen-image-2.1-turbo` | Qwen-Image-2.1-Turbo | HF / ModelScope `Qwen/Qwen-Image-2.1-Turbo` | ~32.4 GB | Official **8-step** distilled BF16. Carries its sampling schedule; steps and CFG are fixed. CUDA, MPS, or CPU. **Non-commercial licence.** |
| `image21-int8` | Image21-INT8 | HF `ixim/Image21-INT8` · ModelScope `iximbox/Image21-INT8` | ~19 GB | Community INT8. **NVIDIA CUDA only.** Start edits at 1024 and leave VAE tiling off. |
| `image21-turbo-int8` | Image21-Turbo-INT8 | HF `ixim/Image21-Turbo-INT8` · ModelScope `iximbox/Image21-Turbo-INT8` | ~18.5 GB | Community INT8 conversion of Turbo; the schedule key is preserved, so it runs the same 8 steps. **NVIDIA CUDA only.** **Non-commercial licence.** |
| `image21-int4` | Image21-INT4 | HF `ixim/Image21-INT4` · ModelScope `iximbox/Image21-INT4` | ~13.4 GB | Community SDNQ UINT4 + rank-32 SVD residual. CUDA, MPS, or CPU. No bitsandbytes. |

The two Turbo entries are the same `QwenImage21Pipeline` class with the same call signature as the base weights — full checkpoints, not LoRAs — so they reuse the BF16 loader untouched. Their one real difference is the recipe: eight steps on a schedule the checkpoint ships in `model_index.json` under `sample_sigmas`. This studio passes that schedule in explicitly, which reaches the same eight values on the diffusers build already installed, so the pinned commit upstream asks for is **not** required — and is not adopted, because every model here shares one pipeline class and upgrading would touch all of them.

INT8 loading uses the upstream sequential loader and offload helper from `ixim/Image21-INT8` (`imgen/int8_runtime.py`). Do not re-quantize those weights at load time. `Image21-Turbo-INT8` uses the same runtime.

INT4 uses the saved SDNQ quantization with ordinary PyTorch dequantization, adapted from the [upstream runtime](https://huggingface.co/ixim/Image21-INT4/blob/main/scripts/runtime.py) in `imgen/int4_runtime.py`. CUDA GPUs with **10 GiB or less** automatically use one transformer block / text-encoder leaf at a time, with VAE encoding and decoding on CPU. Larger CUDA GPUs use model CPU offload. MPS and CPU keep the pipeline resident. Quantized matmul stays off; the weights are not re-quantized on load.

The [model card](https://huggingface.co/ixim/Image21-INT4) suggests **24 GB unified memory** for resident MPS; its low-memory CUDA results were measured with an allocator cap on a larger GPU. An 8 GB Mac is not the target. If an MPS operator rejects BF16, launch with `IMAGE21_DTYPE=float16 python -m imgen`. INT4 also accepts `bfloat16` and `float32`. For editing, use a different seed from the source image's generation seed.

## Recommended parameters

Taken from the [official README](https://github.com/QwenLM/Qwen-Image-2.1) and Diffusers `QwenImage21Pipeline`:

| Parameter | Default | Why |
|---|---|---|
| `num_inference_steps` | **40** | Official default. |
| `true_cfg_scale` | **1.0** | Guidance off. CFG only engages with a negative prompt and `true_cfg_scale > 1`, and roughly doubles work per step. |
| `width` × `height` | **1024 × 1024** | 1K table. The Image21-INT8 card **withdrew** its general 2048px recommendation in favour of a 1024px area scale; the 2K table is still there if you want it. |
| `use_kv_cache` | **true** | Prefix cache for text and condition images. Toggling it changes the sample. |
| `output_resolution` | **2048** at 2K / **1024** at 1K | Resizes reference images when editing. |
| `vae_tiling` | **false** | Tiled VAE decoding stained 2048 editing runs; the card's earlier 2048px recipe was withdrawn in favour of 1024 with untiled VAE. Tiling buys VRAM, not quality — opt in, never assume on. |
| RGBA prompt | official wrapper | `This is an RGBA image with transparency. … The image has alpha channel and the background is transparent.` |

> **Turbo overrides two rows.** Both Turbo entries fix `num_inference_steps` at **8** and `true_cfg_scale` at **1.0** — the distillation only holds on its own recipe, so the studio locks the controls and the engine re-applies these values whatever a request asked for. Because CFG stays at 1, a negative prompt has no effect on a Turbo run. Every other row applies unchanged.

1K sizes are half of this table, each side snapped to a multiple of 32.

Official 2K aspect sizes:

```
1:1   2048 × 2048
4:3   2400 × 1792
3:4   1792 × 2400
3:2   2528 × 1696
2:3   1696 × 2528
16:9  2752 × 1536
9:16  1536 × 2752
```

Consumer GPUs should enable **CPU offload** (automatic below ~40 GiB). Leave **VAE tiling** off unless you run out of VRAM at 2K — it trades memory for short vertical stains, so it is a fallback, not a speed or quality setting.

## Requirements

- Python **3.10+** (3.12 or 3.13 preferred; Image21-INT8 was tested by ixim on 3.13 + CUDA)
- **Windows 10/11** with NVIDIA CUDA, or **macOS 13+** (Apple Silicon for practical speed)
- Disk: ~15 GB (INT4), ~20 GB (INT8 or Turbo-INT8), or ~35 GB (BF16 or Turbo BF16) plus outputs
- GPU memory: INT4 includes group offload for small CUDA GPUs; INT8 and Turbo-INT8 target larger CUDA GPUs. BF16 (base or Turbo) at 2K wants more memory or offload. MPS INT4 keeps packed weights and activations in unified memory.

PyTorch is **not** pinned in `requirements.txt` because the wheel depends on your platform.

## Install

### Windows (NVIDIA)

```powershell
git clone https://github.com/iXimNet/imgen.git
cd imgen
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# INT8 extra:
pip install -r requirements-int8.txt
# Or INT4 extra (SDNQ, no bitsandbytes required):
pip install -r requirements-int4.txt
python -m imgen
```

A helper script lives at `scripts/setup_windows.ps1`.

### macOS (Apple Silicon)

```bash
git clone https://github.com/iXimNet/imgen.git
cd imgen
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
python -m imgen
```

A helper script lives at `scripts/setup_macos.sh`.

For **Image21-INT4** on macOS, also run `pip install -r requirements-int4.txt`, then select it in Settings. Use a current macOS PyTorch wheel (the upstream INT4 CUDA runtime was tested with PyTorch 2.10). INT4 dependencies match SDNQ 0.2.7 and require Accelerate 1.15 or newer. Image21-INT8 remains CUDA-only; macOS can use Qwen-Image-2.1 or Image21-INT4.

### Try the UI without weights

```bash
pip install fastapi uvicorn python-multipart pydantic pillow httpx
python -m imgen --demo
```

Opens [http://127.0.0.1:9100](http://127.0.0.1:9100). Generation is simulated.

## First run

1. Choose language.
2. Choose **Hugging Face** or **ModelScope** (ModelScope is usually faster in mainland China).
3. Choose **Qwen-Image-2.1**, **Qwen-Image-2.1-Turbo**, **Image21-INT8**, **Image21-Turbo-INT8**, or **Image21-INT4**.
4. Optional access token (`HF_TOKEN` / ModelScope token).
5. Download. Progress streams over WebSocket.
6. Generate. The model loads on first job.

Data lives in `~/.imgen/` (override with `IMGEN_HOME`):

```
~/.imgen/
  config.json
  history.sqlite
  outputs/          # PNG, including RGBA
  thumbs/
  refs/
```

Weights stay in the hub cache (`HF_HOME`, default `~/.cache/huggingface`, or ModelScope’s cache).

If loading fails with a missing `*-00001-of-0000N.safetensors` file, the snapshot was interrupted: `model_index.json` is present but some shards are not. Open **Settings → download weights** (or run the first-run wizard again). IMGEN now treats that cache as incomplete and **resumes** only the missing shards. Evaluation images in the Image21-INT8 repo are skipped.

## Usage notes

- **Shortcut:** `Ctrl+Enter` / `⌘Enter` generates.
- **Edit mode:** drop, paste, or upload up to 10 images. Order is the model’s reading order.
- **History:** click a thumbnail to inspect every parameter, restore them, or send the image into edit mode.
- Bind address defaults to `127.0.0.1`. Pass `--host 0.0.0.0` only on a trusted network; this process can load local files and run the GPU.

```bash
python -m imgen --check          # device + download status
python -m imgen --port 9100
python -m imgen --demo --no-browser
```

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check imgen tests
```

Layout:

```
imgen/                 # Python package
  app.py               # FastAPI
  engine.py            # Diffusers / INT8 / INT4 / Turbo recipe / demo
  hub.py               # Hugging Face + ModelScope download
  int8_runtime.py      # ixim offload helper
  int4_runtime.py      # SDNQ UINT4 / CUDA offload / resident MPS and CPU
  static/              # UI (no Node build)
  data/prompts.json    # preset library
tests/
```

## Licence

- **IMGEN source:** MIT. See [LICENSE](LICENSE).
- **Third-party weights:** Qwen Research License, **non-commercial** — this includes the Turbo build, whose licence is inherited from Qwen-Image-2.1. See [NOTICE](NOTICE).
- **Turbo schedule:** read from the checkpoint’s own `model_index.json`; the built-in fallback is the eight values Qwen published with the release.
- **Image21-INT8 loader:** adapted from ixim’s `scripts/runtime.py`, behaviour preserved. `Image21-Turbo-INT8` uses the same loader.
- **Image21-INT4 loader:** adapted from ixim’s `scripts/runtime.py` and `scripts/device.py`, with lazy imports and application-owned cache paths.

## 中文摘要

IMGEN 是面向 **Qwen-Image-2.1** 的本地生图 / 改图工作室：首次运行从 Hugging Face 或 ModelScope 下载权重，内置 5 个模型 —— 官方 BF16（Qwen-Image-2.1）、官方 8 步蒸馏版（Qwen-Image-2.1-Turbo），以及社区转换的 Image21-INT8 / Image21-Turbo-INT8 / Image21-INT4。参数默认值：40 步、关闭 CFG、1024px 面积尺度、VAE Tiling 关闭（依 Image21-INT8 勘误，2K 档位仍可一键切换）；两个 Turbo 条目改由权重内置的 8 步采样配方接管，步数与 CFG 固定、界面锁定该两项控件，负向提示词不参与。改图最多 10 张参考图；按分类提供大量预置提示词；历史记录保存完整生成参数。界面不使用 Gradio。Windows（NVIDIA CUDA）与 macOS（Apple Silicon MPS，BF16 / INT4）均可运行。Image21-INT4 使用 SDNQ UINT4，安装 `requirements-int4.txt` 后即可在设置中选择；小显存 CUDA 自动分组卸载，MPS 建议 24GB 统一内存。应用代码 MIT 开源；模型权重请遵守 Qwen Research License（**仅限非商用**）。
