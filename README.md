# IMGEN

Local visual studio for **Qwen-Image-2.1** — generate from text, edit with up to **10 reference images**, and keep every run in a searchable history.

Not Gradio. A FastAPI backend with a purpose-built interface. Runs on **Windows** and **macOS**.

[Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) · [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-2.1) · [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-2.1) · [Blog](https://qwen.ai/blog?id=qwen-image-2.1)

> **Weights are not MIT.** Qwen-Image-2.1 and Image21-INT8 are released under the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1). Read it before you download or publish results. This repository only ships the studio application (MIT).

---

## Features

- **Generate and edit in one pipeline.** Qwen-Image-2.1 is a unified model (`QwenImage21Pipeline`). Switching modes does not switch checkpoints.
- **Two weights, two hubs.** Choose **Qwen-Image-2.1** (official BF16) or **Image21-INT8** (community bitsandbytes conversion by [ixim](https://huggingface.co/ixim/Image21-INT8)). Download from **Hugging Face** or **ModelScope** on first run.
- **Official sampling defaults.** 40 steps, `true_cfg_scale=1.0` (guidance off), native 2K aspect table, prefix KV cache on, RGBA prompt template from the model card.
- **Multi-reference editing.** Up to 10 images, ordered the way the model reads them. Optional “follow last reference aspect”.
- **Preset library.** 14 generate categories and 5 edit categories — portraits, landscapes, product, Chinese/English lettering, stickers, interiors, multi-ref composites, and more.
- **History with full parameters.** Prompt, seed, size, steps, CFG, model, hub, duration, and reference images. Restore any run.
- **Windows and macOS.** CUDA on NVIDIA; Apple Silicon via MPS for BF16. Image21-INT8 is CUDA-only.
- **Demo mode.** `python -m imgen --demo` opens the full UI without downloading 30 GB of weights.

## Models

| Key | Label | Hub IDs | Approx. size | Notes |
|---|---|---|---|---|
| `qwen-image-2.1` | Qwen-Image-2.1 | HF / ModelScope `Qwen/Qwen-Image-2.1` | ~33 GB | Official BF16. Native 2K, RGBA, 10 refs. CUDA, MPS, or CPU. |
| `image21-int8` | Image21-INT8 | HF `ixim/Image21-INT8` · ModelScope `iximbox/Image21-INT8` | ~19 GB | Community INT8. **NVIDIA CUDA only.** Edit at 2048 with VAE tiling. |

INT8 loading uses the upstream sequential loader and offload helper from `ixim/Image21-INT8` (`imgen/int8_runtime.py`). Do not re-quantize those weights at load time.

## Recommended parameters

Taken from the [official README](https://github.com/QwenLM/Qwen-Image-2.1) and Diffusers `QwenImage21Pipeline`:

| Parameter | Default | Why |
|---|---|---|
| `num_inference_steps` | **40** | Official default. |
| `true_cfg_scale` | **1.0** | Guidance off. CFG only engages with a negative prompt and `true_cfg_scale > 1`, and roughly doubles work per step. |
| `width` × `height` | **2048 × 2048** | Native 2K. 1K table is offered for smaller VRAM. |
| `use_kv_cache` | **true** | Prefix cache for text and condition images. Toggling it changes the sample. |
| `output_resolution` | **2048** at 2K / **1024** at 1K | Resizes reference images when editing. INT8 editing was documented at 2048 with VAE tiling. |
| RGBA prompt | official wrapper | `This is an RGBA image with transparency. … The image has alpha channel and the background is transparent.` |

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

Consumer GPUs should enable **CPU offload** (automatic below ~40 GiB) and **VAE tiling** at 2K.

## Requirements

- Python **3.10+** (3.12 or 3.13 preferred; Image21-INT8 was tested by ixim on 3.13 + CUDA)
- **Windows 10/11** with NVIDIA CUDA, or **macOS 13+** (Apple Silicon for practical speed)
- Disk: ~20 GB (INT8) or ~35 GB (BF16) plus outputs
- GPU memory: INT8 is the practical path around 12–16 GiB; BF16 at 2K wants more, or offload

PyTorch is **not** pinned in `requirements.txt` because the wheel depends on your platform.

## Install

### Windows (NVIDIA)

```powershell
git clone https://github.com/YOUR_USER/imgen.git
cd imgen
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# INT8 extra:
pip install -r requirements-int8.txt
python -m imgen
```

A helper script lives at `scripts/setup_windows.ps1`.

### macOS (Apple Silicon)

```bash
git clone https://github.com/YOUR_USER/imgen.git
cd imgen
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
python -m imgen
```

A helper script lives at `scripts/setup_macos.sh`.

Image21-INT8 will refuse to load on macOS: bitsandbytes INT8 is a CUDA stack. Use Qwen-Image-2.1.

### Try the UI without weights

```bash
pip install fastapi uvicorn python-multipart pydantic pillow httpx
python -m imgen --demo
```

Opens [http://127.0.0.1:9100](http://127.0.0.1:9100). Generation is simulated.

## First run

1. Choose language.
2. Choose **Hugging Face** or **ModelScope** (ModelScope is usually faster in mainland China).
3. Choose **Qwen-Image-2.1** or **Image21-INT8**.
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

Weights stay in the hub cache (`~/.cache/huggingface` or `~/.cache/modelscope`).

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
  engine.py            # Diffusers / INT8 / demo
  hub.py               # Hugging Face + ModelScope download
  int8_runtime.py      # ixim offload helper
  static/              # UI (no Node build)
  data/prompts.json    # preset library
tests/
```

## Licence

- **IMGEN source:** MIT. See [LICENSE](LICENSE).
- **Third-party weights:** Qwen Research License. See [NOTICE](NOTICE).
- **Image21-INT8 loader:** adapted from ixim’s `scripts/runtime.py`, behaviour preserved.

## 中文摘要

IMGEN 是面向 **Qwen-Image-2.1** 的本地生图 / 改图工作室：首次运行从 Hugging Face 或 ModelScope 下载权重；参数默认值对齐官方 40 步、关闭 CFG、原生 2K；改图最多 10 张参考图；按分类提供大量预置提示词；历史记录保存完整生成参数。界面不使用 Gradio。Windows（NVIDIA CUDA）与 macOS（Apple Silicon MPS，仅 BF16）均可运行。应用代码 MIT 开源；模型权重请遵守 Qwen Research License。
