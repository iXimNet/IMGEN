# Contributing to IMGEN

Thank you for helping make a careful local studio for Qwen-Image-2.1.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -q
```

You do not need GPU weights for UI or unit tests. Use:

```bash
python -m imgen --demo
```

## Project rules

- Keep Gradio out of the dependency tree and the UI.
- Sampling defaults must match the official Qwen-Image-2.1 card unless the control is clearly labelled as an override.
- Image21-INT8 must keep using the sequential INT8 loader / offload helper. Do not “simplify” it into a plain `from_pretrained` + `enable_model_cpu_offload()` on CUDA.
- Prefer small, named modules over a single mega-file. The static UI is vanilla JS on purpose: no Node toolchain required to run.

## Adding preset prompts

Edit `imgen/data/prompts.json`.

Each prompt needs:

- `id` — stable, unique
- `title_zh` / `title_en` — short labels for the card
- `prompt` — the exact text sent to the model

Quality bar:

- Specific lighting, materials, lens or medium
- Explicit “no extra text” when lettering is involved, and quote the exact words to render
- Adult subjects only; no sexual content involving minors
- Edit prompts should say what to preserve (identity, pose, background)

After editing:

```bash
pytest tests/test_prompts.py -q
```

## Pull requests

- One concern per PR.
- Include tests when you touch sizes, history, download, or the job API.
- Do not commit weights, tokens, or `~/.imgen` outputs.
