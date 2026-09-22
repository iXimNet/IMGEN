import json

from imgen.hub import missing_weight_files, snapshot_is_complete


def _write_index(folder, shards):
    folder.mkdir(parents=True, exist_ok=True)
    mapping = {f"w{i}": name for i, name in enumerate(shards)}
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping}),
        encoding="utf-8",
    )
    return folder


def test_complete_sharded_snapshot(tmp_path):
    root = tmp_path / "snap"
    root.mkdir()
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    transformer = _write_index(
        root / "transformer",
        [
            "diffusion_pytorch_model-00001-of-00002.safetensors",
            "diffusion_pytorch_model-00002-of-00002.safetensors",
        ],
    )
    for name in (
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ):
        (transformer / name).write_bytes(b"x" * 2048)
    text = root / "text_encoder"
    text.mkdir()
    (text / "model.safetensors").write_bytes(b"x" * 2048)
    vae = root / "vae"
    vae.mkdir()
    (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 2048)
    assert missing_weight_files(root) == []
    assert snapshot_is_complete(root) is True


def test_missing_first_transformer_shard(tmp_path):
    root = tmp_path / "snap"
    root.mkdir()
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    transformer = _write_index(
        root / "transformer",
        [
            "diffusion_pytorch_model-00001-of-00002.safetensors",
            "diffusion_pytorch_model-00002-of-00002.safetensors",
        ],
    )
    (transformer / "diffusion_pytorch_model-00002-of-00002.safetensors").write_bytes(b"x" * 2048)
    for folder in ("text_encoder", "vae"):
        d = root / folder
        d.mkdir()
        (d / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 2048)
    missing = missing_weight_files(root)
    assert any("00001-of-00002" in item for item in missing)
    assert snapshot_is_complete(root) is False


def test_zero_byte_shard_counts_as_missing(tmp_path):
    root = tmp_path / "snap"
    root.mkdir()
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    transformer = _write_index(
        root / "transformer",
        ["diffusion_pytorch_model-00001-of-00002.safetensors"],
    )
    (transformer / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"")
    for folder in ("text_encoder", "vae"):
        d = root / folder
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"x" * 2048)
    missing = missing_weight_files(root)
    assert any("00001-of-00002" in item for item in missing)
