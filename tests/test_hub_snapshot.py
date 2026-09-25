import json
from pathlib import Path

import pytest

from imgen.constants import repo_id
from imgen.hub import (
    hub_cache_root,
    hub_snapshot_state,
    hub_storage,
    missing_weight_files,
    model_status,
    resolve_local_hub,
    snapshot_is_complete,
)


def _write_index(folder, shards):
    folder.mkdir(parents=True, exist_ok=True)
    mapping = {f"w{i}": name for i, name in enumerate(shards)}
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping}),
        encoding="utf-8",
    )
    return folder


def _isolate_caches(tmp_path, monkeypatch):
    """Point both caches at empty directories so the host's real ones stay out."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "modelscope"))


def _complete_snapshot(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    _write_index(root / "transformer", ["diffusion_pytorch_model.safetensors"])
    for folder, name in (
        ("transformer", "diffusion_pytorch_model.safetensors"),
        ("text_encoder", "model.safetensors"),
        ("vae", "diffusion_pytorch_model.safetensors"),
    ):
        target = root / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(b"x" * 2048)
    return root


def _hf_snapshot(tmp_path, model_key):
    repo = repo_id(model_key, "huggingface")
    return (
        tmp_path
        / "hf"
        / "hub"
        / ("models--" + repo.replace("/", "--"))
        / "snapshots"
        / "abc123"
    )


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


@pytest.mark.parametrize("model_key", ["image21-int8", "image21-int4"])
def test_modelscope_hub_models_layout_is_discovered(tmp_path, monkeypatch, model_key):
    """ModelScope stores snapshots under `hub/models/<namespace>/<name>`."""
    _isolate_caches(tmp_path, monkeypatch)
    repo = repo_id(model_key, "modelscope")
    snap = _complete_snapshot(tmp_path / "modelscope" / "hub" / "models" / repo)

    state = hub_snapshot_state(model_key, "modelscope")

    assert state["complete"] is True
    assert state["path"] == snap


@pytest.mark.parametrize("model_key", ["image21-int8", "image21-int4"])
def test_resolve_local_hub_falls_back_to_the_other_source(tmp_path, monkeypatch, model_key):
    """Weights from one hub must stay usable while the studio points at the other."""
    _isolate_caches(tmp_path, monkeypatch)
    _complete_snapshot(_hf_snapshot(tmp_path, model_key))

    assert resolve_local_hub(model_key, "huggingface") == "huggingface"
    assert resolve_local_hub(model_key, "modelscope") == "huggingface"
    assert resolve_local_hub("qwen-image-2.1", "huggingface") is None


def test_model_status_separates_selected_source_from_disk(tmp_path, monkeypatch):
    _isolate_caches(tmp_path, monkeypatch)
    _complete_snapshot(_hf_snapshot(tmp_path, "image21-int8"))

    row = {item["key"]: item for item in model_status("modelscope")}["image21-int8"]

    assert row["downloaded"] is False, "not in the selected source"
    assert row["downloaded_any"] is True, "but present on disk, so it is usable"
    assert row["available_hubs"] == ["huggingface"]
    assert row["local_hub"] == "huggingface"
    assert row["incomplete_any"] is False


def test_incomplete_snapshot_is_reported_with_a_missing_count(tmp_path, monkeypatch):
    _isolate_caches(tmp_path, monkeypatch)
    snap = _complete_snapshot(_hf_snapshot(tmp_path, "image21-int8"))
    (snap / "transformer" / "diffusion_pytorch_model.safetensors").unlink()

    state = hub_snapshot_state("image21-int8", "huggingface")
    row = {item["key"]: item for item in model_status("huggingface")}["image21-int8"]

    assert state["complete"] is False and state["incomplete"] is True
    assert row["downloaded"] is False
    assert row["downloaded_any"] is False
    assert row["incomplete_any"] is True
    assert row["missing_count"] >= 1


def test_hub_storage_prefers_env_vars_over_defaults(tmp_path, monkeypatch):
    """Storage paths follow HF_HUB_CACHE > HF_HOME > MODELSCOPE_CACHE > defaults."""
    for name in ("HF_HUB_CACHE", "HF_HOME", "MODELSCOPE_CACHE"):
        monkeypatch.delenv(name, raising=False)

    hf = hub_storage("huggingface")
    assert hf["path"] == str(Path.home() / ".cache" / "huggingface" / "hub")
    assert hf["env_var"] is None
    assert hub_storage("modelscope")["path"] == str(Path.home() / ".cache" / "modelscope")

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    assert hub_storage("huggingface")["path"] == str(tmp_path / "hfhome" / "hub")
    assert hub_storage("huggingface")["env_var"] == "HF_HOME"

    # HF_HUB_CACHE wins over HF_HOME.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hubcache"))
    assert hub_storage("huggingface")["path"] == str(tmp_path / "hubcache")
    assert hub_storage("huggingface")["env_var"] == "HF_HUB_CACHE"

    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "mscache"))
    assert hub_storage("modelscope")["path"] == str(tmp_path / "mscache")
    assert hub_storage("modelscope")["env_var"] == "MODELSCOPE_CACHE"
    assert hub_cache_root("modelscope") == tmp_path / "mscache"


def test_huggingface_hub_cache_env_drives_lookup(tmp_path, monkeypatch):
    """Weights are found under HF_HUB_CACHE without any HF_HOME in play."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hubcache"))
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "modelscope"))
    repo = repo_id("image21-int8", "huggingface")
    snap = _complete_snapshot(
        tmp_path / "hubcache" / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc"
    )

    assert hub_snapshot_state("image21-int8", "huggingface")["path"] == snap
    assert resolve_local_hub("image21-int8", "huggingface") == "huggingface"
