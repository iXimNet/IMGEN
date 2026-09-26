import json
from pathlib import Path

import pytest

from imgen.constants import repo_id
from imgen.hub import (
    HF_IGNORE_PATTERNS,
    MS_IGNORE_PATTERNS,
    _modelscope_repo_dirs,
    _ms_matches_any,
    describe_weight_dir,
    extra_weight_dirs,
    hub_cache_root,
    hub_snapshot_state,
    hub_storage,
    missing_weight_files,
    model_local_path,
    model_status,
    resolve_local_hub,
    resolve_snapshot_dir,
    set_extra_weight_dirs,
    snapshot_is_complete,
    sniff_snapshot_model,
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
    # Extra dirs are process-global; a folder left by one test must not leak
    # into another's candidate list.
    set_extra_weight_dirs([])


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


def _qwen_snapshot(root, precision="bf16"):
    """A complete snapshot that *identifies itself* the way real ones do.

    Content sniffing reads `model_index.json` for the pipeline family and
    `conversion.json` for the precision, so a folder that should be recognised
    by its contents needs both. `precision` is `bf16`, `int8` or `int4`.
    """
    _complete_snapshot(root)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImage21Pipeline",
                "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
            }
        ),
        encoding="utf-8",
    )
    if precision == "int8":
        (root / "conversion.json").write_text(
            json.dumps({"method": "bitsandbytes LLM.int8"}), encoding="utf-8"
        )
    elif precision == "int4":
        (root / "conversion.json").write_text(
            json.dumps({"method": "sdnq", "weights_dtype": "uint4"}),
            encoding="utf-8",
        )
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


@pytest.mark.parametrize("model_key", ["image21-int8", "image21-int4", "qwen-image-2.1"])
def test_modelscope_current_layout_is_discovered(tmp_path, monkeypatch, model_key):
    """Current SDKs write `models/<owner>--<name>/snapshots/<revision>/`.

    Missing this layout made a finished download look absent: the studio showed
    the Download button again and generation refused with NOT_DOWNLOADED.
    """
    _isolate_caches(tmp_path, monkeypatch)
    repo = repo_id(model_key, "modelscope")
    owner, name = repo.split("/", 1)
    snap = _complete_snapshot(
        tmp_path
        / "modelscope"
        / "models"
        / f"{owner}--{name}"
        / "snapshots"
        / "master"
    )

    assert model_local_path(model_key, "modelscope") == snap
    state = hub_snapshot_state(model_key, "modelscope")
    assert state["complete"] is True
    assert state["missing"] == []

    row = {item["key"]: item for item in model_status("modelscope")}[model_key]
    assert row["downloaded"] is True
    assert row["downloaded_any"] is True
    assert row["local_hub"] == "modelscope"
    assert row["incomplete_any"] is False


def test_modelscope_snapshot_revision_is_picked_within_repo_dir(tmp_path, monkeypatch):
    """A repo dir alone is not a snapshot; the revision folder underneath is."""
    _isolate_caches(tmp_path, monkeypatch)
    repo = repo_id("image21-int8", "modelscope")
    owner, name = repo.split("/", 1)
    repo_dir = tmp_path / "modelscope" / "models" / f"{owner}--{name}"
    older = _complete_snapshot(repo_dir / "snapshots" / "aaa111")
    newer = _complete_snapshot(repo_dir / "snapshots" / "zzz999")

    found = model_local_path("image21-int8", "modelscope")

    assert found in (older, newer)
    assert found.parent.name == "snapshots", "must return the revision dir, not the repo root"


def test_resolve_snapshot_dir_descends_into_snapshots(tmp_path):
    """The downloader may return the repo root; resolution finds the revision."""
    repo_dir = tmp_path / "models--acme--thing"
    snap = _complete_snapshot(repo_dir / "snapshots" / "master")

    assert resolve_snapshot_dir(repo_dir) == snap
    assert resolve_snapshot_dir(snap) == snap


def test_modelscope_repo_dirs_are_unique(tmp_path):
    """Candidate probing must not scan the same directory twice."""
    dirs = _modelscope_repo_dirs(tmp_path, "iximbox/Image21-INT8")

    assert len(dirs) == len({str(item) for item in dirs})


@pytest.mark.parametrize("model_key", ["image21-int8", "image21-int4"])
def test_modelscope_ignore_patterns_actually_match(tmp_path, monkeypatch, model_key):
    """The ModelScope client matches with fnmatch, not regex.

    The old regex-style patterns (`.*\\.png$`) never matched anything — `$` is
    a literal to fnmatch — so README.md and every gallery image were fetched.
    """
    # Files the SDK must skip.
    for path in (
        "README.md",
        "evaluation/gallery/01.png",
        "assets/teaser.jpg",
        "docs/preview.webp",
        "cards/card.png",
        "scripts/run.sh",
        "benchmarks/results.json",
        "tests/test_x.py",
    ):
        assert _ms_matches_any(path, MS_IGNORE_PATTERNS), f"{path} should be skipped"

    # Files the SDK must keep: weight shards and their configs.
    for path in (
        "model_index.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "text_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
        "tokenizer/vocab.json",
    ):
        assert not _ms_matches_any(path, MS_IGNORE_PATTERNS), f"{path} must be kept"


def test_ignore_patterns_agree_across_hubs():
    """Both hubs must skip the same set, or a source switch changes the bytes.

    Verified through each hub's own matcher: Hugging Face uses
    ``huggingface_hub.utils._paths``'s matcher (``fnmatchcase`` before 1.x,
    renamed to plain ``fnmatch`` after) and ModelScope uses
    ``modelscope_hub._download._matches_patterns``.
    """
    hf_paths = pytest.importorskip("huggingface_hub.utils._paths")
    # The helper was renamed between huggingface_hub generations; resolve
    # whichever one this install ships so the pin stays version-agnostic.
    hf_fnmatch = getattr(hf_paths, "fnmatchcase", None) or hf_paths.fnmatch

    def hf_ignores(path: str) -> bool:
        return any(hf_fnmatch(path, pattern) for pattern in HF_IGNORE_PATTERNS)

    samples = [
        "README.md",
        "evaluation/gallery/01.png",
        "assets/teaser.jpg",
        "docs/preview.webp",
        "cards/card.png",
        "scripts/run.sh",
        "benchmarks/results.json",
        "tests/test_x.py",
        "model_index.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
    ]
    for path in samples:
        assert hf_ignores(path) == _ms_matches_any(
            path, MS_IGNORE_PATTERNS
        ), f"hubs disagree about {path}"

    assert HF_IGNORE_PATTERNS == MS_IGNORE_PATTERNS, "one list serves both hubs"


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


def test_extra_dir_pointed_at_snapshot_is_found(tmp_path, monkeypatch):
    """A user folder that *is* the snapshot root counts as a usable copy."""
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "my-weights" / "Image21-INT8"
    _complete_snapshot(extra)
    set_extra_weight_dirs([extra])

    state = hub_snapshot_state("image21-int8", "huggingface")
    assert state["complete"] is True
    assert state["external"] is True
    assert model_local_path("image21-int8", "huggingface") == extra
    # Generation resolves through the same layer, so it sees the folder too.
    assert resolve_local_hub("image21-int8", "huggingface") == "huggingface"


def test_extra_dir_in_hf_cache_layout_is_found(tmp_path, monkeypatch):
    """A folder that looks like a carried-over HF cache is probed too."""
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "carried-over"
    repo = repo_id("image21-int8", "huggingface")
    snap = _complete_snapshot(
        extra / ("models--" + repo.replace("/", "--")) / "snapshots" / "rev1"
    )
    set_extra_weight_dirs([extra])

    assert model_local_path("image21-int8", "huggingface") == snap


def test_extra_dir_in_modelscope_layout_is_found(tmp_path, monkeypatch):
    """A folder written by a ModelScope client also holds a usable copy."""
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "carried-over"
    repo = repo_id("image21-int8", "modelscope")
    snap = _complete_snapshot(
        extra / "models" / repo.replace("/", "--") / "snapshots" / "master"
    )
    set_extra_weight_dirs([extra])

    assert model_local_path("image21-int8", "modelscope") == snap


def test_model_status_marks_extra_dir_copy_external(tmp_path, monkeypatch):
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "elsewhere" / "Qwen-Image-2.1"
    _complete_snapshot(extra)
    set_extra_weight_dirs([extra])

    rows = {row["key"]: row for row in model_status("huggingface")}
    assert rows["qwen-image-2.1"]["downloaded_any"] is True
    assert rows["qwen-image-2.1"]["external"] is True
    # Models without a copy anywhere stay absent.
    assert rows["image21-int8"]["downloaded_any"] is False
    assert rows["image21-int8"]["external"] is False


def test_extra_dirs_never_shadow_the_cache_copy(tmp_path, monkeypatch):
    """The hub cache stays authoritative: an extra folder cannot downgrade it."""
    _isolate_caches(tmp_path, monkeypatch)
    cached = _hf_snapshot(tmp_path, "image21-int8")
    _complete_snapshot(cached)
    extra = tmp_path / "stale" / "Image21-INT8"
    _complete_snapshot(extra)
    set_extra_weight_dirs([extra])

    state = hub_snapshot_state("image21-int8", "huggingface")
    assert state["complete"] is True
    assert state["path"] == cached
    assert state["external"] is False


def test_describe_weight_dir_reports_models_and_reasons(tmp_path, monkeypatch):
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "hand" / "Image21-INT4"
    _complete_snapshot(extra)

    found = describe_weight_dir(extra.parent)
    assert found["ok"] is True
    assert found["models"] == ["Image21-INT4"]

    assert describe_weight_dir(str(tmp_path / "nope"))["reason"] == "missing"
    a_file = tmp_path / "afile"
    a_file.write_text("x", encoding="utf-8")
    assert describe_weight_dir(a_file)["reason"] == "not_a_dir"
    assert describe_weight_dir("   ")["reason"] == "empty"


def test_describe_weight_dir_probes_both_hub_repo_ids(tmp_path, monkeypatch):
    """`ixim/...` (HF) and `iximbox/...` (ModelScope) both count."""
    _isolate_caches(tmp_path, monkeypatch)
    hf_repo = repo_id("image21-int8", "huggingface")
    ms_repo = repo_id("image21-int8", "modelscope")
    assert hf_repo != ms_repo

    ms_style = tmp_path / "ms-dump"
    _complete_snapshot(ms_style / "models" / ms_repo.replace("/", "--") / "snapshots" / "master")
    assert describe_weight_dir(ms_style)["models"] == ["Image21-INT8"]

    hf_style = tmp_path / "hf-dump"
    _complete_snapshot(hf_style / ("models--" + hf_repo.replace("/", "--")) / "snapshots" / "r")
    assert describe_weight_dir(hf_style)["models"] == ["Image21-INT8"]


def test_set_extra_weight_dirs_dedupes_and_skips_blanks(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    stored = set_extra_weight_dirs([str(a), str(b), str(a), "  ", None])
    assert stored == [a, b]
    assert extra_weight_dirs() == [a, b]
    set_extra_weight_dirs(None)
    assert extra_weight_dirs() == []


# ---------------------------------------------------------------------------
# Content sniffing: folders that do not name the repo they hold
# ---------------------------------------------------------------------------


def test_sniff_snapshot_model_reads_each_precision(tmp_path):
    """The snapshot's own metadata says which precision it is."""
    assert sniff_snapshot_model(_qwen_snapshot(tmp_path / "plain")) == "qwen-image-2.1"
    assert sniff_snapshot_model(_qwen_snapshot(tmp_path / "q8", "int8")) == "image21-int8"
    assert sniff_snapshot_model(_qwen_snapshot(tmp_path / "q4", "int4")) == "image21-int4"


def test_sniff_snapshot_model_refuses_foreign_pipelines(tmp_path):
    """A folder holding some other model must never be claimed."""
    other = tmp_path / "sd"
    other.mkdir()
    (other / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}), encoding="utf-8"
    )
    assert sniff_snapshot_model(other) is None

    empty = tmp_path / "not-a-snapshot"
    empty.mkdir()
    assert sniff_snapshot_model(empty) is None


def test_sniff_snapshot_model_refuses_unknown_quantisation(tmp_path):
    """Quantised without saying how — INT8 and INT4 are not guessable."""
    ambiguous = _complete_snapshot(tmp_path / "ambiguous")
    (ambiguous / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImage21Pipeline"}), encoding="utf-8"
    )
    (ambiguous / "transformer-quantization.json").write_text("{}", encoding="utf-8")
    assert sniff_snapshot_model(ambiguous) is None


def test_extra_dir_named_by_precision_is_found(tmp_path, monkeypatch):
    """A folder called `bf16` / `int8` is recognised by its contents."""
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "weights-store"
    _qwen_snapshot(extra / "bf16")
    _qwen_snapshot(extra / "int8", "int8")
    set_extra_weight_dirs([extra])

    assert model_local_path("qwen-image-2.1", "huggingface") == extra / "bf16"
    assert model_local_path("image21-int8", "huggingface") == extra / "int8"
    # Nothing here holds INT4, and the BF16 copy must not answer for it.
    assert model_local_path("image21-int4", "huggingface") is None


def test_extra_dir_sniffing_survives_one_extra_level(tmp_path, monkeypatch):
    """A dump that keeps a tool folder in between is still reached."""
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "dumps"
    _qwen_snapshot(extra / "some-tool" / "Qwen-Image-2.1")
    set_extra_weight_dirs([extra])

    assert model_local_path("qwen-image-2.1", "huggingface") == (
        extra / "some-tool" / "Qwen-Image-2.1"
    )


def test_extra_dir_sniffing_does_not_cross_models(tmp_path, monkeypatch):
    """The whole point of sniffing: precision decides, never the folder order."""
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "store"
    _qwen_snapshot(extra / "int8", "int8")

    # The INT8 folder is complete and sits alone, but must not satisfy INT4.
    assert model_local_path("image21-int4", "huggingface") is None
    assert describe_weight_dir(extra)["models"] == ["Image21-INT8"]


def test_named_candidates_stay_ahead_of_sniffed_ones(tmp_path, monkeypatch):
    """A folder that names itself is not overruled by an unnamed sibling."""
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "mixed"
    named = _qwen_snapshot(extra / "Qwen-Image-2.1")
    _qwen_snapshot(extra / "bf16")
    set_extra_weight_dirs([extra])

    assert model_local_path("qwen-image-2.1", "huggingface") == named


def test_sniffing_is_cached_until_the_dir_list_changes(tmp_path, monkeypatch):
    """Repeated lookups do not re-walk the folder; a config change does.

    The cache only saves the *walk*: file existence is still checked on every
    lookup, so a folder that disappears is noticed. What is asserted here is
    that the folder is not re-read for every model/hub pair asked about.
    """
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "store"
    _qwen_snapshot(extra / "bf16")
    set_extra_weight_dirs([extra])

    import imgen.hub as hub_module

    calls = []
    real_sniff = hub_module.sniff_snapshot_model

    def counting_sniff(path):
        calls.append(path)
        return real_sniff(path)

    monkeypatch.setattr(hub_module, "sniff_snapshot_model", counting_sniff)

    model_local_path("qwen-image-2.1", "huggingface")
    first = len(calls)
    assert first == 1

    # The same folder list answers from cache, for every model and hub asked.
    model_local_path("image21-int8", "huggingface")
    model_local_path("qwen-image-2.1", "modelscope")
    assert len(calls) == first

    # A new folder list invalidates the cache and forces a fresh walk.
    set_extra_weight_dirs([extra])
    model_local_path("qwen-image-2.1", "huggingface")
    assert len(calls) > first


def test_describe_weight_dir_reports_precision_named_folders(tmp_path, monkeypatch):
    """The settings panel names what it found, not just that it found something."""
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "weights-store"
    _qwen_snapshot(extra / "bf16")
    _qwen_snapshot(extra / "int8", "int8")
    _qwen_snapshot(extra / "int4", "int4")

    found = describe_weight_dir(extra)
    assert found["ok"] is True
    assert sorted(found["models"]) == ["Image21-INT4", "Image21-INT8", "Qwen-Image-2.1"]


def test_model_status_marks_sniffed_extra_copy_external(tmp_path, monkeypatch):
    """A precision-named folder shows up as ready and labelled external."""
    _isolate_caches(tmp_path, monkeypatch)
    extra = tmp_path / "weights-store"
    _qwen_snapshot(extra / "bf16")
    set_extra_weight_dirs([extra])

    rows = {row["key"]: row for row in model_status("huggingface")}
    assert rows["qwen-image-2.1"]["downloaded_any"] is True
    assert rows["qwen-image-2.1"]["external"] is True
    assert rows["qwen-image-2.1"]["path"] == str(extra / "bf16")
    assert rows["image21-int8"]["downloaded_any"] is False
