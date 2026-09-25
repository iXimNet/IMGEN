import json

from imgen.config import DEFAULTS, ConfigStore
from imgen.engine import resolve_vae_tiling
from imgen.paths import AppPaths


def test_default_is_off():
    assert DEFAULTS["vae_tiling"] == "off"


def test_legacy_auto_in_a_config_file_is_retired(tmp_path):
    """Existing installs stored `auto`; loading must not leave it in place."""
    paths = AppPaths(tmp_path)
    paths.ensure()
    paths.config_file.write_text(json.dumps({"vae_tiling": "auto"}), encoding="utf-8")

    assert ConfigStore(paths).load()["vae_tiling"] == "off"


def test_nothing_enables_tiling_implicitly():
    """The Image21-INT8 card traced 2K stains to tiled VAE decoding, so no mode
    or resolution may switch tiling on by itself any more."""
    assert resolve_vae_tiling("auto") is False
    assert resolve_vae_tiling("off") is False
    assert resolve_vae_tiling(None) is False
    assert resolve_vae_tiling("") is False
    assert resolve_vae_tiling(False) is False


def test_legacy_auto_no_longer_tiles_at_2k():
    """`auto` used to mean "tile 2K edits" — that recipe was withdrawn."""
    assert resolve_vae_tiling("auto") is False


def test_explicit_on_is_the_only_way():
    assert resolve_vae_tiling("true") is True
    assert resolve_vae_tiling("on") is True
    assert resolve_vae_tiling("1") is True
    assert resolve_vae_tiling(True) is True
