from imgen.engine import resolve_vae_tiling


def test_auto_t2i_never_tiles():
    assert resolve_vae_tiling("auto", mode="generate", output_resolution=2048) is False
    assert resolve_vae_tiling("auto", mode="generate", output_resolution=1024) is False


def test_auto_edit_tiles_only_at_2k():
    assert resolve_vae_tiling("auto", mode="edit", output_resolution=2048) is True
    assert resolve_vae_tiling("auto", mode="edit", output_resolution=1024) is False


def test_explicit_false_string_is_off():
    assert resolve_vae_tiling("false", mode="generate", output_resolution=2048) is False
    assert resolve_vae_tiling("off", mode="edit", output_resolution=2048) is False


def test_explicit_true_is_on():
    assert resolve_vae_tiling("true", mode="generate", output_resolution=1024) is True
    assert resolve_vae_tiling(True, mode="generate", output_resolution=1024) is True
