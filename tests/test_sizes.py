from imgen.constants import ASPECT_RATIOS_1K, ASPECT_RATIOS_2K, DEFAULT_SCALE
from imgen.rgba import is_rgba_prompt, wrap_rgba_prompt
from imgen.sizes import calculate_dimensions, catalog, follow_reference_size, size_for


def test_default_scale_is_the_1k_table():
    """1K is the default: the card withdrew its general 2048px recommendation."""
    assert DEFAULT_SCALE == "1k"
    assert catalog()["default_scale"] == "1k"
    assert catalog()["scales"]["1k"]["sizes"]["1:1"] == (1024, 1024)


def test_official_2k_multiples_of_32():
    for ratio, (w, h) in ASPECT_RATIOS_2K.items():
        assert w % 32 == 0 and h % 32 == 0, ratio


def test_1k_multiples_of_32():
    for ratio, (w, h) in ASPECT_RATIOS_1K.items():
        assert w % 32 == 0 and h % 32 == 0, (ratio, w, h)


def test_size_for_matches_tables():
    assert size_for("2k", "1:1") == (2048, 2048)
    assert size_for("2k", "16:9") == (2752, 1536)
    w, h = size_for("1k", "1:1")
    assert (w, h) == (1024, 1024)


def test_calculate_dimensions_square():
    w, h = calculate_dimensions(2048 * 2048, 1.0)
    assert w == 2048 and h == 2048


def test_rgba_wrapper():
    wrapped = wrap_rgba_prompt("a sticker of a fox")
    assert is_rgba_prompt(wrapped)
    assert wrapped.startswith("This is an RGBA image with transparency.")
    again = wrap_rgba_prompt(wrapped)
    assert again == wrapped


def test_follow_reference_size_takes_the_resolved_area():
    """The reference area control is what the engine and the studio both use,
    so the same input must give the same size — no scale-key lookup."""
    w, h = follow_reference_size(1024, 1200, 1600)
    assert (w, h) == calculate_dimensions(1024 * 1024, 1200 / 1600)
    assert w % 32 == 0 and h % 32 == 0
    assert w < h, "portrait reference must stay portrait"
    # A different area must actually change the result.
    assert follow_reference_size(2048, 1200, 1600) != (w, h)


def test_follow_reference_size_survives_a_degenerate_reference():
    w, h = follow_reference_size(1024, 800, 0)
    assert w > 0 and h > 0


def test_follow_reference_size_takes_the_resolved_area():
    """The reference area control is what the engine and the studio both use,
    so the same input must give the same size — no scale-key lookup."""
    w, h = follow_reference_size(1024, 1200, 1600)
    assert (w, h) == calculate_dimensions(1024 * 1024, 1200 / 1600)
    assert w % 32 == 0 and h % 32 == 0
    assert w < h, "portrait reference must stay portrait"
    # A different area must actually change the result.
    assert follow_reference_size(2048, 1200, 1600) != (w, h)


def test_follow_reference_size_survives_a_degenerate_reference():
    w, h = follow_reference_size(1024, 800, 0)
    assert w > 0 and h > 0
