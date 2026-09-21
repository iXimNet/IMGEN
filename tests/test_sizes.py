from imgen.constants import ASPECT_RATIOS_1K, ASPECT_RATIOS_2K
from imgen.rgba import is_rgba_prompt, wrap_rgba_prompt
from imgen.sizes import calculate_dimensions, size_for


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
