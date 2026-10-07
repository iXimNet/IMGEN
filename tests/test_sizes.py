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


def test_a_non_square_reference_is_never_measured_by_an_edge():
    """The reference control is an area, so neither edge is the number.

    `side` is the side of the square with the same pixel count — the geometric
    mean of the two edges — so a portrait reference comes out taller than the
    number and a landscape one wider. Both directions have to hold, or the
    studio would be printing a shape the engine never produces.
    """
    for refw, refh in ((1792, 2400), (1920, 1080), (3000, 1000)):
        w, h = calculate_dimensions(1024 * 1024, refw / refh)
        assert abs(w * h / (1024 * 1024) - 1) < 0.04, (refw, refh, w, h)
        assert min(w, h) < 1024 < max(w, h), (refw, refh, w, h)
        assert w % 32 == 0 and h % 32 == 0


def test_one_reference_makes_the_canvas_and_the_reference_the_same_shape():
    """Follow on with a single reference, and the two coincide.

    Same area, same ratio, so the engine derives one shape twice — which is why
    the panel can print a single resolved pair for it. Nothing of the sort holds
    for the other references in a batch: the canvas takes the last reference's
    ratio while every reference keeps its own, so 896x1184 is what the panel
    prints for this one and not what the output frame will be if it is not last.
    """
    refw, refh = 1792, 2400
    resolved = follow_reference_size(1024, refw, refh)
    assert resolved == calculate_dimensions(1024 * 1024, refw / refh)
    assert resolved == (896, 1184)
