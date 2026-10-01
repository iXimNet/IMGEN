from imgen.prompts import catalog


def test_preset_volume():
    data = catalog()
    counts = data["counts"]
    assert counts["generate_categories"] >= 10
    assert counts["generate_prompts"] >= 70
    assert counts["edit_categories"] >= 4
    assert counts["edit_prompts"] >= 18


def test_preset_shape():
    data = catalog()
    for group in ("generate", "edit"):
        for category in data[group]:
            assert category["id"]
            assert category["zh"] and category["en"]
            assert category["prompts"]
            for prompt in category["prompts"]:
                assert prompt["prompt"].strip()
                assert prompt["title_zh"] and prompt["title_en"]


def test_negative_bundles():
    """The negative library is a flat list of toggleable bundles."""
    data = catalog()
    bundles = data["negative"]
    assert len(bundles) >= 5
    assert data["counts"]["negative_bundles"] == len(bundles)

    ids = set()
    for bundle in bundles:
        assert bundle["id"] and bundle["id"] not in ids
        ids.add(bundle["id"])
        assert bundle["zh"] and bundle["en"]
        zh, en = bundle["terms_zh"], bundle["terms_en"]
        assert zh and en
        # Both languages describe the same bundle, so a missing translation
        # would quietly drop a term or ship an untranslated one.
        assert len(zh) == len(en), bundle["id"]
        assert all(term.strip() for term in zh + en)


def test_negative_terms_are_disjoint():
    """No term may belong to two bundles.

    The panel derives each bundle's on/off state from the textarea and, when a
    bundle is switched off, removes its terms. A shared term would therefore be
    torn out of a bundle that is still supposed to be on.
    """
    bundles = catalog()["negative"]
    owner: dict[str, str] = {}
    for bundle in bundles:
        for term in bundle["terms_zh"] + bundle["terms_en"]:
            key = term.strip().lower()
            assert key not in owner, f"{term!r} in both {owner.get(key)} and {bundle['id']}"
            owner[key] = bundle["id"]
