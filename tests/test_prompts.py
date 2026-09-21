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
