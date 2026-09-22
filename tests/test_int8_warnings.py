import warnings

from imgen.int8_runtime import silence_int8_bf16_cast_warnings


def test_matmul8bit_warning_is_swallowed_even_with_always_filter():
    silence_int8_bf16_cast_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_int8_bf16_cast_warnings()
        warnings.warn(
            "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
        )
        warnings.warn("unrelated warning")
    texts = [str(item.message) for item in caught]
    assert not any("MatMul8bitLt" in text for text in texts)
    assert any("unrelated warning" in text for text in texts)
