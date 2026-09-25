import logging
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


def test_bitsandbytes_logger_warning_is_filtered(caplog):
    silence_int8_bf16_cast_warnings()
    log = logging.getLogger("bitsandbytes.autograd._functions")
    with caplog.at_level(logging.WARNING, logger="bitsandbytes"):
        log.warning(
            "MatMul8bitLt: inputs will be cast from %s to float16 during quantization",
            "torch.bfloat16",
        )
        log.warning("other bnb warning")
    messages = [record.getMessage() for record in caplog.records]
    assert not any("MatMul8bitLt" in text for text in messages)
    assert any("other bnb warning" in text for text in messages)
