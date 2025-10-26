import numpy as np
import pytest

from fuse.core.policies import QuantizedSpec


def _quantize(values: np.ndarray, scale: np.ndarray, zero: np.ndarray) -> np.ndarray:
    q = np.round(values / scale + zero).astype(np.int32)
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)


def test_quantized_spec_roundtrip_numpy():
    values = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    scale = np.full((1,), 0.05, dtype=np.float32)
    zero_point = np.zeros_like(scale)
    quantized = _quantize(values, scale, zero_point)
    spec = QuantizedSpec(mode="int8", scale=scale, zero_point=zero_point)

    restored = spec.dequantize_numpy(quantized)
    np.testing.assert_allclose(restored, values, atol=0.05, rtol=0.0)


def test_quantized_spec_roundtrip_torch():
    torch = pytest.importorskip("torch")
    values = torch.linspace(-0.5, 0.5, steps=8, dtype=torch.float32).reshape(2, 4)
    scale = np.full((1,), 0.02, dtype=np.float32)
    zero_point = np.zeros_like(scale)
    scale_t = torch.as_tensor(scale, dtype=torch.float32)
    zero_t = torch.as_tensor(zero_point, dtype=torch.float32)
    quantized = torch.clamp(
        torch.round(values / scale_t + zero_t),
        min=-128,
        max=127,
    ).to(torch.int8)
    spec = QuantizedSpec(mode="int8", scale=scale, zero_point=zero_point)

    restored = spec.dequantize_torch(quantized)
    assert restored.dtype == torch.float32
    torch.testing.assert_close(restored, values, atol=0.02, rtol=0.0)


def test_lora_adapter_shape_mismatch_raises():
    from fuse.core.policies import LoRAAdapter

    adapter = LoRAAdapter(
        name="bad",
        up=np.ones((4, 2), dtype=np.float32),
        down=np.ones((3, 3), dtype=np.float32),
        alpha=1.0,
    )
    with pytest.raises(ValueError, match="mismatched inner dims"):
        adapter.merge_numpy(np.zeros((4, 2), dtype=np.float32))
