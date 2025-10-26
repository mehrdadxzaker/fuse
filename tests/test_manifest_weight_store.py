import numpy as np
import pytest

from fuse.core.policies import ManifestWeightStore, RuntimePolicies


def _quantize(values: np.ndarray, scale: np.ndarray, zero_point: np.ndarray) -> np.ndarray:
    return np.round(values / scale + zero_point).astype(np.int8)

def test_manifest_weight_store_memmaps_numpy(tmp_path):
    values = np.arange(8, dtype=np.float32)
    np.save(tmp_path / "vector.npy", values)
    manifest = {"weights": [{"name": "vector", "path": "vector.npy"}]}

    store = ManifestWeightStore(manifest, base_path=tmp_path, cache_bytes=4096, strict=True)
    resolved = store.resolve("vector")

    assert isinstance(resolved.data, np.memmap)
    np.testing.assert_allclose(resolved.data, values)


def test_manifest_weight_store_strict_rejects_compressed_npz(tmp_path):
    values = np.arange(4, dtype=np.float32)
    compressed_path = tmp_path / "compressed.npz"
    np.savez_compressed(compressed_path, arr=values)
    manifest = {"weights": [{"name": "compressed", "path": "compressed.npz"}]}

    store = ManifestWeightStore(manifest, base_path=tmp_path, cache_bytes=4096, strict=True)

    with pytest.raises(RuntimeError):
        store.resolve("compressed")


def test_manifest_weight_store_prefetch_quant_lora(tmp_path):
    # Prepare shard files for block0
    block0_part0 = np.full((1, 2), 1.0, dtype=np.float32)
    block0_part1 = np.full((1, 2), 3.0, dtype=np.float32)
    np.save(tmp_path / "block0_part0.npy", block0_part0)
    np.save(tmp_path / "block0_part1.npy", block0_part1)

    # Prepare quantized payload for block1
    base_float = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale = np.array([0.5], dtype=np.float32)
    zero_point = np.array([0.0], dtype=np.float32)
    quant_data = _quantize(base_float, scale, zero_point)
    np.save(tmp_path / "block1_quant.npy", quant_data)

    # LoRA adapters
    up_base = np.array([[0.1], [0.2]], dtype=np.float32)
    down_base = np.array([[0.3, 0.4]], dtype=np.float32)
    up_alt = np.array([[0.05], [0.01]], dtype=np.float32)
    down_alt = np.array([[0.6, -0.2]], dtype=np.float32)

    manifest = {
        "weights": [
            {
                "name": "block0.weight",
                "shards": [
                    {"path": "block0_part0.npy"},
                    {"path": "block0_part1.npy"},
                ],
                "dtype": "float32",
                "shard_axis": 0,
                "scale": {"value": 2.0},
                "prefetch": {"group": "decoder", "order": 0, "window": 1},
            },
            {
                "name": "block1.weight",
                "path": "block1_quant.npy",
                "dtype": "int8",
                "quant": {
                    "mode": "int8",
                    "scale": {"value": scale.tolist()},
                    "zero_point": {"value": zero_point.tolist()},
                },
                "lora": {
                    "slot": "decoder",
                    "merge": "on_the_fly",
                    "default": "base",
                    "adapters": {
                        "base": {
                            "up": {"value": up_base.tolist()},
                            "down": {"value": down_base.tolist()},
                            "alpha": 2.0,
                        },
                        "alt": {
                            "up": {"value": up_alt.tolist()},
                            "down": {"value": down_alt.tolist()},
                            "alpha": 3.0,
                        },
                    },
                },
                "prefetch": {"group": "decoder", "order": 1, "window": 1},
            },
        ]
    }

    store = ManifestWeightStore(manifest, base_path=tmp_path, cache_bytes=4096)
    policies = RuntimePolicies(weight_store=store)

    # Resolve block0 and ensure shards concatenate with scale applied.
    resolved_block0 = store.resolve("block0.weight")
    block0_array = policies.materialize_weight(
        "block0.weight",
        resolved_block0,
        backend="numpy",
        device=None,
    )
    expected_block0 = np.concatenate([block0_part0, block0_part1], axis=0) * 2.0
    np.testing.assert_allclose(block0_array, expected_block0)

    # Prefetch window should have pulled block1 into cache.
    assert "block1.weight" in store._cache  # noqa: SLF001 - inspecting cache for test

    # Resolve block1 with base adapter.
    resolved_block1 = store.resolve("block1.weight")
    block1_array = policies.materialize_weight(
        "block1.weight",
        resolved_block1,
        backend="numpy",
        device=None,
    )
    base_reconstructed = (quant_data.astype(np.float32) - zero_point) * scale
    update_base = (2.0 / float(up_base.shape[1])) * (up_base @ down_base)
    expected_block1_base = base_reconstructed + update_base
    np.testing.assert_allclose(block1_array, expected_block1_base)

    # Switch adapter on the fly and expect the resolved weight to change.
    store.activate_adapter("decoder", "alt")
    resolved_block1_alt = store.resolve("block1.weight")
    block1_array_alt = policies.materialize_weight(
        "block1.weight",
        resolved_block1_alt,
        backend="numpy",
        device=None,
    )
    update_alt = (3.0 / float(up_alt.shape[1])) * (up_alt @ down_alt)
    expected_block1_alt = base_reconstructed + update_alt
    np.testing.assert_allclose(block1_array_alt, expected_block1_alt)
    assert not np.allclose(block1_array_alt, block1_array)

    # Cache budget respected
    assert store._resident_bytes <= store.cache_bytes
