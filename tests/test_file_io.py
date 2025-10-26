import json
from pathlib import Path

import numpy as np
import pytest

from fuse.core.builtins import (
    BagOfWordsTensor,
    read_tensor_from_file,
    write_tensor_to_file,
)


def test_read_tensor_from_json_dense(tmp_path):
    payload = [[1, 2], [3, 4]]
    path = tmp_path / "tensor.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = read_tensor_from_file(str(path))

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.asarray(payload))


def test_write_jsonl_topk_includes_schema_and_valid_payload(tmp_path):
    topk = [[(0, 0.5), (1, 0.25)], [(2, 0.9)]]
    path = tmp_path / "scores.jsonl"

    write_tensor_to_file(path, topk)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines, "jsonl output must not be empty"
    meta = json.loads(lines[0])
    assert meta == {"schema": "fuse.topk", "version": 1}
    first_row = json.loads(lines[1])
    assert "topk" in first_row
    assert first_row["topk"][0] == {"index": 0, "value": 0.5}


def test_write_jsonl_tensor_handles_unicode(tmp_path):
    arr = np.asarray([["α", "β"]])
    path = tmp_path / "tensor.jsonl"

    write_tensor_to_file(path, arr)

    payload_lines = path.read_text(encoding="utf-8").splitlines()
    assert "α" in payload_lines[1]
    meta = json.loads(payload_lines[0])
    assert meta["schema"] == "fuse.tensor"
    value = json.loads(payload_lines[1])["value"]
    assert value == [["α", "β"]]


def test_text_to_bow_returns_tensor_and_sidecar(tmp_path):
    text_path = tmp_path / "doc.txt"
    text_path.write_text("hello world hello", encoding="utf-8")

    tensor = read_tensor_from_file(str(text_path))

    assert isinstance(tensor, BagOfWordsTensor)
    np_tensor = np.asarray(tensor)
    assert np_tensor.shape[0] == 3
    assert set(tensor.vocab.keys()) == {"hello", "world"}
    vocab_path = Path(str(text_path.with_suffix("")) + ".vocab.json")
    assert vocab_path.exists()

    # Re-reading should honour the persisted vocabulary.
    tensor_round_trip = read_tensor_from_file(str(text_path))
    assert tensor_round_trip.vocab == tensor.vocab
    np.testing.assert_array_equal(np.asarray(tensor_round_trip), np_tensor)


def test_read_tensor_memmaps_npy(tmp_path):
    data = np.arange(6, dtype=np.float32)
    path = tmp_path / "tensor.npy"
    np.save(path, data)

    loaded = read_tensor_from_file(str(path), strict=True)

    assert isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(loaded, data)


def test_read_tensor_strict_rejects_compressed_npz(tmp_path):
    data = np.arange(3, dtype=np.float32)
    path = tmp_path / "tensor.npz"
    np.savez_compressed(path, arr=data)

    with pytest.raises(RuntimeError):
        read_tensor_from_file(str(path), strict=True, mmap_threshold_bytes=0)
