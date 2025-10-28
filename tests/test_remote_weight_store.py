import io
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pytest
from urllib import error as urllib_error

from fuse.core.policies import HTTPWeightStore, S3WeightStore


class _FakeHTTPResponse:
    def __init__(self, data: bytes, headers: Mapping[str, str]):
        self._buffer = io.BytesIO(data)
        self._headers = dict(headers)

    def info(self):
        return self

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._headers.get(key, default)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._buffer.close()
        return False

    def close(self) -> None:
        self._buffer.close()


def _npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def test_http_weight_store_caches_and_validates_etag(tmp_path, monkeypatch):
    array_a = np.arange(16, dtype=np.float32)
    array_b = np.arange(16, dtype=np.float32).reshape(4, 4)
    data_a = _npy_bytes(array_a)
    data_b = _npy_bytes(array_b)

    url_a = "https://example.com/vector.npy"
    url_b = "https://example.com/matrix.npy"

    resources: Dict[str, Dict[str, Any]] = {
        url_a: {"data": data_a, "etag": '"etag-vector"'},
        url_b: {"data": data_b, "etag": '"etag-matrix"'},
    }
    head_calls: Dict[str, int] = {}
    get_calls: Dict[str, int] = {}

    def fake_urlopen(request, timeout=None):
        if hasattr(request, "get_method"):
            method = request.get_method()
            url = request.full_url
        else:
            method = "GET"
            url = str(request)
        resource = resources.get(url)
        if resource is None:
            raise urllib_error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        if method == "HEAD":
            head_calls[url] = head_calls.get(url, 0) + 1
            headers = {"ETag": resource["etag"], "Content-Length": str(len(resource["data"]))}
            return _FakeHTTPResponse(b"", headers)
        get_calls[url] = get_calls.get(url, 0) + 1
        return _FakeHTTPResponse(resource["data"], {"ETag": resource["etag"]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    weights = {
        "vector": {"url": url_a, "etag": "etag-vector"},
        "matrix": {"url": url_b, "etag": "etag-matrix"},
    }
    cache_dir = tmp_path / "cache"
    max_bytes = min(len(data_a), len(data_b)) + 64
    store = HTTPWeightStore(weights, cache_dir=cache_dir, max_bytes=max_bytes, timeout=5.0)

    resolved_a = store.resolve("vector")
    assert isinstance(resolved_a, np.memmap)
    np.testing.assert_allclose(resolved_a, array_a)
    assert get_calls[url_a] == 1

    # Cache hit should not trigger another GET
    resolved_a_cached = store.resolve("vector")
    np.testing.assert_allclose(resolved_a_cached, array_a)
    assert get_calls[url_a] == 1

    # Loading the second weight should evict the first due to budget
    resolved_b = store.resolve("matrix")
    assert isinstance(resolved_b, np.memmap)
    np.testing.assert_allclose(resolved_b, array_b)
    assert get_calls[url_b] == 1
    assert "vector" not in store._records  # noqa: SLF001

    # Update remote ETag without updating spec; validation should fail
    resources[url_a]["etag"] = '"etag-vector-new"'
    resources[url_a]["data"] = _npy_bytes(array_a * 2)
    with pytest.raises(RuntimeError, match="failed ETag validation"):
        store.resolve("vector")
    assert get_calls[url_a] == 1  # Download should not have retriggered


class _FakeS3Client:
    def __init__(self, objects: Mapping[Tuple[str, Optional[str]], Dict[str, Any]]):
        self._objects = dict(objects)

    def head_object(self, **params):
        record = self._lookup(params)
        return {"ETag": record["etag"], "ContentLength": len(record["data"])}

    def get_object(self, **params):
        record = self._lookup(params)
        return {"Body": io.BytesIO(record["data"])}

    def _lookup(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        key = params["Key"]
        version = params.get("VersionId")
        record = self._objects.get((key, version))
        if record is None:
            raise KeyError(f"Missing object for key={key}, version={version}")
        return record


def test_s3_weight_store_etag_validation(tmp_path):
    array_a = np.arange(8, dtype=np.float32)
    array_b = np.arange(8, dtype=np.float32) + 1
    data_a = _npy_bytes(array_a)
    data_b = _npy_bytes(array_b)

    objects: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {
        ("weights/a.npy", None): {"data": data_a, "etag": '"etag-a"'},
        ("weights/b.npy", None): {"data": data_b, "etag": '"etag-b"'},
    }
    fake_client = _FakeS3Client(objects)

    specs = {
        "a": {"key": "weights/a.npy", "etag": "etag-a"},
        "b": {"key": "weights/b.npy", "etag": "etag-b"},
    }
    cache_dir = tmp_path / "cache"
    max_bytes = len(data_a) + 64
    store = S3WeightStore(
        bucket="test-bucket",
        objects=specs,
        cache_dir=cache_dir,
        max_bytes=max_bytes,
        client=fake_client,
    )

    resolved_a = store.resolve("a")
    assert isinstance(resolved_a, np.memmap)
    np.testing.assert_allclose(resolved_a, array_a)

    resolved_b = store.resolve("b")
    assert isinstance(resolved_b, np.memmap)
    np.testing.assert_allclose(resolved_b, array_b)
    assert "a" not in store._records  # noqa: SLF001

    # Change ETag in backing store without updating spec -> expect validation failure
    objects[("weights/a.npy", None)]["etag"] = '"etag-a-new"'
    with pytest.raises(RuntimeError, match="failed ETag validation"):
        store.resolve("a")
