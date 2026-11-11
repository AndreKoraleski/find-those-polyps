from pathlib import Path

import torch

from find_those_polyps.evaluation.store import FeatureStore


def test_shards_persist_features_across_stores(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    store.put(["a/1.png", "a/2.png", ""], torch.arange(6.0).reshape(3, 2), dense=False)
    store.put(["a/1.png"], torch.ones(1, 2, 2, 2).half(), dense=True)
    value = store.get("a/2.png", False)
    assert value is not None and value.tolist() == [2.0, 3.0]
    store.flush()
    fresh = FeatureStore(tmp_path)
    first, dense = fresh.get("a/1.png", False), fresh.get("a/1.png", True)
    assert first is not None and first.tolist() == [0.0, 1.0]
    assert dense is not None and dense.dtype == torch.float16 and dense.shape == (2, 2, 2)
    assert fresh.get("", False) is None
    assert fresh.get("a/2.png", True) is None


def test_memory_store_keeps_global_features_only() -> None:
    store = FeatureStore()
    store.put(["a/1.png"], torch.ones(1, 2), dense=False)
    store.put(["a/1.png"], torch.ones(1, 2, 2, 2), dense=True)
    store.flush()
    assert store.get("a/1.png", False) is not None
    assert store.get("a/1.png", True) is None


def test_unreadable_shards_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "global").mkdir()
    (tmp_path / "global" / "broken.pt").write_bytes(b"not a shard")
    assert FeatureStore(tmp_path).get("a/1.png", False) is None
