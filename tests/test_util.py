"""
Tests for axsdb.util.
"""

from __future__ import annotations

from axsdb.util import ClosingLRUCache


class TestClosingLRUCache:
    class FakeDataset:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def test_closes_on_eviction(self):
        cache = ClosingLRUCache(maxsize=1)
        ds1 = self.FakeDataset()
        ds2 = self.FakeDataset()

        cache["1"] = ds1
        cache["2"] = ds2  # evicts "1" (maxsize=1)

        assert ds1.closed
        assert not ds2.closed
        assert "2" in cache

    def test_closes_on_clear(self):
        cache = ClosingLRUCache(maxsize=4)
        ds1 = self.FakeDataset()
        ds2 = self.FakeDataset()

        cache["1"] = ds1
        cache["2"] = ds2

        cache.clear()

        assert ds1.closed and ds2.closed
        assert len(cache) == 0
