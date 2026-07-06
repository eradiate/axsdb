"""
Generic utilities.

This module collects small, generic, reusable helpers that are not specific
to any particular AxsDB domain concept (databases, interpolation, units...).
"""

from __future__ import annotations

from cachetools import LRUCache


class ClosingLRUCache(LRUCache):
    """
    LRUCache that closes evicted xarray Datasets, so that lazily opened files
    do not stay open until Python garbage-collects them.
    """

    def popitem(self):
        key, ds = super().popitem()
        ds.close()
        return key, ds
