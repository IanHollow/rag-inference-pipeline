from collections import OrderedDict
import logging
import time
from typing import Generic, TypeVar

import lz4.frame  # type: ignore
import orjson

from ..config import get_settings
from ..telemetry import metrics

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")
settings = get_settings()


class LRUCache(Generic[K, V]):
    """
    Simple LRU Cache with TTL support.
    Not thread-safe, but safe for asyncio if used within a single event loop without context switching during get/put.
    """

    def __init__(self, capacity: int, ttl: float | None = None, name: str = "default") -> None:
        self.capacity = capacity
        self.ttl = ttl
        self.name = name
        self.cache: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Labels for metrics
        self.metric_labels = {
            "run_id": settings.profiling_run_id,
            "node": str(settings.node_number),
            "cache_name": name,
        }

        # These two lines cache frequently used methods and label dict to local variables (perf: micro/good for tight loops)
        self._misses_metric = metrics.cache_misses_total.labels(**self.metric_labels)
        self._hits_metric = metrics.cache_hits_total.labels(**self.metric_labels)

    def get(self, key: K) -> V | None:
        # Micro-optimization: pull frequently used locals
        cache = self.cache
        ttl = self.ttl
        misses_metric = self._misses_metric
        hits_metric = self._hits_metric

        if key not in cache:
            self.misses += 1
            misses_metric.inc()
            return None

        value, timestamp = cache[key]
        if ttl is not None and (time.time() - timestamp > ttl):
            del cache[key]
            self.misses += 1
            misses_metric.inc()
            return None

        cache.move_to_end(key)
        self.hits += 1
        hits_metric.inc()
        return value

    def put(self, key: K, value: V) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = (value, time.time())

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
            self.evictions += 1
            metrics.cache_evictions_total.labels(**self.metric_labels).inc()

    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0


class CompressedLRUCache(LRUCache[K, V]):
    """
    LRU Cache that compresses values using LZ4 before storage.
    Values must be serializable by orjson.
    """

    def get(self, key: K) -> V | None:
        compressed = super().get(key)
        if compressed is None:
            return None

        try:
            serialized = lz4.frame.decompress(compressed)
        except Exception:
            return None
        # orjson.loads returns on Exception for malformed input, no need to further catch unless you want to catch MemoryError
        try:
            return orjson.loads(serialized)
        except Exception:
            return None

    def put(self, key: K, value: V) -> None:
        try:
            serialized = orjson.dumps(value)
            compressed = lz4.frame.compress(serialized)
            super().put(key, compressed)
        except Exception as e:
            logger.warning("Failed to compress/store value in cache: %s", e)
