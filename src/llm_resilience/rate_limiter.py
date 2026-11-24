"""
Multi-key rotation and sliding window rate limiting for LLM provider pools.
"""
from __future__ import annotations

import asyncio
from collections import deque
import time
from typing import Any, Deque, Dict, List, Optional


class SlidingWindowLimiter:
    """
    In-memory / Redis-compatible sliding window rate limiter.
    Accurately limits calls within a moving time window without fixed-boundary burst spikes.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: Deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Attempt to acquire permission for one request. Returns True if granted, False if rate limited."""
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return True
            return False

    async def remaining(self) -> int:
        """Return the number of requests remaining in the current window."""
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            return max(0, self.max_requests - len(self._timestamps))


class KeyPool:
    """
    Manages a pool of API keys for a provider (e.g., free tier keys or multi-tenant credits),
    rotating between available keys while respecting per-key sliding window limits.
    """

    def __init__(self, keys: List[str], rpm_limit: int = 30, window_seconds: float = 60.0):
        if not keys:
            raise ValueError("KeyPool must contain at least one key.")
        self.keys = keys
        self.rpm_limit = rpm_limit
        self.window_seconds = window_seconds
        self._limiters: Dict[str, SlidingWindowLimiter] = {
            k: SlidingWindowLimiter(rpm_limit, window_seconds) for k in keys
        }
        self._rr_index = 0
        self._lock = asyncio.Lock()

    async def get_available_key(self) -> Optional[str]:
        """
        Returns the next available API key from the pool that has capacity remaining.
        Returns None if all keys in the pool are currently rate-limited.
        """
        async with self._lock:
            num_keys = len(self.keys)
            for i in range(num_keys):
                candidate_index = (self._rr_index + i) % num_keys
                candidate_key = self.keys[candidate_index]
                limiter = self._limiters[candidate_key]
                if await limiter.acquire():
                    self._rr_index = (candidate_index + 1) % num_keys
                    return candidate_key
            return None
