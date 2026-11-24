import asyncio
import pytest
from llm_resilience.rate_limiter import KeyPool, SlidingWindowLimiter

@pytest.mark.anyio
async def test_sliding_window_limiter():
    limiter = SlidingWindowLimiter(max_requests=3, window_seconds=0.2)

    assert await limiter.acquire() is True
    assert await limiter.acquire() is True
    assert await limiter.acquire() is True
    assert await limiter.acquire() is False
    assert await limiter.remaining() == 0

    await asyncio.sleep(0.22)
    assert await limiter.acquire() is True

@pytest.mark.anyio
async def test_key_pool_rotation_and_limits():
    pool = KeyPool(keys=["key_a", "key_b"], rpm_limit=1, window_seconds=0.2)

    k1 = await pool.get_available_key()
    assert k1 == "key_a"

    k2 = await pool.get_available_key()
    assert k2 == "key_b"

    # Both keys exhausted
    k3 = await pool.get_available_key()
    assert k3 is None

    # After window passes, keys become available again
    await asyncio.sleep(0.22)
    k4 = await pool.get_available_key()
    assert k4 in ("key_a", "key_b")
