import asyncio
import pytest
from llm_resilience.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)

@pytest.mark.anyio
async def test_circuit_breaker_transitions():
    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1, half_open_success_threshold=2)
    cb = AsyncCircuitBreaker("test-service", config)

    assert cb.state == CircuitState.CLOSED

    async def faulty():
        raise ValueError("Provider down")

    async def healthy():
        return "ok"

    # 1. First failure -> still CLOSED
    with pytest.raises(ValueError):
        await cb.call(faulty)
    assert cb.state == CircuitState.CLOSED

    # 2. Second failure -> trips to OPEN
    with pytest.raises(ValueError):
        await cb.call(faulty)
    assert cb.state == CircuitState.OPEN

    # 3. Call while OPEN raises CircuitBreakerOpenError without calling handler
    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(healthy)

    # 4. Wait for recovery timeout -> transitions to HALF_OPEN
    await asyncio.sleep(0.12)
    assert cb.state == CircuitState.HALF_OPEN

    # 5. First success in HALF_OPEN -> stays HALF_OPEN until threshold reached
    res1 = await cb.call(healthy)
    assert res1 == "ok"

    # 6. Second success -> resets to CLOSED
    res2 = await cb.call(healthy)
    assert res2 == "ok"
    assert cb.state == CircuitState.CLOSED
