"""
Stateful async circuit breaker for unreliable upstream LLM providers.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(Exception):
    """Raised when an execution is attempted while the circuit breaker is OPEN."""
    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker '{name}' is OPEN. Retry after {retry_after:.1f}s.")


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    half_open_success_threshold: int = 2


class AsyncCircuitBreaker:
    """
    Thread/async-safe circuit breaker protecting downstream services from cascading LLM timeouts/503s.
    Transitions:
      CLOSED -> (failure_threshold consecutive errors) -> OPEN
      OPEN -> (recovery_timeout expires) -> HALF_OPEN
      HALF_OPEN -> (half_open_success_threshold consecutive successes) -> CLOSED
      HALF_OPEN -> (1 failure) -> OPEN
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.config.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute the async callable guarded by the circuit breaker state."""
        async with self._lock:
            current_state = self.state
            if current_state == CircuitState.OPEN:
                retry_after = max(0.0, self.config.recovery_timeout - (time.monotonic() - self._last_failure_time))
                raise CircuitBreakerOpenError(self.name, retry_after)
            elif current_state == CircuitState.HALF_OPEN:
                self._state = CircuitState.HALF_OPEN

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.config.half_open_success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._consecutive_successes = 0
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._last_failure_time = time.monotonic()
            if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                self._consecutive_failures += 1
                if self._state == CircuitState.HALF_OPEN or self._consecutive_failures >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._consecutive_successes = 0
