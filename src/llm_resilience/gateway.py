"""
Multi-provider LLM gateway orchestrator with automatic failover and circuit protection.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional
from pydantic import BaseModel

from llm_resilience.circuit_breaker import AsyncCircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError


class ProviderResult(BaseModel):
    content: str
    provider: str
    model: str
    latency_seconds: float
    fallback_used: bool = False


@dataclass
class ProviderEndpoint:
    name: str
    model: str
    handler: Callable[[str, Dict[str, Any]], Awaitable[str]]
    breaker: AsyncCircuitBreaker = field(init=False)
    breaker_config: Optional[CircuitBreakerConfig] = None

    def __post_init__(self):
        self.breaker = AsyncCircuitBreaker(name=self.name, config=self.breaker_config)


class LLMGateway:
    """
    Resilient orchestrator that queries configured LLM providers in priority order,
    bypassing tripped providers and falling back seamlessly to healthy alternatives.
    """

    def __init__(self, providers: List[ProviderEndpoint]):
        if not providers:
            raise ValueError("LLMGateway requires at least one ProviderEndpoint.")
        self.providers = providers

    async def complete(self, prompt: str, **kwargs: Any) -> ProviderResult:
        """
        Execute completion with failover through the provider chain.
        Raises RuntimeError if all providers fail or are tripped.
        """
        errors = []
        for idx, provider in enumerate(self.providers):
            try:
                import time
                t0 = time.monotonic()
                content = await provider.breaker.call(provider.handler, prompt, kwargs)
                latency = time.monotonic() - t0
                return ProviderResult(
                    content=content,
                    provider=provider.name,
                    model=provider.model,
                    latency_seconds=latency,
                    fallback_used=(idx > 0),
                )
            except CircuitBreakerOpenError as e:
                errors.append(f"[{provider.name}] Circuit OPEN ({e})")
                continue
            except Exception as e:
                errors.append(f"[{provider.name}] Execution error: {e}")
                continue

        error_summary = "; ".join(errors)
        raise RuntimeError(f"All LLM providers failed. Diagnostics: {error_summary}")
