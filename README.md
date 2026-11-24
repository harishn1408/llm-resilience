# llm-resilience

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Production-grade resilience primitives for asynchronous LLM gateways:
- **Stateful Async Circuit Breakers**: Prevents cascading 429/503 timeouts across OpenAI, Anthropic, Groq, and NVIDIA NIM endpoints.
- **Sliding-Window Rate Limiting & Multi-Key Pools**: Smooth rate-limit management and round-robin multi-key rotation avoiding burst edge penalties.
- **Deterministic Gateway Orchestration**: Automatic fallback failover through provider chains with latency telemetry.

---

## Architecture

```
Client Request
      │
      ▼
┌───────────────┐      Circuit OPEN
│  LLMGateway   │ ──────────────────────┐
└───────┬───────┘                       │
        │ Try Primary                   ▼
        ▼                        ┌──────────────┐
┌───────────────┐   Fail/Trip    │ Try Fallback │
│  Groq / NIM   │ ─────────────> │    OpenAI    │
└───────────────┘                └──────┬───────┘
                                        │
                                        ▼
                                 ProviderResult
```

---

## Installation

```bash
pip install llm-resilience
```

---

## Quickstart

```python
import asyncio
from llm_resilience import AsyncCircuitBreaker, CircuitBreakerConfig

breaker = AsyncCircuitBreaker(
    name="openai-primary",
    config=CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
)

async def call_llm():
    return await breaker.call(client.chat.completions.create, model="gpt-4o", messages=[...])
```

---

## License

MIT &copy; 2026 Harish Nair
