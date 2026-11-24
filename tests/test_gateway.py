import pytest
from llm_resilience.circuit_breaker import CircuitBreakerConfig
from llm_resilience.gateway import LLMGateway, ProviderEndpoint

@pytest.mark.anyio
async def test_gateway_failover():
    call_log = []

    async def fail_handler(prompt, kwargs):
        call_log.append("groq")
        raise ConnectionError("503 Service Unavailable")

    async def success_handler(prompt, kwargs):
        call_log.append("nvidia")
        return "response from nvidia"

    cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0)
    p1 = ProviderEndpoint(name="groq", model="llama-3", handler=fail_handler, breaker_config=cfg)
    p2 = ProviderEndpoint(name="nvidia", model="llama-3.2", handler=success_handler, breaker_config=cfg)

    gateway = LLMGateway(providers=[p1, p2])

    result = await gateway.complete("Generate summary")
    assert result.content == "response from nvidia"
    assert result.provider == "nvidia"
    assert result.fallback_used is True
    assert call_log == ["groq", "nvidia"]
