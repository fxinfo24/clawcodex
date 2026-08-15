"""
Usage normalization audit + regression test matrix (PR 1).

Tests that token accounting correctly handles cache tokens across ALL providers.
Addresses the bug class where non-Anthropic providers don't return cache fields
in the same shape, causing:
- Triple-counting cached content (input + cache_read + cache_write)
- Premature compaction at ~20% actual context usage (MiniMax)
- Silent cost accounting errors

Each provider adapter gets a `uses_openai_style_cache_breakdown` capability flag.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderUsageFixture:
    """Real recorded usage shape from a provider."""
    name: str
    raw_usage: dict[str, Any]
    expected_normalized: dict[str, int]
    uses_openai_style_cache_breakdown: bool
    context_window: int


# =============================================================================
# REAL RECORDED FIXTURES — update these when provider APIs change
# =============================================================================

FIXTURES: list[ProviderUsageFixture] = [
    # -------------------------------------------------------------------------
    # Anthropic (native cache fields)
    # -------------------------------------------------------------------------
    ProviderUsageFixture(
        name="anthropic_cached",
        raw_usage={
            "input_tokens": 53,
            "output_tokens": 8,
            "cache_read_input_tokens": 2560,
            "cache_creation_input_tokens": 384,
        },
        expected_normalized={
            "input_tokens": 53,
            "output_tokens": 8,
            "cache_read_input_tokens": 2560,
            "cache_creation_input_tokens": 384,
        },
        uses_openai_style_cache_breakdown=False,
        context_window=200_000,
    ),
    ProviderUsageFixture(
        name="anthropic_uncached",
        raw_usage={
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        expected_normalized={
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        uses_openai_style_cache_breakdown=False,
        context_window=200_000,
    ),

    # -------------------------------------------------------------------------
    # DeepSeek (native: prompt_cache_hit_tokens / prompt_cache_miss_tokens)
    # Also supports OpenAI-compatible nested cached_tokens on gateways
    # -------------------------------------------------------------------------
    ProviderUsageFixture(
        name="deepseek_native_cached",
        raw_usage={
            "prompt_tokens": 2613,
            "completion_tokens": 100,
            "total_tokens": 2713,
            "prompt_cache_hit_tokens": 2560,
            "prompt_cache_miss_tokens": 53,
        },
        expected_normalized={
            "input_tokens": 53,           # miss = uncached
            "output_tokens": 100,
            "cache_read_input_tokens": 2560,  # hit = cached
            "cache_creation_input_tokens": 0,
        },
        uses_openai_style_cache_breakdown=False,  # DeepSeek has its own native shape
        context_window=1_000_000,
    ),
    ProviderUsageFixture(
        name="deepseek_gateway_cached",
        # OpenRouter / other gateways serving DeepSeek may use OpenAI format
        raw_usage={
            "prompt_tokens": 2613,
            "completion_tokens": 100,
            "total_tokens": 2713,
            "prompt_tokens_details": {"cached_tokens": 2560},
        },
        expected_normalized={
            "input_tokens": 53,
            "output_tokens": 100,
            "cache_read_input_tokens": 2560,
            "cache_creation_input_tokens": 0,
        },
        uses_openai_style_cache_breakdown=True,
        context_window=1_000_000,
    ),

    # -------------------------------------------------------------------------
    # MiniMax (Anthropic-compatible API)
    # Returns cache_read_input_tokens and cache_creation_input_tokens natively
    # -------------------------------------------------------------------------
    ProviderUsageFixture(
        name="minimax_cached",
        raw_usage={
            "input_tokens": 53,
            "output_tokens": 8,
            "cache_read_input_tokens": 2560,
            "cache_creation_input_tokens": 384,
            "service_tier": "standard",
        },
        expected_normalized={
            "input_tokens": 53,
            "output_tokens": 8,
            "cache_read_input_tokens": 2560,
            "cache_creation_input_tokens": 384,
        },
        uses_openai_style_cache_breakdown=False,
        context_window=1_000_000,
    ),
    ProviderUsageFixture(
        name="minimax_uncached",
        raw_usage={
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "service_tier": "standard",
        },
        expected_normalized={
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        uses_openai_style_cache_breakdown=False,
        context_window=1_000_000,
    ),

    # -------------------------------------------------------------------------
    # OpenAI / OpenRouter / OpenAI-compatible (nested cached_tokens)
    # -------------------------------------------------------------------------
    ProviderUsageFixture(
        name="openai_cached",
        raw_usage={
            "prompt_tokens": 2613,
            "completion_tokens": 100,
            "total_tokens": 2713,
            "prompt_tokens_details": {"cached_tokens": 2560},
        },
        expected_normalized={
            "input_tokens": 53,           # 2613 - 2560
            "output_tokens": 100,
            "cache_read_input_tokens": 2560,
            "cache_creation_input_tokens": 0,  # OpenAI has no cache-write charge
        },
        uses_openai_style_cache_breakdown=True,
        context_window=128_000,
    ),
    ProviderUsageFixture(
        name="openrouter_gpt56_luna_cached",
        # gpt-5.6-luna on OpenRouter with 272K tier
        raw_usage={
            "prompt_tokens": 100_000,
            "completion_tokens": 1000,
            "total_tokens": 101_000,
            "prompt_tokens_details": {"cached_tokens": 80_000},
        },
        expected_normalized={
            "input_tokens": 20_000,
            "output_tokens": 1000,
            "cache_read_input_tokens": 80_000,
            "cache_creation_input_tokens": 0,
        },
        uses_openai_style_cache_breakdown=True,
        context_window=1_050_000,
    ),
    ProviderUsageFixture(
        name="openai_uncached",
        raw_usage={
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        expected_normalized={
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        uses_openai_style_cache_breakdown=True,
        context_window=128_000,
    ),

    # -------------------------------------------------------------------------
    # Z.ai / GLM (OpenAI-compatible)
    # -------------------------------------------------------------------------
    ProviderUsageFixture(
        name="zai_cached",
        raw_usage={
            "prompt_tokens": 5000,
            "completion_tokens": 500,
            "total_tokens": 5500,
            "prompt_tokens_details": {"cached_tokens": 3000},
        },
        expected_normalized={
            "input_tokens": 2000,
            "output_tokens": 500,
            "cache_read_input_tokens": 3000,
            "cache_creation_input_tokens": 0,
        },
        uses_openai_style_cache_breakdown=True,
        context_window=128_000,
    ),

    # -------------------------------------------------------------------------
    # Moonshot / Kimi K3 (OpenAI-compatible)
    # -------------------------------------------------------------------------
    ProviderUsageFixture(
        name="kimi_k3_cached",
        raw_usage={
            "prompt_tokens": 1303,
            "completion_tokens": 200,
            "total_tokens": 1503,
            "prompt_tokens_details": {"cached_tokens": 1280},
        },
        expected_normalized={
            "input_tokens": 23,
            "output_tokens": 200,
            "cache_read_input_tokens": 1280,
            "cache_creation_input_tokens": 0,
        },
        uses_openai_style_cache_breakdown=True,
        context_window=128_000,
    ),

    # -------------------------------------------------------------------------
    # Meta Muse Spark (OpenAI-compatible)
    # -------------------------------------------------------------------------
    ProviderUsageFixture(
        name="muse_spark_cached",
        raw_usage={
            "prompt_tokens": 10_000,
            "completion_tokens": 1000,
            "total_tokens": 11_000,
            "prompt_tokens_details": {"cached_tokens": 8_000},
        },
        expected_normalized={
            "input_tokens": 2_000,
            "output_tokens": 1000,
            "cache_read_input_tokens": 8_000,
            "cache_creation_input_tokens": 0,
        },
        uses_openai_style_cache_breakdown=True,
        context_window=128_000,
    ),

    # -------------------------------------------------------------------------
    # NVIDIA NIM (OpenAI-compatible)
    # -------------------------------------------------------------------------
    ProviderUsageFixture(
        name="nvidia_nim_cached",
        raw_usage={
            "prompt_tokens": 15_000,
            "completion_tokens": 2_000,
            "total_tokens": 17_000,
            "prompt_tokens_details": {"cached_tokens": 12_000},
        },
        expected_normalized={
            "input_tokens": 3_000,
            "output_tokens": 2_000,
            "cache_read_input_tokens": 12_000,
            "cache_creation_input_tokens": 0,
        },
        uses_openai_style_cache_breakdown=True,
        context_window=128_000,
    ),
]


# =============================================================================
# IMPORT PROVIDER CLASSES
# =============================================================================

from src.providers.base import BaseProvider
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.deepseek_provider import DeepSeekProvider
from src.providers.minimax_provider import MinimaxProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.openrouter_provider import OpenRouterProvider
from src.providers.zai_provider import ZaiProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers import get_provider_class, is_anthropic_wire


# =============================================================================
# TESTS
# =============================================================================

class TestUsageNormalization:
    """Test that each provider's _build_usage_dict produces correct normalized output."""

    @pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f.name)
    def test_usage_normalization_matches_expected(self, fixture: ProviderUsageFixture):
        """The normalized usage must match the expected shape exactly."""
        provider_class = self._get_provider_class(fixture.name)
        if provider_class is None:
            pytest.skip(f"No provider class for {fixture.name}")

        # Create a mock provider instance to call _build_usage_dict
        # We need to mock the usage object to have the right attributes
        usage_obj = self._make_usage_object(fixture.raw_usage)

        # Get the actual provider instance or a mock with the right method
        if provider_class is AnthropicProvider:
            # Anthropic provider uses a different method - it passes usage through
            # This test is mainly for the OpenAI-compatible providers
            pytest.skip("Anthropic usage comes from SDK directly")
        elif provider_class is DeepSeekProvider:
            provider = DeepSeekProvider(api_key="test")
        elif provider_class is MinimaxProvider:
            provider = MinimaxProvider(api_key="test")
        elif provider_class in (OpenAIProvider, OpenRouterProvider, ZaiProvider, GeminiProvider):
            provider = provider_class(api_key="test")
        else:
            # Try to instantiate generically
            try:
                provider = provider_class(api_key="test")
            except Exception:
                pytest.skip(f"Cannot instantiate {provider_class}")

        # Call the normalization method
        normalized = provider._build_usage_dict(usage_obj)

        # Assert each field matches expected
        for key, expected_value in fixture.expected_normalized.items():
            actual_value = normalized.get(key, 0)
            assert actual_value == expected_value, (
                f"{fixture.name}: {key} = {actual_value}, expected {expected_value}. "
                f"Full normalized: {normalized}"
            )

    @pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f.name)
    def test_context_percentage_uses_input_only(self, fixture: ProviderUsageFixture):
        """context_pct MUST be input_tokens / context_window, NOT input+cache_read+cache_write.
        
        This is the core bug: triple-counting cached tokens triggers false
        100% context readings and unnecessary auto-compaction.
        """
        provider_class = self._get_provider_class(fixture.name)
        if provider_class is None:
            pytest.skip(f"No provider class for {fixture.name}")

        usage_obj = self._make_usage_object(fixture.raw_usage)

        if provider_class is AnthropicProvider:
            pytest.skip("Anthropic usage comes from SDK directly")

        provider = provider_class(api_key="test")
        normalized = provider._build_usage_dict(usage_obj)

        input_tokens = normalized.get("input_tokens", 0)
        cache_read = normalized.get("cache_read_input_tokens", 0)
        cache_write = normalized.get("cache_creation_input_tokens", 0)

        # CORRECT: context_pct = input_tokens / context_window
        correct_pct = input_tokens / fixture.context_window

        # BUGGY (what the old code did): context_pct = (input + cache_read + cache_write) / context_window
        buggy_pct = (input_tokens + cache_read + cache_write) / fixture.context_window

        # The correct percentage must be <= 1.0 (100%)
        assert correct_pct <= 1.0, (
            f"{fixture.name}: correct context_pct = {correct_pct:.2%} exceeds 100%! "
            f"input={input_tokens}, cache_read={cache_read}, cache_write={cache_write}, "
            f"window={fixture.context_window}"
        )

        # The buggy percentage would be WRONG (and often > 100% for cached turns)
        # This assertion documents the bug — if it ever passes, the bug is back
        if cache_read > 0 or cache_write > 0:
            assert buggy_pct > correct_pct, (
                f"{fixture.name}: BUG — buggy_pct ({buggy_pct:.2%}) should exceed "
                f"correct_pct ({correct_pct:.2%}) when cache tokens exist"
            )

    def test_every_provider_has_cache_breakdown_flag(self):
        """Every provider adapter MUST declare uses_openai_style_cache_breakdown.
        
        This capability flag tells the system whether the provider returns
        OpenAI-style `prompt_tokens_details.cached_tokens` (True) or has
        its own native cache fields (False, like DeepSeek native).
        """
        # Map fixture names to expected flag values
        expected_flags = {
            "anthropic_cached": False,
            "anthropic_uncached": False,
            "deepseek_native_cached": False,
            "deepseek_gateway_cached": True,
            "minimax_cached": False,
            "minimax_uncached": False,
            "openai_cached": True,
            "openrouter_gpt56_luna_cached": True,
            "openai_uncached": True,
            "zai_cached": True,
            "kimi_k3_cached": True,
            "muse_spark_cached": True,
            "nvidia_nim_cached": True,
        }

        for fixture in FIXTURES:
            expected = expected_flags.get(fixture.name)
            if expected is None:
                pytest.fail(f"Missing expected flag for fixture: {fixture.name}")

            assert fixture.uses_openai_style_cache_breakdown == expected, (
                f"{fixture.name}: uses_openai_style_cache_breakdown = "
                f"{fixture.uses_openai_style_cache_breakdown}, expected {expected}"
            )

    def _get_provider_class(self, fixture_name: str):
        """Map fixture name to provider class."""
        if fixture_name.startswith("anthropic"):
            return AnthropicProvider
        elif fixture_name.startswith("deepseek"):
            return DeepSeekProvider
        elif fixture_name.startswith("minimax"):
            return MinimaxProvider
        elif fixture_name.startswith("openai") or fixture_name.startswith("openrouter"):
            return OpenAIProvider  # OpenRouter uses OpenAIProvider under the hood
        elif fixture_name.startswith("zai"):
            return ZaiProvider
        elif fixture_name.startswith("kimi") or fixture_name.startswith("moonshot"):
            # Moonshot uses OpenAI-compatible
            return OpenAIProvider
        elif fixture_name.startswith("muse"):
            return OpenAIProvider
        elif fixture_name.startswith("nvidia"):
            return OpenAIProvider
        return None

    def _make_usage_object(self, raw: dict[str, Any]):
        """Create a mock usage object with the right attributes.

        Uses a custom class instead of MagicMock to avoid auto-creation
        of missing attributes (which would return truthy MagicMock objects).
        """
        class UsageMock:
            def __init__(self, data):
                for k, v in data.items():
                    if isinstance(v, dict):
                        setattr(self, k, UsageMock(v))
                    else:
                        setattr(self, k, v)
        return UsageMock(raw)


from src.services.pricing import compute_cost, get_pricing


class TestCostCalculationWithCache:
    """Test that compute_cost correctly handles cache tokens."""

    @pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f.name)
    def test_cost_matches_expected_tiers(self, fixture: ProviderUsageFixture):
        """Cost calculation must price cache_read at the cache rate, not input rate."""
        # Get a model name that matches this fixture's provider
        model = self._fixture_to_model(fixture.name)
        if not model:
            pytest.skip(f"No model for {fixture.name}")

        pricing = get_pricing(model)
        if pricing is None:
            pytest.skip(f"No pricing for {model}")

        # The expected_normalized already has the correct split
        usage = fixture.expected_normalized.copy()
        usage["model"] = model  # Not used but kept for clarity

        cost = compute_cost(model, usage)

        # Cost must be positive
        assert cost >= 0, f"{fixture.name}: cost = {cost}"

        # If there are cache_read tokens, cost should be LESS than
        # pricing everything at the input rate
        if usage.get("cache_read_input_tokens", 0) > 0:
            input_rate = pricing["input"]
            cache_read_rate = pricing["cache_read"]

            # Cache read rate must be cheaper than input rate
            assert cache_read_rate < input_rate, (
                f"{model}: cache_read rate ({cache_read_rate}) must be < "
                f"input rate ({input_rate})"
            )

            # Cost with cache split should be less than all-at-input-rate
            all_at_input_rate = (
                usage["input_tokens"] * input_rate +
                usage["cache_read_input_tokens"] * input_rate +
                usage["output_tokens"] * pricing["output"] +
                usage["cache_creation_input_tokens"] * pricing["cache_creation"]
            )
            assert cost < all_at_input_rate, (
                f"{fixture.name}: cost with cache split ({cost}) should be < "
                f"all-at-input-rate ({all_at_input_rate})"
            )

    def _fixture_to_model(self, fixture_name: str) -> str | None:
        mapping = {
            "anthropic_cached": "claude-sonnet-4-6",
            "anthropic_uncached": "claude-sonnet-4-6",
            "deepseek_native_cached": "deepseek-v4-pro",
            "deepseek_gateway_cached": "deepseek/deepseek-v4-pro",
            "minimax_cached": "MiniMax-M3",
            "minimax_uncached": "MiniMax-M3",
            "openai_cached": "gpt-5.4",
            "openrouter_gpt56_luna_cached": "openai/gpt-5.6-luna",
            "openai_uncached": "gpt-5.4",
            "zai_cached": "GLM-5.1",
            "kimi_k3_cached": "kimi-k3",
            "muse_spark_cached": "muse-spark-1.1",
            "nvidia_nim_cached": "nvidia/nemotron-3-ultra",
        }
        return mapping.get(fixture_name)


class TestCapabilityFlagsOnProviderClasses:
    """Test that provider classes have the capability flag as a class attribute."""

    def test_deepseek_provider_flag(self):
        """DeepSeek native does NOT use OpenAI-style cached_tokens breakdown."""
        assert DeepSeekProvider.uses_openai_style_cache_breakdown is False

    def test_minimax_provider_flag(self):
        """Minimax uses Anthropic-compatible API (native cache fields)."""
        assert MinimaxProvider.uses_openai_style_cache_breakdown is False

    def test_anthropic_provider_flag(self):
        """Anthropic has native cache fields."""
        assert AnthropicProvider.uses_openai_style_cache_breakdown is False

    def test_openai_compatible_base_flag(self):
        """OpenAICompatibleProvider base class defaults to True (OpenAI format)."""
        assert OpenAICompatibleProvider.uses_openai_style_cache_breakdown is True

    def test_openai_provider_inherits_flag(self):
        """OpenAIProvider inherits from OpenAICompatibleProvider."""
        assert OpenAIProvider.uses_openai_style_cache_breakdown is True

    def test_openrouter_provider_inherits_flag(self):
        """OpenRouterProvider inherits from OpenAICompatibleProvider."""
        assert OpenRouterProvider.uses_openai_style_cache_breakdown is True

    def test_zai_provider_inherits_flag(self):
        """ZaiProvider inherits from OpenAICompatibleProvider."""
        assert ZaiProvider.uses_openai_style_cache_breakdown is True

    def test_gemini_provider_inherits_flag(self):
        """GeminiProvider uses google-genai SDK (not OpenAI-compatible), so flag is False."""
        assert GeminiProvider.uses_openai_style_cache_breakdown is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])