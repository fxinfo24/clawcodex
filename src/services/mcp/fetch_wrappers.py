"""Fetch wrappers: httpx client factory with MCP-appropriate timeouts.

Phase 2 WI-2.4 (FU#6). Mirrors TS' ``wrapFetchWithTimeout`` from
typescript/src/services/mcp/auth.ts. The default ``httpx.AsyncClient``
timeout is 5 seconds — far too short for slow MCP servers, long-running
tool calls, and corporate networks with deep TLS chains. This module
exposes a single factory ``build_mcp_http_client(headers=...)`` that
returns a client with TS-canonical timeouts:

* ``connect=15s``  — long enough for slow TLS handshakes; short enough
  that an unreachable host fails fast (matches the cold-OAuth-discovery
  budget AUTH_REQUEST_TIMEOUT_MS=30000 minus headroom).
* ``read=300s``    — five-minute read budget, matching the per-tool
  timeout ``DEFAULT_MCP_TOOL_TIMEOUT_MS`` so a long-running MCP tool
  call doesn't get killed by the transport. Per-tool timeouts override.
* ``write=30s``    — generous write budget for large request payloads.
* ``pool=10s``     — short pool-acquire budget; connection-pool
  starvation should fail loud rather than wait minutes.

Why a factory: the SDK's ``streamable_http_client`` takes an optional
``http_client`` parameter and adopts caller-provided clients without
closing them. We hand it a pre-configured client built here so the
timeout policy is uniform across HttpTransport, SseTransport, and
ad-hoc fetches (claudeai loader, OAuth discovery — though those have
their own narrower timeouts where appropriate).
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# Per-segment timeouts. Tuned for MCP workloads: long-running tool
# calls + cold TLS handshakes + corporate proxies. Operators can
# override via the MCP_*_TIMEOUT_S env vars below.
DEFAULT_CONNECT_TIMEOUT_S: float = 15.0
DEFAULT_READ_TIMEOUT_S: float = 300.0
DEFAULT_WRITE_TIMEOUT_S: float = 30.0
DEFAULT_POOL_TIMEOUT_S: float = 10.0

# httpx's default User-Agent is ``python-httpx/x.y``, which Cloudflare rejects
# with Error 1010 (``browser_signature_banned``) on any site running browser
# integrity checks. That silently broke every Streamable-HTTP / SSE MCP server
# behind such a WAF — the handshake 403s, so no tools are ever listed and the
# server is absent from server-sessions.json, with no error surfaced to the
# user. Send a browser UA by default; caller-supplied headers still win.
# Same reasoning as ``tool_system/tools/web_fetch.py``'s ``_BROWSER_UA``.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _env_float(name: str, default: float) -> float:
    """Read a positive-float env-var override; fall back to ``default``.

    Invalid / non-positive values are logged and ignored.
    """
    import os

    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    logger.warning(
        "%s=%r is not a positive float; using default %s",
        name, raw, default,
    )
    return default


def build_mcp_timeout() -> httpx.Timeout:
    """Construct the httpx ``Timeout`` used by every MCP transport."""
    return httpx.Timeout(
        connect=_env_float("MCP_CONNECT_TIMEOUT_S", DEFAULT_CONNECT_TIMEOUT_S),
        read=_env_float("MCP_READ_TIMEOUT_S", DEFAULT_READ_TIMEOUT_S),
        write=_env_float("MCP_WRITE_TIMEOUT_S", DEFAULT_WRITE_TIMEOUT_S),
        pool=_env_float("MCP_POOL_TIMEOUT_S", DEFAULT_POOL_TIMEOUT_S),
    )


def build_mcp_http_client(
    *,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """Return an ``httpx.AsyncClient`` with MCP-appropriate timeouts.

    The SDK's ``streamable_http_client`` adopts caller-provided clients
    without closing them, so the caller is responsible for the client's
    lifecycle (the transport adapters register ``aclose()`` on their
    exit stack). Headers when provided are baked into the client so all
    requests carry them — matches the SDK's expectation for header
    propagation on Streamable HTTP and SSE.

    A browser ``User-Agent`` is always set (see ``_BROWSER_UA``) because
    httpx's default trips Cloudflare bot protection. A caller-supplied
    ``User-Agent`` overrides the default.
    """
    return httpx.AsyncClient(
        timeout=build_mcp_timeout(),
        headers={"User-Agent": _BROWSER_UA, **(headers or {})},
    )
