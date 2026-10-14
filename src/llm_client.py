"""Adaptive LLM Client routing exclusively through Cloudflare Secure Edge Proxy or Local Fallback.

Design Rules:
1. No local API keys: Evaluators never need to configure API keys on their laptop.
2. Cloudflare Edge Proxy: The secret Gemini API key lives securely inside Cloudflare Worker secrets.
3. Offline Grounded Fallback: If running offline or unreachable, falls back to deterministic local synthesis.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model_name: str
    provider: str
    cost_usd: float


class AdaptiveLLMClient:
    """Two-tier client:
    Tier 1: Cloudflare Edge Proxy hosting the secret Gemini API key.
    Tier 2: Local Offline Grounded Synthesizer (100% reproducible, zero-cost fallback).
    """

    # Published rates for Gemini 2.5 Flash / 1.5 Flash: $0.075 / 1M input, $0.30 / 1M output
    # (Free tier gives 1,500 free requests per day, 1M TPM)
    INPUT_RATE_PER_MILLION = 0.075
    OUTPUT_RATE_PER_MILLION = 0.30

    def __init__(self, proxy_url: Optional[str] = None):
        self.proxy_url = proxy_url or os.getenv(
            "FREIGHTTIGER_LLM_PROXY_URL",
            "https://freighttiger-llm-proxy.vatxzz.workers.dev"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using word/subword heuristic (~4 chars per token)."""
        words = len(re.findall(r"\w+|[^\w\s]", text))
        char_estimate = len(text) // 4
        return max(words, char_estimate, 1)

    def _calculate_cost(self, in_tokens: int, out_tokens: int) -> float:
        """Calculate nominal cost in USD based on published Gemini Flash rates."""
        in_cost = (in_tokens / 1_000_000) * self.INPUT_RATE_PER_MILLION
        out_cost = (out_tokens / 1_000_000) * self.OUTPUT_RATE_PER_MILLION
        return round(in_cost + out_cost, 6)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are an expert shipping cost auditor for FreightTiger.",
        deterministic_fallback_text: str = "",
    ) -> LLMResponse:
        """Generate text using Cloudflare Proxy or local offline fallback."""
        # --- Tier 1: Cloudflare Edge Proxy (Gemini) ---
        if self.proxy_url:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "FreightTiger-Assistant/1.0",
                }
                payload = {"prompt": prompt, "system_prompt": system_prompt}
                req = urllib.request.Request(
                    f"{self.proxy_url.rstrip('/')}/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if "output" in data:
                        text = data["output"].strip()
                        in_toks = data.get("input_tokens", self._estimate_tokens(prompt))
                        out_toks = data.get("output_tokens", self._estimate_tokens(text))
                        cost = self._calculate_cost(in_toks, out_toks)
                        return LLMResponse(
                            text=text,
                            input_tokens=in_toks,
                            output_tokens=out_toks,
                            model_name=data.get("model", "gemini-2.5-flash (Edge Relay)"),
                            provider="Cloudflare Edge Proxy (Google Gemini)",
                            cost_usd=cost,
                        )
            except Exception as e:
                logger.warning("Cloudflare Proxy invocation failed: %s; falling back to local synthesis", e)

        # --- Tier 2: Local Offline Grounded Synthesizer ---
        # Deterministic, zero-cost template synthesis without external API dependencies
        in_toks = self._estimate_tokens(system_prompt + " " + prompt)
        out_text = deterministic_fallback_text.strip()
        out_toks = self._estimate_tokens(out_text)

        return LLMResponse(
            text=out_text,
            input_tokens=in_toks,
            output_tokens=out_toks,
            model_name="Offline Grounded Synthesizer (Zero-Cost Template)",
            provider="Deterministic Local Fallback",
            cost_usd=0.0,
        )
