"""Helpers for Anthropic prompt caching.

Wraps a string into a `SystemMessage` whose content is a single text block
marked with `cache_control: {type: ephemeral}`, which tells the Anthropic API
to cache the prefix for ~5 minutes. Since every agent sends the same system
prompt on every call within a revision loop, this cuts input-token cost on
cache hits by ~90%."""
from __future__ import annotations

import os

from langchain_core.messages import SystemMessage


def _cache_enabled() -> bool:
    return os.getenv("ENABLE_PROMPT_CACHE", "true").lower() not in {"0", "false", "no"}


def cached_system(prompt: str) -> SystemMessage:
    """Build a cache-marked `SystemMessage` — falls back to plain text when disabled."""
    if not _cache_enabled():
        return SystemMessage(content=prompt)

    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )
