"""In-process serving cache. Never trains; never silent-fallback."""

from __future__ import annotations

from src.mlops.serving import ServingBundle

_serving: ServingBundle | None = None


def get_cached_bundle() -> ServingBundle | None:
    return _serving


def set_cached_bundle(bundle: ServingBundle | None) -> None:
    global _serving
    _serving = bundle


def clear_cached_bundle() -> None:
    set_cached_bundle(None)
