"""
Plugin Pricing Loaders and Remote Pricing Registry Fetcher.

Allows dynamic loading of model pricing from local YAML, remote URLs,
OpenRouter, or custom user-provided providers with disk caching and ETag support.
"""

import os
import json
import time
import logging
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger("llm_context_forge.pricing")


class PricingProvider(ABC):
    """Abstract base class for custom pricing providers."""

    @abstractmethod
    def fetch(self) -> Dict[str, Any]:
        """Fetch model pricing dictionary keyed by model name."""
        pass


class BundledYAMLPricingProvider(PricingProvider):
    """Default pricing provider loading from local YAML data files."""

    def __init__(self, custom_path: Optional[str] = None):
        self.custom_path = custom_path

    def fetch(self) -> Dict[str, Any]:
        path = self.custom_path or os.environ.get("LLM_CONTEXT_FORGE_PRICING_FILE")
        if not path:
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(pkg_dir, "data", "pricing_registry.yaml")

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}


class OpenRouterPricingProvider(PricingProvider):
    """Pricing provider fetching live model pricing from OpenRouter API."""

    OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

    def fetch(self) -> Dict[str, Any]:
        result = {}
        try:
            req = urllib.request.Request(
                self.OPENROUTER_URL,
                headers={"User-Agent": "llm-context-forge/1.0.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    pricing = model.get("pricing", {})
                    prompt_price = float(pricing.get("prompt", 0)) * 1_000_000
                    completion_price = float(pricing.get("completion", 0)) * 1_000_000
                    
                    if model_id and (prompt_price > 0 or completion_price > 0):
                        result[model_id] = {
                            "provider": "openrouter",
                            "input_per_1m": prompt_price,
                            "output_per_1m": completion_price,
                            "context_window": int(model.get("context_length", 4096)),
                            "source_url": self.OPENROUTER_URL,
                            "date_retrieved": time.strftime("%Y-%m-%d"),
                            "confidence": "official",
                            "registry_version": "1.0.0",
                        }
        except Exception as e:
            logger.debug(f"OpenRouter pricing fetch failed: {e}")
        return result


def update_pricing_registry(
    url: str = "https://raw.githubusercontent.com/dhruv-atomic-mui21/llm-context-forge/main/pricing/registry.json",
    cache_ttl: int = 86400,
    offline_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Fetch pricing data from remote URL with local disk cache (~/.llm_context_forge/pricing_cache.json).

    Args:
        url: Remote JSON endpoint URL.
        cache_ttl: Cache expiration time in seconds (default: 24h).
        offline_fallback: Fall back to bundled YAML if fetch fails.

    Returns:
        Pricing registry dictionary.
    """
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".llm_context_forge")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_path = os.path.join(cache_dir, "pricing_cache.json")
    meta_path = os.path.join(cache_dir, "pricing_cache_meta.json")

    # Read cached metadata
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    last_fetched = meta.get("timestamp", 0)
    etag = meta.get("etag")
    now = time.time()

    # If cache is valid and not expired, return cached data
    if (now - last_fetched) < cache_ttl and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                logger.debug(f"Using local pricing cache from {cache_path}")
                return json.load(f)
        except Exception:
            pass

    # Fetch from remote URL
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-context-forge/1.0.0"})
        if etag:
            req.add_header("If-None-Match", etag)

        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8")
            data = json.loads(content)
            new_etag = resp.headers.get("ETag")

            # Cache response
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": now, "etag": new_etag}, f)

            logger.debug(f"Successfully fetched remote pricing registry from {url}")
            return data
    except urllib.error.HTTPError as e:
        if e.code == 304 and os.path.exists(cache_path):
            logger.debug("Remote pricing registry unchanged (304 Not Modified)")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.debug(f"HTTP error fetching remote pricing: {e}")
    except Exception as e:
        logger.debug(f"Failed to fetch remote pricing registry from {url}: {e}")

    if offline_fallback:
        logger.debug("Falling back to bundled local YAML pricing registry")
        return BundledYAMLPricingProvider().fetch()

    return {}
