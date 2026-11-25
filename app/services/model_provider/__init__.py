"""Model provider registry.

Providers are keyed by ``name``. Adding a vendor is one module implementing
``ModelProvider`` + one branch in ``resolve_provider``; callers only use
``resolve_provider`` and the shapes in ``models``.

``resolve_provider`` decides where each provider is reached, when it is
asked: typesafe and openrouter at their URL and key from the environment,
custom (``system_one``: any System One endpoint but TypeSafe's) at the
endpoint the configuration names. Each is built the first time it is asked
for and then kept, one per (provider, URL, key), so each endpoint keeps one
pooled client.
"""

from typing import Any, Callable, Dict, Mapping, Tuple

from app.core.config.static import (
    CUSTOM_SYSTEM_ONE_TIMEOUT_SECONDS,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OPENROUTER_TIMEOUT_SECONDS,
    TYPESAFE_API_KEY,
    TYPESAFE_API_URL,
    TYPESAFE_TIMEOUT_SECONDS,
)
from app.core.logger import logger
from app.services.live_config.store import get_config
from app.services.model_provider.base import ModelProvider
from app.services.model_provider.models import (
    GenerateRequest,
    GenerateResponse,
    GenerationSettings,
    Input,
    JsonSchema,
    Message,
    ProviderConfig,
    ProviderError,
    Usage,
)
from app.services.model_provider.openrouter import OpenRouterProvider
from app.services.model_provider.system_one import (
    CustomSystemOneProvider,
    TypeSafeProvider,
)

# every provider built so far, one per (provider, URL, key): each keeps one
# pooled client, reused by every call to that endpoint; close_all closes them
_PROVIDERS: Dict[Tuple[str, str, str], ModelProvider] = {}


def _provider(
    kind: Callable[[ProviderConfig], ModelProvider],
    name: str,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
) -> ModelProvider:
    """The provider ``name`` at this URL with this key: the one built
    before, else a new one."""
    provider = _PROVIDERS.get((name, base_url, api_key))
    if provider is None:
        provider = kind(
            ProviderConfig(
                api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds
            )
        )
        _PROVIDERS[(name, base_url, api_key)] = provider
    return provider


async def _config_value(config_key: str, what: str) -> str:
    """A custom endpoint's URL or key: the config key's value (Redis, then
    the environment), read on every use, so it changes without a deploy."""
    value = await get_config(config_key, "", str)
    if not value:
        raise ProviderError(
            CustomSystemOneProvider.name,
            f"{what} not found for config key: {config_key}",
        )
    return value


async def resolve_provider(configuration: Mapping[str, Any]) -> ModelProvider:
    """The provider a configuration names (``provider``), where it says.
    An eval's configuration was validated when saved (evals/definition.py).
    Raises ``ProviderError``."""
    name = str(configuration.get("provider"))
    if name == TypeSafeProvider.name:
        # TypeSafe's URL and key, from the environment
        return _provider(
            TypeSafeProvider,
            name,
            TYPESAFE_API_URL,
            TYPESAFE_API_KEY,
            TYPESAFE_TIMEOUT_SECONDS,
        )
    if name == OpenRouterProvider.name:
        # OpenRouter's URL and key, from the environment
        return _provider(
            OpenRouterProvider,
            name,
            OPENROUTER_API_URL,
            OPENROUTER_API_KEY,
            OPENROUTER_TIMEOUT_SECONDS,
        )
    if name == CustomSystemOneProvider.name:
        # the URL and key held by the config keys the configuration names
        url_name = configuration.get("api_url_name")
        key_name = configuration.get("api_key_name")
        if not url_name or not key_name:
            raise ProviderError(name, "api_url_name and api_key_name are required")
        return _provider(
            CustomSystemOneProvider,
            name,
            await _config_value(url_name, "API URL"),
            await _config_value(key_name, "API key"),
            CUSTOM_SYSTEM_ONE_TIMEOUT_SECONDS,
        )
    raise ProviderError(name, "unknown provider")


async def close_all() -> None:
    """Close every provider's pooled client (process shutdown). One failing
    close never stops the others, nor the rest of the shutdown."""
    for provider in list(_PROVIDERS.values()):
        try:
            await provider.close()
        except Exception as exc:
            logger.warning(f"closing model provider {provider.name!r} failed: {exc}")


__all__ = [
    "CustomSystemOneProvider",
    "GenerateRequest",
    "GenerateResponse",
    "GenerationSettings",
    "Input",
    "JsonSchema",
    "Message",
    "ModelProvider",
    "OpenRouterProvider",
    "ProviderConfig",
    "ProviderError",
    "TypeSafeProvider",
    "Usage",
    "close_all",
    "resolve_provider",
]
