"""The custom provider: System One served somewhere other than TypeSafe's own
endpoint (Grid's, say).

It speaks TypeSafe's API (typesafe.py) and is named apart only so its errors
say which provider failed. It has no URL or key of its own: the registry
reaches it at the endpoint a configuration names (model_provider's
``resolve_provider``).
"""

from app.services.model_provider.system_one.typesafe import TypeSafeProvider


class CustomSystemOneProvider(TypeSafeProvider):
    name = "custom"
