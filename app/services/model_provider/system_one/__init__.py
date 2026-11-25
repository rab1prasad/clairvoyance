"""The System One providers: judge models behind System One's task-specific
API (a structured payload of ``state`` and ``questions`` in, ``answers``
out), wherever it is served.

  typesafe.py  TypeSafe's own System One (TYPESAFE_API_URL /
               TYPESAFE_API_KEY); provider ``typesafe``
  custom.py    any other System One endpoint (Grid's, say), at the URL and
               with the key that config keys named by the caller hold;
               provider ``custom``
"""

from app.services.model_provider.system_one.custom import CustomSystemOneProvider
from app.services.model_provider.system_one.typesafe import TypeSafeProvider

__all__ = ["CustomSystemOneProvider", "TypeSafeProvider"]
