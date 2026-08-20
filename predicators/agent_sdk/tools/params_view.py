"""Read-only view of fitted simulator parameters for synthesis tools."""
from typing import Any, Dict

# ── Predicate-invention tools ─────────────────────────────────────


class _ParamsView:
    """Read-through view onto a fitted-parameters dict.

    Holds the dict directly (not the approach) so predicate classifiers
    that close over this view do not transitively reference the
    approach. The approach must mutate the same dict object in place on
    each re-fit (clear + update) so the view picks up new values
    automatically; replacing the dict would break the live link.
    """

    def __init__(self, params: Dict[str, float]) -> None:
        self._params = params

    def __getitem__(self, key: str) -> float:
        if key not in self._params:
            raise KeyError(
                f"params[{key!r}] accessed before any parameter fit; "
                "call sim.fit() to "
                "populate self._fitted_params first.")
        return self._params[key]

    def __contains__(self, key: object) -> bool:
        return key in self._params

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style fallback lookup; mirrors ``dict.get``."""
        return self._params.get(key, default)

    def __repr__(self) -> str:
        return f"_ParamsView({self._params!r})"
