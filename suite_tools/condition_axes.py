"""Display/grouping axes for tested-system conditions (spec 015 §3.1).

Comparability stays at full-condition granularity (condition hashes); these
axes let registries and reports group the same underlying model across routes
(OpenRouter vs direct), split by reasoning effort, and track system-under-test
profiles (prompt bundles) with lineage. Never used inside identity hashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from suite_tools.model_config import load_suite_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE_MODELS = REPO_ROOT / "suite_models.yaml"

_PROFILE_FIELDS = ("profile_id", "profile_hash", "parent_profile_id")


def load_model_aliases(path: Path | str | None = None) -> dict[str, str]:
    """Read the ``model_aliases`` map (route-specific id -> canonical slug).

    Reuses ``load_suite_config`` — the existing YAML loader for
    ``suite_models.yaml`` — to avoid re-reading the file with a separate
    ``yaml.safe_load`` call.
    """
    yaml_path = Path(path) if path is not None else DEFAULT_SUITE_MODELS
    try:
        data = load_suite_config(yaml_path)
    except (OSError, ValueError):
        return {}
    aliases = data.get("model_aliases") or {}
    if not isinstance(aliases, dict):
        return {}
    return {str(key): str(value) for key, value in aliases.items()}


def condition_axes(
    condition: dict[str, Any],
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return grouping axes for one model condition.

    Args:
        condition: A rendered condition dict (as stored in run contracts /
            identity fixtures).
        aliases: Optional override alias map (route-specific model_id ->
            canonical slug).  Pass ``{}`` to disable alias resolution.
            When ``None``, aliases are loaded from the default
            ``suite_models.yaml``.

    Returns:
        Dict with keys ``canonical_model``, ``route``, ``effort``, and
        ``profile`` (``None`` or a dict with ``profile_id``,
        ``profile_hash``, ``parent_profile_id``).
    """
    alias_map = aliases if aliases is not None else load_model_aliases()
    model_id = str(condition.get("model_id") or "")
    metadata = condition.get("condition_metadata") or {}
    profile = None
    if condition.get("profile_id") is not None:
        profile = {field: condition.get(field) for field in _PROFILE_FIELDS}
    return {
        "canonical_model": alias_map.get(model_id, model_id),
        "route": condition.get("endpoint") or metadata.get("provider_route"),
        "effort": metadata.get("effort"),
        "profile": profile,
    }
