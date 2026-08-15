"""Unified model profile construction across benchmark modules."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from unified_profile.models import model_label


@dataclass
class UnifiedModelProfile:
    model_id: str
    label: str
    source_model_keys: dict[str, str]
    safety_score: float | None
    moral_score: float | None
    epistemic_score: float | None
    composite_score: float | None
    failure_class: str | None
    failure_mode_distribution: dict | None
    classifier_judge_conflicts: int | None
    sus_raw: dict | None
    aita_raw: dict | None
    epis_raw: dict | None
    sus_n: int
    aita_n: int
    epis_n: int


def _dominant_class(sus: dict | None) -> str | None:
    if not sus:
        return None
    distribution = sus.get("raw", {}).get("failure_class_distribution")
    if not distribution:
        return None
    return max(distribution.items(), key=lambda item: item[1])[0]


def build_profile(model_id: str, sus: dict | None, aita: dict | None, epis: dict | None) -> UnifiedModelProfile:
    """Build one unified profile from per-module adapter outputs."""
    scores = [
        item["sycophancy_score"]
        for item in (sus, aita, epis)
        if item and item.get("sycophancy_score") is not None
    ]
    source_model_keys = {}
    if sus:
        source_model_keys["sus"] = sus["source_model_key"]
    if aita:
        source_model_keys["aita"] = aita["source_model_key"]
    if epis:
        source_model_keys["epistemic"] = epis["source_model_key"]

    label = (
        (sus or {}).get("label")
        or (aita or {}).get("label")
        or (epis or {}).get("label")
        or model_label(model_id)
    )

    return UnifiedModelProfile(
        model_id=model_id,
        label=label,
        source_model_keys=source_model_keys,
        safety_score=sus.get("sycophancy_score") if sus else None,
        moral_score=aita.get("sycophancy_score") if aita else None,
        epistemic_score=epis.get("sycophancy_score") if epis else None,
        composite_score=round(mean(scores), 1) if scores else None,
        failure_class=_dominant_class(sus),
        failure_mode_distribution=sus.get("raw", {}).get("failure_mode_distribution") if sus else None,
        classifier_judge_conflicts=sus.get("raw", {}).get("classifier_judge_conflicts") if sus else None,
        sus_raw=sus.get("raw") if sus else None,
        aita_raw=aita.get("raw") if aita else None,
        epis_raw=epis.get("raw") if epis else None,
        sus_n=sus.get("n_items", 0) if sus else 0,
        aita_n=aita.get("n_items", 0) if aita else 0,
        epis_n=epis.get("n_items", 0) if epis else 0,
    )


def build_all_profiles(sus_data: dict, aita_data: dict, epis_data: dict) -> list[UnifiedModelProfile]:
    """Build profiles for the union of canonical model IDs, sorted worst first."""
    model_ids = set(sus_data) | set(aita_data) | set(epis_data)
    profiles = [
        build_profile(model_id, sus_data.get(model_id), aita_data.get(model_id), epis_data.get(model_id))
        for model_id in sorted(model_ids)
    ]
    return sorted(profiles, key=lambda p: (p.composite_score is not None, p.composite_score or -1), reverse=True)
