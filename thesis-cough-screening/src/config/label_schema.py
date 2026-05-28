"""Document and implement the thesis label schema."""

from __future__ import annotations

from typing import Any


TARGET_CLASSES = ("TB", "COVID", "HEALTHY_OR_NONTARGET")
LABEL_TO_INDEX = {label: index for index, label in enumerate(TARGET_CLASSES)}
POSITIVE_DISEASE_LABELS = {"TB", "COVID"}

# Scientific note:
# The first project version merges healthy participants and broader non-target
# controls into a single control class. This is pragmatic for early model
# development, but it should be treated as a mixed-control cohort in the thesis.


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _is_any(value: str, options: set[str]) -> bool:
    return normalize_text(value) in options


COSWARA_COVID_POSITIVE = {
    "positive",
    "positive_mild",
    "positive_moderate",
    "positive_asymp",
    "resp_illness_not_identified",
    "covid_positive",
}
COSWARA_CONTROL = {
    "healthy",
    "no_resp_illness_exposed",
    "recovered_full",
    "negative",
    "covid_negative",
}

COUGHVID_COVID_POSITIVE = {
    "covid-19",
    "covid_19",
    "covid_positive",
    "positive",
}
COUGHVID_CONTROL = {
    "healthy",
    "negative",
    "covid_negative",
}

TB_POSITIVE = {
    "tb",
    "tuberculosis",
    "positive",
    "microbiologically_confirmed_tb",
}
TB_CONTROL = {
    "negative",
    "not_tb",
    "tb_negative",
    "control",
}


def label_binary_from_multiclass(label_multiclass: str | None) -> str | None:
    if not label_multiclass:
        return None
    return "TARGET_DISEASE" if label_multiclass in POSITIVE_DISEASE_LABELS else "CONTROL"


def map_coswara_label(row: dict[str, Any]) -> tuple[str | None, str]:
    raw = row.get("covid_status") or row.get("health_status") or row.get("label_raw")
    normalized = normalize_text(raw)
    if normalized in COSWARA_COVID_POSITIVE:
        return "COVID", "Mapped from documented Coswara COVID-positive status."
    if normalized in COSWARA_CONTROL:
        return "HEALTHY_OR_NONTARGET", (
            "Mapped from Coswara healthy/negative/recovered control label; "
            "this remains a mixed control class."
        )
    return None, "Excluded because the Coswara label could not be mapped cleanly."


def map_coughvid_label(row: dict[str, Any]) -> tuple[str | None, str]:
    raw = (
        row.get("status")
        or row.get("covid_status")
        or row.get("assessment_result")
        or row.get("label_raw")
    )
    normalized = normalize_text(raw)
    if normalized in COUGHVID_COVID_POSITIVE:
        return "COVID", "Mapped from COUGHVID COVID-positive status."
    if normalized in COUGHVID_CONTROL:
        return "HEALTHY_OR_NONTARGET", (
            "Mapped from COUGHVID healthy/negative label; "
            "control examples may include non-target symptomatic cases."
        )
    return None, "Excluded because the COUGHVID label could not be mapped cleanly."


def map_tb_label(row: dict[str, Any]) -> tuple[str | None, str]:
    raw = row.get("tb_status") or row.get("diagnosis") or row.get("label_raw")
    normalized = normalize_text(raw)
    if normalized in TB_POSITIVE:
        return "TB", "Mapped from TB-positive diagnosis."
    if normalized in TB_CONTROL:
        return "HEALTHY_OR_NONTARGET", (
            "Mapped from TB-negative/control diagnosis; "
            "this is a non-target control rather than guaranteed healthy."
        )
    return None, "Excluded because the TB label could not be mapped cleanly."


def map_label(dataset_name: str, row: dict[str, Any]) -> tuple[str | None, str]:
    dataset = normalize_text(dataset_name)
    if dataset == "coswara":
        return map_coswara_label(row)
    if dataset == "coughvid":
        return map_coughvid_label(row)
    if dataset == "tb":
        return map_tb_label(row)
    return None, f"Excluded because dataset '{dataset_name}' is unknown."
