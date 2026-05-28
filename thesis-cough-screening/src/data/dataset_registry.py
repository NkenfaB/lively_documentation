"""Registry describing supported datasets and acquisition rules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    access: str
    archive_url: str | None = None
    archive_name: str | None = None
    primary_source: str | None = None
    supporting_sources: tuple[str, ...] = ()
    preferred_recording_types: tuple[str, ...] = ("cough",)
    notes: tuple[str, ...] = ()
    manual_steps: tuple[str, ...] = field(default_factory=tuple)


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "coswara": DatasetSpec(
        name="coswara",
        description="COVID-era respiratory audio dataset with cough and metadata.",
        access="open",
        archive_url=(
            "https://zenodo.org/api/records/7188627/files/"
            "iiscleap/Coswara-Data-dataset-paper-publication.zip/content"
        ),
        archive_name="coswara_publication.zip",
        primary_source="https://zenodo.org/records/7188627",
        supporting_sources=(
            "https://coswara.iisc.ac.in/about",
            "https://www.nature.com/articles/s41597-023-02266-0",
        ),
        preferred_recording_types=("cough", "cough-heavy", "cough-shallow"),
        notes=(
            "Use cough-focused subsets first.",
            "Map healthy and negative participants carefully into the control class.",
        ),
    ),
    "coughvid": DatasetSpec(
        name="coughvid",
        description="Large public cough corpus with COVID-era metadata.",
        access="open",
        archive_url="https://zenodo.org/api/records/4498364/files/public_dataset.zip/content",
        archive_name="coughvid_public.zip",
        primary_source="https://www.epfl.ch/labs/esl/index-html/datasets/coughviddataset/",
        supporting_sources=(
            "https://zenodo.org/records/4498364",
            "https://github.com/esl-epfl/COUGHVID",
        ),
        preferred_recording_types=("cough",),
        notes=(
            "Prefer cough-only records.",
            "Use documented metadata and any expert-assessed fields when available.",
        ),
    ),
    "tb": DatasetSpec(
        name="tb",
        description="Controlled-access TB cough dataset.",
        access="manual",
        primary_source="https://tbdata.ucsf.edu/s/rdc-dataset/a0U5w00000fTCKiEAO/ds000731",
        supporting_sources=(
            "https://www.nature.com/articles/s41597-024-03972-z",
            "https://www.synapse.org/Synapse:syn31472953",
        ),
        preferred_recording_types=("cough",),
        notes=(
            "Do not fake or bypass controlled access.",
            "Create the folder and manual instructions only.",
        ),
        manual_steps=(
            "Open the UCSF dataset page and review the dataset access conditions.",
            "Create or sign in to a Synapse account if the released files are routed through Synapse.",
            "Request access or accept the required usage terms if prompted.",
            "Download the approved TB cough audio and accompanying metadata manually.",
            "Place the extracted files into data/raw/tb/ while preserving the original metadata files.",
            "Re-run the audit and metadata unification scripts after the files are present.",
        ),
    ),
}


def get_dataset_names() -> list[str]:
    return sorted(DATASET_REGISTRY)
