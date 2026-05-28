"""Download open datasets and prepare manual instructions for gated datasets."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from http.client import HTTPResponse
from dataclasses import dataclass
from pathlib import Path

from src.config.paths import DATASET_DIRS, EXTERNAL_DATA_DIR, ensure_project_dirs
from src.data.dataset_registry import DATASET_REGISTRY, DatasetSpec, get_dataset_names
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass
class DownloadResult:
    dataset_name: str
    status: str
    archive_path: str | None = None
    extracted_to: str | None = None
    message: str = ""


def create_manual_download_readme(spec: DatasetSpec, dataset_dir: Path) -> None:
    target = dataset_dir / "README_MANUAL_DOWNLOAD.md"
    lines = [
        f"# Manual Download Required: {spec.name}",
        "",
        f"Dataset: {spec.description}",
        "",
        "This dataset is not downloaded automatically by the project scaffold.",
        "",
        "## Why",
        "",
        "- Access may require account creation, approval, registration, or usage acceptance.",
        "- The pipeline intentionally avoids faking gated downloads.",
        "",
        "## Sources",
        "",
        f"- Primary source: {spec.primary_source}",
    ]
    for source in spec.supporting_sources:
        lines.append(f"- Supporting source: {source}")

    lines.extend(["", "## Required Steps", ""])
    for index, step in enumerate(spec.manual_steps, start=1):
        lines.append(f"{index}. {step}")

    lines.extend(
        [
            "",
            "## Expected Result",
            "",
            f"Place the extracted dataset contents inside `{dataset_dir}`.",
            "The downstream scripts will detect the files automatically if metadata and audio files are present.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        existing_size = destination.stat().st_size
        if existing_size > 0:
            LOGGER.info("Archive already exists, attempting resume-aware download: %s", destination)
        else:
            LOGGER.info("Archive placeholder exists but is empty, restarting download: %s", destination)
    else:
        existing_size = 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "thesis-cough-screening/0.1"}
    write_mode = "wb"
    if existing_size > 0 and not force:
        headers["Range"] = f"bytes={existing_size}-"
        write_mode = "ab"

    request = urllib.request.Request(url, headers=headers)
    LOGGER.info("Downloading %s -> %s", url, destination)
    with contextlib.ExitStack() as stack:
        response = stack.enter_context(urllib.request.urlopen(request))
        response = _validate_resume_response(response, destination, existing_size, force, stack)
        with destination.open(write_mode) as handle:
            shutil.copyfileobj(response, handle)
    return destination


def _validate_resume_response(
    response: HTTPResponse,
    destination: Path,
    existing_size: int,
    force: bool,
    stack: contextlib.ExitStack,
) -> HTTPResponse:
    if existing_size <= 0 or force:
        return response

    if response.status == 206:
        LOGGER.info("Server accepted range request, resuming from byte %s.", existing_size)
        return response

    # If the server ignores the Range header and returns 200, restart cleanly.
    LOGGER.warning(
        "Server did not honor resume request for %s. Restarting full download.",
        destination,
    )
    destination.unlink(missing_ok=True)
    restart_request = urllib.request.Request(
        response.geturl(),
        headers={"User-Agent": "thesis-cough-screening/0.1"},
    )
    return stack.enter_context(urllib.request.urlopen(restart_request))


def extract_archive(archive_path: Path, destination: Path, force: bool = False) -> None:
    if destination.exists() and any(destination.iterdir()) and not force:
        LOGGER.info("Extraction target already populated, skipping: %s", destination)
        return

    destination.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Extracting %s -> %s", archive_path, destination)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(destination)
        return

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(destination)
        return

    raise ValueError(f"Unsupported archive format: {archive_path}")


def dataset_already_present(dataset_dir: Path) -> bool:
    return dataset_dir.exists() and any(dataset_dir.iterdir())


def handle_open_dataset(spec: DatasetSpec, force: bool = False) -> DownloadResult:
    dataset_dir = DATASET_DIRS[spec.name]
    archive_path = EXTERNAL_DATA_DIR / (spec.archive_name or f"{spec.name}.zip")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        if dataset_already_present(dataset_dir) and not force:
            return DownloadResult(
                dataset_name=spec.name,
                status="skipped_existing",
                archive_path=str(archive_path),
                extracted_to=str(dataset_dir),
                message="Raw dataset directory already contains files.",
            )

        downloaded = download_file(spec.archive_url or "", archive_path, force=force)
        extract_archive(downloaded, dataset_dir, force=force)
        return DownloadResult(
            dataset_name=spec.name,
            status="downloaded",
            archive_path=str(downloaded),
            extracted_to=str(dataset_dir),
            message="Archive downloaded and extracted successfully.",
        )
    except urllib.error.HTTPError as error:
        return DownloadResult(
            dataset_name=spec.name,
            status="failed",
            archive_path=str(archive_path),
            extracted_to=str(dataset_dir),
            message=f"HTTP error {error.code}: {error.reason}",
        )
    except urllib.error.URLError as error:
        return DownloadResult(
            dataset_name=spec.name,
            status="failed",
            archive_path=str(archive_path),
            extracted_to=str(dataset_dir),
            message=f"URL error: {error.reason}",
        )
    except Exception as error:  # pragma: no cover - defensive logging path
        return DownloadResult(
            dataset_name=spec.name,
            status="failed",
            archive_path=str(archive_path),
            extracted_to=str(dataset_dir),
            message=str(error),
        )


def handle_manual_dataset(spec: DatasetSpec) -> DownloadResult:
    dataset_dir = DATASET_DIRS[spec.name]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    create_manual_download_readme(spec, dataset_dir)
    return DownloadResult(
        dataset_name=spec.name,
        status="manual_required",
        extracted_to=str(dataset_dir),
        message="Controlled-access dataset; manual download instructions generated.",
    )


def run_downloads(dataset_names: list[str] | None = None, force: bool = False) -> list[DownloadResult]:
    ensure_project_dirs()
    requested = dataset_names or get_dataset_names()
    results: list[DownloadResult] = []

    for name in requested:
        spec = DATASET_REGISTRY[name]
        LOGGER.info("Processing dataset: %s", name)
        if spec.access == "open":
            result = handle_open_dataset(spec, force=force)
        else:
            result = handle_manual_dataset(spec)

        LOGGER.info("%s | %s | %s", result.dataset_name, result.status, result.message)
        results.append(result)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=get_dataset_names(),
        default=None,
        help="Optional subset of datasets to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=os.getenv("THESIS_FORCE_DOWNLOAD", "0") == "1",
        help="Re-download archives and re-extract directories when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_downloads(dataset_names=args.datasets, force=args.force)
    LOGGER.info("Completed dataset preparation for %s dataset(s).", len(results))


if __name__ == "__main__":
    main()
