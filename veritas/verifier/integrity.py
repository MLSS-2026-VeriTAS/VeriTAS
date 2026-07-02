"""Protected-file digest helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from veritas.verifier.io import read_json, to_jsonable, write_json
from veritas.verifier.types import FileDigest, ProtectedDigestManifest


DIGEST_ALGORITHM = "sha256"

CATEGORY_METRIC_CODE = "metric_code"
CATEGORY_DATA_SPLIT = "data_split"
CATEGORY_PROTECTED_FILE = "protected_file"

FLAG_BY_CATEGORY = {
    CATEGORY_METRIC_CODE: "metric_code_modified",
    CATEGORY_DATA_SPLIT: "data_split_modified",
    CATEGORY_PROTECTED_FILE: "protected_file_modified",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_protected_digest_manifest(
    *,
    run_id: str,
    task_root: str | Path,
    protected_paths: Iterable[str] = (),
    metric_paths: Iterable[str] = (),
    split_paths: Iterable[str] = (),
    audit_data_paths: Iterable[str] = (),
    read_only_file: str | Path | None = None,
    optional_paths: Iterable[str] = (),
    created_at: str | None = None,
) -> ProtectedDigestManifest:
    task_root_path = Path(task_root)
    optional = {_normalize_relative(path) for path in optional_paths}

    path_categories: dict[str, str] = {}
    protected_sources: list[str] = []

    _add_paths(path_categories, protected_paths, CATEGORY_PROTECTED_FILE)
    if protected_paths:
        protected_sources.append("task_manifest.protected_paths")

    read_only_paths = _read_read_only_paths(read_only_file)
    _add_paths(path_categories, read_only_paths, CATEGORY_PROTECTED_FILE)
    if read_only_file is not None:
        protected_sources.append(str(read_only_file))

    _add_paths(path_categories, split_paths, CATEGORY_DATA_SPLIT)
    _add_paths(path_categories, audit_data_paths, CATEGORY_DATA_SPLIT)
    if split_paths:
        protected_sources.append("task_adapter.split_paths")
    if audit_data_paths:
        protected_sources.append("task_adapter.audit_data_paths")

    _add_paths(path_categories, metric_paths, CATEGORY_METRIC_CODE)
    if metric_paths:
        protected_sources.append("task_adapter.metric_paths")

    files_by_path: dict[str, FileDigest] = {}
    missing_files: list[str] = []
    for relative_path, category in sorted(path_categories.items()):
        absolute_path = task_root_path / relative_path
        if not absolute_path.exists():
            if relative_path not in optional:
                missing_files.append(relative_path)
            continue

        if absolute_path.is_dir():
            for file_path in sorted(path for path in absolute_path.rglob("*") if path.is_file()):
                relative_file = file_path.relative_to(task_root_path).as_posix()
                effective_category = path_categories.get(relative_file, category)
                _upsert_file_digest(
                    files_by_path,
                    path=relative_file,
                    category=effective_category,
                    sha256=sha256_file(file_path),
                )
        elif absolute_path.is_file():
            _upsert_file_digest(
                files_by_path,
                path=relative_path,
                category=category,
                sha256=sha256_file(absolute_path),
            )

    return ProtectedDigestManifest(
        run_id=run_id,
        created_at=created_at or _utc_now(),
        digest_algorithm=DIGEST_ALGORITHM,
        protected_sources=protected_sources,
        files=[files_by_path[path] for path in sorted(files_by_path)],
        missing_files=missing_files,
    )


def write_protected_digest_manifest(
    path: str | Path,
    manifest: ProtectedDigestManifest,
) -> None:
    write_json(path, manifest)


def load_protected_digest_manifest(path: str | Path) -> ProtectedDigestManifest:
    data = read_json(path)
    return ProtectedDigestManifest(
        run_id=data["run_id"],
        created_at=data["created_at"],
        digest_algorithm=data["digest_algorithm"],
        protected_sources=list(data["protected_sources"]),
        files=[
            FileDigest(
                path=item["path"],
                category=item["category"],
                sha256=item["sha256"],
            )
            for item in data["files"]
        ],
        missing_files=list(data["missing_files"]),
    )


def compare_digest_manifests(
    before: ProtectedDigestManifest,
    after: ProtectedDigestManifest,
) -> list[str]:
    flags: set[str] = set()

    if before.digest_algorithm != DIGEST_ALGORITHM or after.digest_algorithm != DIGEST_ALGORITHM:
        flags.add("integrity_not_enforced")

    if before.missing_files:
        flags.add("integrity_not_enforced")

    before_by_path = {item.path: item for item in before.files}
    after_by_path = {item.path: item for item in after.files}

    for path, before_digest in before_by_path.items():
        after_digest = after_by_path.get(path)
        if after_digest is None or after_digest.sha256 != before_digest.sha256:
            flags.add(_flag_for_category(before_digest.category))

    for path, after_digest in after_by_path.items():
        if path not in before_by_path:
            flags.add(_flag_for_category(after_digest.category))

    for missing_path in after.missing_files:
        before_digest = before_by_path.get(missing_path)
        if before_digest is None:
            flags.add("integrity_not_enforced")
        else:
            flags.add(_flag_for_category(before_digest.category))

    return sorted(flags)


def manifest_to_dict(manifest: ProtectedDigestManifest) -> dict:
    return to_jsonable(manifest)


def _add_paths(
    path_categories: dict[str, str],
    paths: Iterable[str],
    category: str,
) -> None:
    for path in paths:
        path_categories[_normalize_relative(path)] = category


def _read_read_only_paths(read_only_file: str | Path | None) -> list[str]:
    if read_only_file is None:
        return []
    file_path = Path(read_only_file)
    if not file_path.exists():
        return []
    paths: list[str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            paths.append(clean)
    return paths


def _normalize_relative(path: str) -> str:
    return Path(path).as_posix().lstrip("./")


def _flag_for_category(category: str) -> str:
    return FLAG_BY_CATEGORY.get(category, "protected_file_modified")


def _upsert_file_digest(
    files_by_path: dict[str, FileDigest],
    *,
    path: str,
    category: str,
    sha256: str,
) -> None:
    existing = files_by_path.get(path)
    if existing is None or _category_priority(category) >= _category_priority(existing.category):
        files_by_path[path] = FileDigest(path=path, category=category, sha256=sha256)


def _category_priority(category: str) -> int:
    if category == CATEGORY_METRIC_CODE:
        return 3
    if category == CATEGORY_DATA_SPLIT:
        return 2
    return 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
