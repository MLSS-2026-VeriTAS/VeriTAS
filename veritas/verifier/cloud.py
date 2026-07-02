"""Cloud run metadata helpers.

These helpers intentionally avoid Google Cloud API dependencies. They capture
the run metadata that should be written into verifier artifacts by the runner.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping

from veritas.verifier.types import CloudRunManifest


SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "PRIVATE_KEY",
)

DEFAULT_ENV_KEYS = (
    "GCP_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "CLOUDSDK_COMPUTE_ZONE",
    "GCE_INSTANCE_NAME",
    "TASK_NAME",
    "MODEL",
    "GPU_ID",
    "CONDA_DEFAULT_ENV",
    "MY_OPENAI_API_KEY",
    "MY_AZURE_OPENAI_ENDPOINT",
)


def redact_environment(
    env: Mapping[str, str],
    keys: tuple[str, ...] = DEFAULT_ENV_KEYS,
) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key in keys:
        if key not in env:
            continue
        if any(marker in key.upper() for marker in SENSITIVE_ENV_MARKERS):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = env[key]
    return redacted


def build_cloud_run_manifest(
    run_id: str,
    command: str,
    env: Mapping[str, str] | None = None,
    *,
    git_commit: str | None = None,
    mlrc_commit: str | None = None,
    durable_artifact_uri: str | None = None,
    machine_type: str | None = None,
    accelerator_type: str | None = None,
    accelerator_count: int | None = None,
    disk_image: str | None = None,
    cuda_version: str | None = None,
    driver_version: str | None = None,
) -> CloudRunManifest:
    env = os.environ if env is None else env
    return CloudRunManifest(
        run_id=run_id,
        command=command,
        gcp_project_id=env.get("GCP_PROJECT_ID") or env.get("GOOGLE_CLOUD_PROJECT"),
        zone=env.get("CLOUDSDK_COMPUTE_ZONE"),
        instance_name=env.get("GCE_INSTANCE_NAME"),
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        disk_image=disk_image,
        conda_env=env.get("CONDA_DEFAULT_ENV"),
        python_version=platform.python_version(),
        cuda_version=cuda_version,
        driver_version=driver_version,
        git_commit=git_commit,
        mlrc_commit=mlrc_commit,
        durable_artifact_uri=durable_artifact_uri,
        redacted_environment=redact_environment(env),
    )

