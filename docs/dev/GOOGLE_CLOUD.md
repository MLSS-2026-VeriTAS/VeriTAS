# Google Cloud Runbook

This project uses Google Cloud for hackathon compute, including dev/test runs
and final results. Treat cloud execution as part of the experiment protocol:
record the environment, preserve artifacts, and avoid leaking secrets.

## Access

1. Ask the project owner for access to the Google Cloud project.
2. Use the smallest IAM role that lets you run the assigned VM or job.
3. Do not commit service account keys, API keys, Kaggle credentials, or hidden
   audit labels.
4. Prefer project-managed identities over downloaded service account keys when
   possible.

## Connect To The VM

Install `gcloud` locally, then connect:

```bash
gcloud compute ssh \
  --zone "us-central1-b" \
  "instance-20260622-053006" \
  --project "veritas-500122"
```

Use `tmux` for long runs:

```bash
tmux a -t 0
```

## Environment Setup

Activate the task environment and set run variables:

```bash
conda activate product-recommendation

export GCP_PROJECT_ID="veritas-500122"
export TASK_NAME="product-recommendation"
export MODEL="gemini-1.5-flash-002"
export GPU_ID="0"
```

Initialize and launch MLRC-Bench:

```bash
bash scripts/init_env.sh "${TASK_NAME}" "${MODEL}" "${GPU_ID}" "TEST_MODEL"
bash launch.sh "${TASK_NAME}" "${MODEL}" "${GPU_ID}"
```

## Required Cloud Run Metadata

Every verifier run should write a `cloud_run_manifest.json` beside the run
ledger. It should include:

- `gcp_project_id`
- zone
- instance name
- machine type
- accelerator type/count
- disk image or VM image identifier
- conda environment
- Python version
- CUDA and driver versions when available
- VeriTAS git commit
- MLRC-Bench git commit
- exact command
- durable artifact URI
- redacted environment variables

The helper `veritas.verifier.cloud.build_cloud_run_manifest` creates the
in-process portion of this manifest without requiring Google Cloud SDK imports.

## Artifact Durability

Do not leave final evidence only on the VM disk or in `tmux` scrollback.

After each run, copy these artifacts to durable storage, preferably a GCS bucket:

- `runs.jsonl`
- `candidates.jsonl`
- `verifier_report.json`
- `scheduler_state.json`
- `cloud_run_manifest.json`
- protected digest manifests
- unit outputs
- candidate snapshots
- MLRC `env_log`
- final report

Record the durable URI in the run manifest.

## Cost Controls

Before expensive runs:

- agree on max dev runs, max reruns, max audit looks, and max wall-clock time;
- configure budget alerts for the project or billing account;
- stop idle VMs;
- record runtime and model/API cost when available.

Budget alerts are notifications, not a hard spending cap. Do not rely on alerts
alone to stop runaway jobs.

## Reproducibility Rules

- Use a fresh run workdir for each experiment.
- Hash protected files before and after each candidate run.
- Record the exact command and seed.
- Record package state, at minimum with a `pip freeze` artifact.
- Do not reuse partial artifacts from interrupted runs.
- Mark interrupted or failed commands as failed ledger rows.

## Audit/Test Secrecy

Final/audit labels and verifier-only unit scores must not be visible to the LLM
coding agent before a candidate is frozen for audit.

If a cloud workdir contains hidden labels or verifier-only scores, restrict read
access and avoid copying those files into agent-visible prompts, logs, or
workspace paths.
