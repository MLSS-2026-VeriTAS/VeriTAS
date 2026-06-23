#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MLRC_DIR="${1:-${REPO_ROOT}/../MLRC-Bench}"
PATCH_FILE="${REPO_ROOT}/patches/mlrc/gpt_responses_api.patch"

if [[ ! -d "${MLRC_DIR}" ]]; then
  echo "MLRC-Bench directory not found: ${MLRC_DIR}" >&2
  echo "Usage: bash scripts/apply_mlrc_patch.sh /path/to/MLRC-Bench" >&2
  exit 2
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Patch file not found: ${PATCH_FILE}" >&2
  exit 2
fi

for required in \
  "MLAgentBench/LLM.py" \
  "MLAgentBench/runner.py" \
  "MLAgentBench/high_level_actions.py" \
  "MLAgentBench/utils.py" \
  "MLAgentBench/LLM_as_a_Judge.py" \
  "MLAgentBench/llm_test_cases.py" \
  "launch.sh"; do
  if [[ ! -f "${MLRC_DIR}/${required}" ]]; then
    echo "Expected MLRC file is missing: ${MLRC_DIR}/${required}" >&2
    exit 2
  fi
done

cd "${MLRC_DIR}"

if patch -p1 --dry-run -R < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "MLRC GPT Responses API patch is already applied."
  exit 0
fi

echo "Checking MLRC GPT Responses API patch..."
patch -p1 --dry-run < "${PATCH_FILE}" >/dev/null

echo "Applying MLRC GPT Responses API patch..."
patch -p1 < "${PATCH_FILE}"

echo "Patch applied. Current MLRC GPT defaults:"
grep -R "gpt-5.4-mini" \
  MLAgentBench/LLM.py \
  MLAgentBench/runner.py \
  MLAgentBench/high_level_actions.py \
  MLAgentBench/utils.py \
  MLAgentBench/LLM_as_a_Judge.py \
  MLAgentBench/llm_test_cases.py \
  | head -20
