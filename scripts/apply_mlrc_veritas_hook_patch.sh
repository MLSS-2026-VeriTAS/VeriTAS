#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MLRC_DIR="${1:-${REPO_ROOT}/../MLRC-Bench}"

python3 "${REPO_ROOT}/scripts/apply_mlrc_veritas_hook_patch.py" "${MLRC_DIR}"

python3 -m py_compile "${MLRC_DIR}/MLAgentBench/low_level_actions.py"

echo "VeriTAS hook patch verification passed."
