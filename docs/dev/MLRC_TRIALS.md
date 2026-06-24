# MLRC Trial Runbook

VeriTAS should sit beside MLRC-Bench, not contain it. Treat MLRC-Bench as an
external benchmark checkout that emits artifacts, and treat VeriTAS as the
verifier/controller layer that consumes those artifacts.

## VM Layout

```text
~/work/
  VeriTAS/
  MLRC-Bench/
  trials/
    product-recommendation/
      run-001/
        preds/
        ledger/
```

## Apply The MLRC GPT Patch

Apply this once after cloning or resetting MLRC-Bench:

```bash
cd ~/work/VeriTAS
bash scripts/apply_mlrc_patch.sh ../MLRC-Bench
```

The script is idempotent. If the patch is already applied, it exits cleanly.

The patch also replaces MLRC's hardcoded author Python path in `launch.sh`
(`/home/yunxiang/...`) with a portable `${HOME}/miniconda3/...` lookup and
fallback to `which python`.

## Required API Environment

For current GPT models, use the public OpenAI API path:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_REASONING_EFFORT="low"
unset MY_AZURE_OPENAI_ENDPOINT
```

Use `OPENAI_REASONING_EFFORT=none` only for plumbing smoke tests. Use `low` for
normal dev trials and consider `medium` for final candidate generation if
budget allows.

## Run MLRC

```bash
cd ~/work/MLRC-Bench
conda activate mlab

MODEL="gpt-5.4-mini"
bash scripts/init_env.sh product-recommendation "${MODEL}" 0 "TEST_MODEL"
bash launch.sh product-recommendation "${MODEL}" 0 "TEST_MODEL"
```

## Preserve Candidate Predictions

The product-recommendation verifier needs candidate `pred.csv` files and dev
labels. If MLRC cleanup removes trace `output/` directories, rerun frozen
snapshots manually and copy each prediction file into the trial folder.

```bash
mkdir -p ~/work/trials/product-recommendation/run-001/preds
cp /path/to/output/pred.csv \
  ~/work/trials/product-recommendation/run-001/preds/cand_001.csv
```

Create candidate specs:

```jsonl
{"candidate_id":"baseline","method_name":"my_method","prediction_path":"/home/USER/work/trials/product-recommendation/run-001/preds/baseline.csv","is_incumbent":true}
{"candidate_id":"cand_001","method_name":"method_a","prediction_path":"/home/USER/work/trials/product-recommendation/run-001/preds/cand_001.csv"}
```

## Run VeriTAS

```bash
cd ~/work/VeriTAS

python3 scripts/validate_verifier.py \
  --ledger-dir ~/work/trials/product-recommendation/run-001/ledger \
  --mlrc-task product-recommendation \
  --candidate-specs ~/work/trials/product-recommendation/run-001/candidate_specs.jsonl \
  --labels-path ~/work/MLRC-Bench/MLAgentBench/benchmarks/product-recommendation/env/data/dev_labels.csv \
  --incumbent-id baseline \
  --product-recommendation-metric parsed_mrr \
  --epsilon 0.01 \
  --bootstrap-samples 2000 \
  --seed 0
```

The stable integration boundary is artifact-based:

```text
MLRC-Bench -> pred.csv / idea_evals.json / logs
VeriTAS -> unit_outputs / candidates / verifier_report / scheduler action
```

## One-Command Completed-Run Audit

For product-recommendation runs, use this wrapper after `launch.sh` finishes.
It selects the best dev snapshot from MLRC's `idea_evals.json`, regenerates
baseline and candidate `pred.csv` files in an isolated trial directory, prepares
the ledger, runs VeriTAS, and writes `audit_summary.json`.

```bash
cd ~/work/VeriTAS
conda activate mlab

python scripts/audit_mlrc_product_run.py \
  --mlrc-dir ~/work/MLRC-Bench \
  --run-dir ~/work/MLRC-Bench/logs/product-recommendation/gpt-5.4/RUN_ID \
  --task-python ~/miniconda3/envs/product-recommendation/bin/python \
  --bootstrap-samples 200 \
  --epsilon 0.01
```

The main outputs are:

```text
~/work/trials/product-recommendation/RUN_ID/preds/
~/work/trials/product-recommendation/RUN_ID/ledger/verifier_report.json
~/work/trials/product-recommendation/RUN_ID/ledger/scheduler_next_action.json
~/work/trials/product-recommendation/RUN_ID/ledger/agent_feedback.txt
```

## Verifier Hook From Existing Predictions

Use this when a baseline `pred.csv` and candidate `pred.csv` already exist. It
is the smallest online-control boundary: MLRC can call it after a candidate dev
evaluation, then read `agent_feedback.txt` or `scheduler_next_action.json`.

```bash
cd ~/work/VeriTAS
conda activate mlab

python scripts/mlrc_verifier_hook.py \
  --ledger-dir ~/work/trials/product-recommendation/RUN_ID/hook_ledger \
  --baseline-pred ~/work/trials/product-recommendation/RUN_ID/preds/baseline.csv \
  --candidate-pred ~/work/trials/product-recommendation/RUN_ID/preds/CANDIDATE.csv \
  --labels-path ~/work/MLRC-Bench/MLAgentBench/benchmarks/product-recommendation/env/data/dev_labels.csv \
  --candidate-id CANDIDATE_ID \
  --candidate-method CANDIDATE_METHOD \
  --bootstrap-samples 200 \
  --epsilon 0.01
```

## Enable The MLRC Online Hook

This is the minimal in-run integration for product-recommendation. It preserves
each successful dev `pred.csv` and, once a baseline prediction exists, runs the
VeriTAS hook after candidate dev evaluations. With `VERITAS_HOOK_STEER=1`, the
hook also appends concise verifier feedback to the agent's script observation.
Hook failures are logged and MLRC continues.

Apply the patch after the GPT patch:

```bash
cd ~/work/VeriTAS
bash scripts/apply_mlrc_veritas_hook_patch.sh ../MLRC-Bench
```

Enable it before `launch.sh`:

```bash
cd ~/work/MLRC-Bench
conda activate mlab

export VERITAS_HOOK_ENABLE=1
export VERITAS_HOOK_STEER=1
export VERITAS_REPO_DIR=~/work/VeriTAS
export VERITAS_HOOK_BOOTSTRAP_SAMPLES=50
export VERITAS_HOOK_EPSILON=0.01
export VERITAS_HOOK_TIMEOUT=300

bash launch.sh product-recommendation gpt-5.4 0 TEST_MODEL
```

During the run, inspect:

```bash
RUN_DIR=logs/product-recommendation/gpt-5.4/RUN_ID

find "$RUN_DIR/env_log/preds" -type f
find "$RUN_DIR/env_log/veritas" -name scheduler_next_action.json -o -name agent_feedback.txt
```
