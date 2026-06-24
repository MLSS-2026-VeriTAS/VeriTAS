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
  --epsilon 0.005 \
  --bootstrap-samples 2000 \
  --seed 0
```

The stable integration boundary is artifact-based:

```text
MLRC-Bench -> pred.csv / idea_evals.json / logs
VeriTAS -> unit_outputs / candidates / verifier_report / scheduler action
```
