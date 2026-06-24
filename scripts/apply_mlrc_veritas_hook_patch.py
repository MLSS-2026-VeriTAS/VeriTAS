#!/usr/bin/env python3
"""Patch MLRC-Bench to run the VeriTAS product hook after dev evals."""

from __future__ import annotations

import argparse
from pathlib import Path


HELPER_MARKER = "VERITAS_HOOK_ENABLE"

HELPERS = r'''
def _veritas_hook_enabled():
    return os.getenv("VERITAS_HOOK_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_veritas_name(value):
    safe = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in str(value)
    ).strip("._")
    return safe or "candidate"


def _parse_main_dev_eval(script_name_and_args):
    try:
        parts = shlex.split(script_name_and_args)
    except ValueError:
        parts = script_name_and_args.split()

    if not parts or os.path.basename(parts[0]) != "main.py":
        return None

    method_name = None
    phase = "dev"
    for index, part in enumerate(parts):
        if part in {"-m", "--method"} and index + 1 < len(parts):
            method_name = parts[index + 1]
        elif part.startswith("--method="):
            method_name = part.split("=", 1)[1]
        elif part in {"-p", "--phase"} and index + 1 < len(parts):
            phase = parts[index + 1]
        elif part.startswith("--phase="):
            phase = part.split("=", 1)[1]

    if method_name and phase == "dev":
        return method_name
    return None


def _env_log_dir_from_tool_log(log_file):
    if log_file:
        tool_dir = os.path.dirname(os.path.abspath(log_file))
        if os.path.basename(tool_dir) == "tool_logs":
            return os.path.dirname(tool_dir)
    log_dir = os.getenv("LOG_DIR")
    if log_dir:
        return os.path.join(log_dir, "env_log")
    return None


def _write_veritas_hook_error(ledger_dir, message):
    os.makedirs(ledger_dir, exist_ok=True)
    with open(os.path.join(ledger_dir, "hook_error.txt"), "a", encoding="utf-8") as writer:
        writer.write(str(message).rstrip() + "\n")


def _maybe_run_veritas_hook(script_name_and_args, work_dir, return_code, log_file):
    """Preserve product predictions and run VeriTAS without blocking MLRC success."""
    if return_code != 0 or not _veritas_hook_enabled():
        return ""

    method_name = _parse_main_dev_eval(script_name_and_args)
    if not method_name:
        return ""

    pred_path = os.path.join(work_dir, "output", "pred.csv")
    labels_path = os.path.join(work_dir, "data", "dev_labels.csv")
    if not os.path.exists(pred_path) or not os.path.exists(labels_path):
        return ""

    env_log_dir = _env_log_dir_from_tool_log(log_file)
    if not env_log_dir:
        return "VeriTAS hook skipped: could not locate env_log directory."

    curr_step = os.getenv("CURR_STEP", "unknown")
    safe_method = _safe_veritas_name(method_name)
    preds_dir = os.path.join(env_log_dir, "preds")
    veritas_dir = os.path.join(env_log_dir, "veritas")
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(veritas_dir, exist_ok=True)

    candidate_pred = os.path.join(preds_dir, f"step_{curr_step}_{safe_method}.csv")
    shutil.copyfile(pred_path, candidate_pred)

    baseline_method = os.getenv("VERITAS_BASELINE_METHOD", "my_method")
    baseline_path_file = os.path.join(veritas_dir, "baseline_pred_path.txt")
    if method_name == baseline_method:
        with open(baseline_path_file, "w", encoding="utf-8") as writer:
            writer.write(candidate_pred + "\n")
        return f"VeriTAS hook: preserved baseline prediction at {candidate_pred}"

    if not os.path.exists(baseline_path_file):
        return (
            "VeriTAS hook: preserved candidate prediction at "
            f"{candidate_pred}; baseline prediction not available yet."
        )

    with open(baseline_path_file, "r", encoding="utf-8") as reader:
        baseline_pred = reader.read().strip()
    if not baseline_pred or not os.path.exists(baseline_pred):
        return "VeriTAS hook skipped: stored baseline prediction path is missing."

    hook_ledger_dir = os.path.join(veritas_dir, f"step_{curr_step}_{safe_method}")
    veritas_repo_dir = os.path.expanduser(os.getenv("VERITAS_REPO_DIR", "~/work/VeriTAS"))
    hook_script = os.path.join(veritas_repo_dir, "scripts", "mlrc_verifier_hook.py")
    if not os.path.exists(hook_script):
        _write_veritas_hook_error(hook_ledger_dir, f"missing hook script: {hook_script}")
        return f"VeriTAS hook failed: missing hook script {hook_script}"

    cmd = [
        sys.executable,
        hook_script,
        "--ledger-dir",
        hook_ledger_dir,
        "--baseline-pred",
        baseline_pred,
        "--candidate-pred",
        candidate_pred,
        "--labels-path",
        labels_path,
        "--candidate-id",
        f"step_{curr_step}_{safe_method}",
        "--candidate-method",
        method_name,
        "--metric",
        os.getenv("VERITAS_HOOK_METRIC", "parsed_mrr"),
        "--epsilon",
        os.getenv("VERITAS_HOOK_EPSILON", "0.005"),
        "--bootstrap-samples",
        os.getenv("VERITAS_HOOK_BOOTSTRAP_SAMPLES", "50"),
        "--seed",
        os.getenv("VERITAS_HOOK_SEED", "0"),
    ]
    if str(curr_step).isdigit():
        cmd.extend(["--candidate-step", str(curr_step)])

    timeout = float(os.getenv("VERITAS_HOOK_TIMEOUT", "300"))
    try:
        result = subprocess.run(
            cmd,
            cwd=veritas_repo_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        with open(os.path.join(hook_ledger_dir, "hook_stdout.txt"), "w", encoding="utf-8") as writer:
            writer.write(result.stdout)
        with open(os.path.join(hook_ledger_dir, "hook_stderr.txt"), "w", encoding="utf-8") as writer:
            writer.write(result.stderr)
        if result.returncode != 0:
            _write_veritas_hook_error(
                hook_ledger_dir,
                f"hook exited with code {result.returncode}\n{result.stderr}",
            )
            return f"VeriTAS hook failed for {method_name}; MLRC continues."
    except Exception as exc:
        _write_veritas_hook_error(hook_ledger_dir, repr(exc))
        return f"VeriTAS hook failed for {method_name}; MLRC continues."

    return f"VeriTAS hook: wrote verifier artifacts to {hook_ledger_dir}"


'''


def patch_low_level_actions(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if "_maybe_run_veritas_hook" in source:
        return False

    patched = source
    if "import shlex\n" not in patched:
        patched = patched.replace("import inspect\n", "import inspect\nimport shlex\n", 1)

    insert_anchor = "\n\n# @check_file_in_work_dir([\"script_name_and_args\"])\n"
    if insert_anchor not in patched:
        raise RuntimeError("could not find execute_script insertion anchor")
    patched = patched.replace(insert_anchor, "\n\n" + HELPERS + "# @check_file_in_work_dir([\"script_name_and_args\"])\n", 1)

    old_cmd = '        cmd = f"PYTHONPATH=`pwd` CUDA_VISIBLE_DEVICES={device} {python} -u {script_path}"\n'
    new_cmd = (
        '        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))\n'
        '        cmd = f"PYTHONPATH={repo_root}:`pwd`:${{PYTHONPATH:-}} CUDA_VISIBLE_DEVICES={device} {python} -u {script_path}"\n'
    )
    if old_cmd in patched:
        patched = patched.replace(old_cmd, new_cmd, 1)

    return_anchor = (
        '        if observation == "" and return_code == 0:\n'
        '            # printed to stderr only\n'
        '            observation = "".join(stderr_lines)\n'
        '        return "The script has been executed. Here is the output:\\n" + observation\n'
    )
    hooked_return = (
        '        if observation == "" and return_code == 0:\n'
        '            # printed to stderr only\n'
        '            observation = "".join(stderr_lines)\n'
        '        veritas_observation = _maybe_run_veritas_hook(script_name_and_args, work_dir, return_code, log_file)\n'
        '        if veritas_observation:\n'
        '            observation = observation + "\\n" + veritas_observation\n'
        '        return "The script has been executed. Here is the output:\\n" + observation\n'
    )
    if return_anchor not in patched:
        raise RuntimeError("could not find execute_script return anchor")
    patched = patched.replace(return_anchor, hooked_return, 1)

    path.write_text(patched, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mlrc_dir",
        nargs="?",
        default="../MLRC-Bench",
        help="Path to the MLRC-Bench checkout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mlrc_dir = Path(args.mlrc_dir).expanduser().resolve()
    target = mlrc_dir / "MLAgentBench" / "low_level_actions.py"
    if not target.exists():
        raise SystemExit(f"Expected MLRC file is missing: {target}")

    changed = patch_low_level_actions(target)
    if changed:
        print(f"Applied VeriTAS online hook patch to {target}")
    else:
        print(f"VeriTAS online hook patch is already applied to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
