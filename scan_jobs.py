#!/usr/bin/env python3
"""Scan model training/inference status and write a markdown report."""

import os
import subprocess
import re
from pathlib import Path

BASE = Path("/projects/vig/alberto/medical/exploration/deform")
WEIGHTS_DIR = BASE / "model_weights"
LOGS_DIR = BASE / "logs"
RESULTS_DIR = BASE / "results"

def get_username():
    return os.environ.get("USER", subprocess.check_output("whoami").decode().strip())

def get_latest_checkpoint(model_dir: Path):
    """Return latest checkpoint name, list of all .pth files, and mtime of latest."""
    pth_files = sorted([f.name for f in model_dir.glob("*.pth")])
    latest = None
    mtime = 0.0
    if pth_files:
        # Find the most recently modified .pth file
        newest = max(model_dir.glob("*.pth"), key=lambda f: f.stat().st_mtime)
        latest = newest.name
        mtime = newest.stat().st_mtime
    return latest, pth_files, mtime

def get_running_jobs(username: str):
    """Return list of (jobid, state, name) from squeue."""
    result = subprocess.run(
        ["squeue", "-u", username, "-o", "%i %t %j"],
        capture_output=True, text=True
    )
    jobs = []
    for line in result.stdout.strip().splitlines()[1:]:  # skip header
        parts = line.split(None, 2)
        if len(parts) == 3:
            jobs.append((parts[0], parts[1], parts[2]))
    return jobs

def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)

def match_job_to_model(job_id: str, job_name: str):
    """Look up the log file for this job and find the NAME: "model_name" or Model: model_name line."""
    log_pattern = f"exec.{job_id}."
    log_file = None
    for f in LOGS_DIR.iterdir():
        if f.name.startswith(log_pattern):
            log_file = f
            break

    if log_file is None:
        return None, None

    is_train = "train" in log_file.name
    is_infer = "infer" in log_file.name
    job_type = "train" if is_train else "infer" if is_infer else None

    model_name = None
    try:
        with open(log_file) as fh:
            for i, line in enumerate(fh):
                if i > 200:
                    break
                clean = strip_ansi(line.strip())
                # Check for NAME: "model_name" pattern (older format)
                match = re.match(r'^\s*NAME:\s*"([^"]+)"', clean)
                if match:
                    model_name = match.group(1)
                    break
                # Check for Model: model_name pattern (newer format)
                match = re.match(r'^Model:\s+(.+)$', clean)
                if match:
                    model_name = match.group(1).strip()
                    break
    except Exception:
        pass

    # If we found a model name but couldn't determine job type from filename,
    # infer it from the log content or default to "train"
    if model_name and job_type is None:
        # Look for inference-specific indicators in the file
        try:
            with open(log_file) as fh:
                content = fh.read(5000)  # Read first 5000 chars
                if "eval_only=True" in content or "--eval-only" in content:
                    job_type = "infer"
                else:
                    # Default to train if no inference indicators found
                    job_type = "train"
        except Exception:
            job_type = "train"  # Default to train on error

    return model_name, job_type

def get_inference_checkpoints(model_name: str):
    """For each dataset, count inference checkpoint dirs for this model."""
    results = {}
    if not RESULTS_DIR.exists():
        return results
    for dataset_dir in sorted(RESULTS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        model_dir = dataset_dir / model_name
        if not model_dir.exists():
            continue
        # Count only inference_* directories (not mode_*)
        ckpt_dirs = sorted([
            d.name for d in model_dir.iterdir()
            if d.is_dir() and d.name.startswith("inference_")
        ])
        results[dataset_dir.name] = ckpt_dirs
    return results

def main():
    username = get_username()

    # 1. Gather model info
    models = sorted([d.name for d in WEIGHTS_DIR.iterdir() if d.is_dir()])

    # 2. Get running jobs and resolve model names
    jobs = get_running_jobs(username)
    # Map model_name -> list of (job_id, state, job_type)
    job_model_map = {}
    for job_id, state, job_name in jobs:
        model_name, job_type = match_job_to_model(job_id, job_name)
        if model_name:
            job_model_map.setdefault(model_name, []).append((job_id, state, job_type))
        elif "train" not in job_name and "infer" not in job_name:
            # Check if job name itself matches a model name (queued jobs)
            if job_name in models:
                job_model_map.setdefault(job_name, []).append((job_id, state, "queued"))

    # 3. Build report
    lines = []
    lines.append("# Model Status Report\n")
    lines.append(f"**User:** `{username}`\n")
    lines.append("---\n")

    # Pre-compute datasets list for final inference % column
    datasets = sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])
    n_datasets = len(datasets)

    # Training status table
    lines.append("## Training Status\n")
    lines.append("| Model | Latest Checkpoint | Total Checkpoints | Training Job | Status | Infer Job | Final Infer |")
    lines.append("|-------|-------------------|-------------------|--------------|--------|-----------|-------------|")

    # Include pending/queued jobs whose model name isn't in model_weights
    pending_only_models = set()
    for model_name, job_list in job_model_map.items():
        if model_name not in models:
            if any(st in ("PD", "CF", "CG") for _, st, _ in job_list):
                pending_only_models.add(model_name)
    # Also check unmatched jobs (no log file yet) whose job_name matches no model_weights folder
    for job_id, state, job_name in jobs:
        if job_id not in {jid for jl in job_model_map.values() for jid, _, _ in jl}:
            if state in ("PD", "CF", "CG") and job_name not in models:
                pending_only_models.add(job_name)
                job_model_map.setdefault(job_name, []).append((job_id, state, "queued"))

    all_models = sorted(set(models) | pending_only_models)

    # Collect rows with mtime for sorting
    table_rows = []
    for model in all_models:
        model_dir = WEIGHTS_DIR / model
        if model_dir.is_dir():
            latest, pth_files, mtime = get_latest_checkpoint(model_dir)
            latest_str = latest if latest else "None"
            n_ckpts = len(pth_files)
        else:
            latest, pth_files, mtime = None, [], 0.0
            latest_str = "(no weights)"
            n_ckpts = 0

        # Check for exact match training or queued job
        train_jobs = []
        queued_jobs = []
        if model in job_model_map:
            train_jobs = [(jid, st) for jid, st, jt in job_model_map[model] if jt == "train"]
            queued_jobs = [(jid, st) for jid, st, jt in job_model_map[model] if jt == "queued"]

        if train_jobs:
            job_strs = ", ".join(f"{jid} ({st})" for jid, st in train_jobs)
            status = "Running" if any(st == "R" for _, st in train_jobs) else "Pending"
        elif queued_jobs:
            job_strs = ", ".join(f"{jid} ({st})" for jid, st in queued_jobs)
            status = "Queued"
        else:
            job_strs = "-"
            status = "Finished" if latest == "model_final.pth" else "Not running"

        # Count datasets with inference_final
        final_count = sum(
            1 for ds in datasets
            if (RESULTS_DIR / ds / model / "inference_final").is_dir()
        )
        if n_datasets > 0:
            final_pct = f"{final_count}/{n_datasets} ({100 * final_count // n_datasets}%)"
        else:
            final_pct = "-"

        # Check for inference jobs
        infer_jobs_for_model = []
        if model in job_model_map:
            infer_jobs_for_model = [(jid, st) for jid, st, jt in job_model_map[model] if jt == "infer"]
        if infer_jobs_for_model:
            infer_strs = ", ".join(f"{jid} ({st})" for jid, st in infer_jobs_for_model)
        else:
            infer_strs = "-"

        row = f"| `{model}` | `{latest_str}` | {n_ckpts} | {job_strs} | **{status}** | {infer_strs} | {final_pct} |"
        table_rows.append((mtime, row))

    # Sort by most recent checkpoint first
    table_rows.sort(key=lambda x: x[0], reverse=True)
    for _, row in table_rows:
        lines.append(row)

    lines.append("")

    # Inference results
    lines.append("## Inference Results\n")

    for dataset in datasets:
        lines.append(f"### {dataset}\n")
        lines.append("| Model | Checkpoints Completed | Checkpoints |")
        lines.append("|-------|-----------------------|-------------|")

        dataset_dir = RESULTS_DIR / dataset
        model_dirs = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])

        if not model_dirs:
            lines.append("| (none) | - | - |")

        for model_name in model_dirs:
            ckpts = get_inference_checkpoints(model_name).get(dataset, [])
            ckpt_list = ", ".join(c.replace("inference_", "") for c in ckpts)
            lines.append(f"| `{model_name}` | {len(ckpts)} | {ckpt_list} |")

        # Check for running inference jobs for this dataset
        lines.append("")

    # Running inference jobs section
    lines.append("## Currently Running Inference Jobs\n")
    infer_jobs = [(m, jid, st) for m, job_list in job_model_map.items()
                  for jid, st, jt in job_list if jt == "infer"]
    if infer_jobs:
        lines.append("| Model | Job ID | Status |")
        lines.append("|-------|--------|--------|")
        for model_name, jid, st in sorted(infer_jobs):
            status = "Running" if st == "R" else "Pending"
            lines.append(f"| `{model_name}` | {jid} | **{status}** |")
    else:
        lines.append("No inference jobs currently running.")

    lines.append("")

    # Unmatched jobs (jobs whose resolved model name doesn't match any model_weights folder,
    # or pending jobs with no log file yet)
    matched_job_ids = set()
    for job_list in job_model_map.values():
        for jid, _, _ in job_list:
            matched_job_ids.add(jid)

    unmatched = []
    for job_id, state, job_name in jobs:
        if job_id not in matched_job_ids:
            # Try to resolve model name from log anyway
            model_name, job_type = match_job_to_model(job_id, job_name)
            reason = "no log file (pending)" if model_name is None else f"model `{model_name}` not in model_weights"
            jtype = job_type if job_type else ("train" if "train" in job_name else "infer" if "infer" in job_name else "other")
            unmatched.append((job_id, state, job_name, jtype, reason))

    if unmatched:
        lines.append("## Unmatched Jobs\n")
        lines.append("These jobs are running/pending but could not be matched to a model_weights folder.\n")
        lines.append("| Job ID | State | Job Name | Type | Reason |")
        lines.append("|--------|-------|----------|------|--------|")
        for jid, st, jname, jtype, reason in unmatched:
            lines.append(f"| {jid} | {st} | `{jname}` | {jtype} | {reason} |")
        lines.append("")

    report = "\n".join(lines)
    output_path = BASE / "status_report.md"
    output_path.write_text(report)
    print(f"Report written to {output_path}")
    print()
    print(report)

if __name__ == "__main__":
    main()
