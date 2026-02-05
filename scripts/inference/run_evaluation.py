"""
Python wrapper for running model evaluation across multiple datasets and checkpoints.

Usage:
    python scripts/inference/run_evaluation.py \
        --family lnt \
        --model lnt_cnn_trx_input_edt_lia_fps \
        --num-gpus 2

    python scripts/inference/run_evaluation.py \
        --family lnt \
        --model lnt_cnn_trx_input_edt_lia_fps \
        --num-gpus 2 \
        --threshold 0.9 \
        --datasets-config my_datasets.json \
        --dry-run
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RUN_INFERENCE_SCRIPT = SCRIPT_DIR / "run_inference.sh"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_config(family: str, model: str):
    """Load model config using detectron2's CfgNode to resolve _BASE_ inheritance."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from detectron2.config import get_cfg

    from src.config import add_config

    cfg = get_cfg()
    add_config(cfg)
    config_path = PROJECT_ROOT / "configs" / family / f"{model}.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    return cfg


def get_checkpoint_id(checkpoint: str) -> str:
    """Convert checkpoint to the ID used in results folder names.

    Matches the logic in train_net.py:get_inference_iters()
    """
    if checkpoint == "final" or "final" in checkpoint:
        return "final"
    return f"{math.ceil(int(checkpoint) / 1000)}k"


def result_exists(dataset_name: str, model_name: str, checkpoint: str) -> bool:
    """Check if inference result already exists for this dataset/model/checkpoint."""
    checkpoint_id = get_checkpoint_id(checkpoint)
    result_path = (
        RESULTS_DIR / dataset_name / model_name
        / f"inference_{checkpoint_id}" / "predict.csv"
    )
    return result_path.exists()


def load_datasets_config(datasets_config_path: str):
    """Load the datasets JSON file (array of dataset groups)."""
    path = Path(datasets_config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"Error: Datasets config not found: {path}")
        sys.exit(1)
    with open(path) as f:
        groups = json.load(f)
    if not isinstance(groups, list):
        print("Error: datasets.json must be a JSON array of dataset groups.")
        sys.exit(1)
    return groups


def run_inference(
    dataset_name: str,
    family: str,
    model: str,
    checkpoint: str,
    threshold: float,
    num_gpus: int,
    num_workers: int,
    patches_per_iter: int,
    root: str,
    dry_run: bool = False,
):
    """Call run_inference.sh with the given parameters."""
    cmd = [
        str(RUN_INFERENCE_SCRIPT),
        dataset_name,
        family,
        model,
        checkpoint,
        str(threshold),
        str(num_gpus),
        str(num_workers),
        str(patches_per_iter),
        root,
    ]

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return 0

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run model evaluation across datasets and checkpoints."
    )
    parser.add_argument(
        "--family", required=True, help="Config subdirectory (e.g., lnt, trx, vst)"
    )
    parser.add_argument("--model", required=True, help="Config file name without .yaml")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override POSTPROCESS.THRESHOLD from config",
    )
    parser.add_argument(
        "--datasets-config",
        default="scripts/inference/datasets.json",
        help="Path to datasets JSON file (default: scripts/inference/datasets.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    args = parser.parse_args()

    # Load model config
    print(f"Loading config: configs/{args.family}/{args.model}.yaml")
    cfg = load_config(args.family, args.model)

    checkpoints = list(cfg.TEST.EVALUATION_CHECKPOINTS)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    patches_per_iter = cfg.TEST.PATCHES_PER_ITER
    threshold = (
        args.threshold if args.threshold is not None else cfg.POSTPROCESS.THRESHOLD
    )
    config_root = cfg.DATA.DIR.ROOT

    print(f"  Checkpoints: {checkpoints}")
    print(f"  NUM_WORKERS: {num_workers}")
    print(f"  PATCHES_PER_ITER: {patches_per_iter}")
    print(f"  Threshold: {threshold}")
    print(f"  Config DATA.DIR.ROOT: {config_root}")
    print()

    # Load datasets config
    dataset_groups = load_datasets_config(args.datasets_config)

    # Count total runs for progress
    total_runs = sum(len(g["datasets"]) * len(checkpoints) for g in dataset_groups)
    current_run = 0
    skipped_runs = 0

    for group in dataset_groups:
        root = group.get("root", "") or config_root
        datasets = group["datasets"]
        print(f"Dataset group root: {root}")

        for dataset_name in datasets:
            print(f"\n  Dataset: {dataset_name} (root: {root})")

            for checkpoint in checkpoints:
                current_run += 1

                # Check if result already exists
                if result_exists(dataset_name, args.model, checkpoint):
                    print(
                        f"\n  [{current_run}/{total_runs}] Checkpoint: {checkpoint}"
                        " - SKIPPED (already exists)"
                    )
                    skipped_runs += 1
                    continue

                print(f"\n  [{current_run}/{total_runs}] Checkpoint: {checkpoint}")
                returncode = run_inference(
                    dataset_name=dataset_name,
                    family=args.family,
                    model=args.model,
                    checkpoint=checkpoint,
                    threshold=threshold,
                    num_gpus=args.num_gpus,
                    num_workers=num_workers,
                    patches_per_iter=patches_per_iter,
                    root=root,
                    dry_run=args.dry_run,
                )
                if returncode != 0:
                    print(f"  Warning: run_inference.sh exited with code {returncode}")

    executed = current_run - skipped_runs
    print(f"\nDone. {executed} inference runs completed, {skipped_runs} skipped.")


if __name__ == "__main__":
    main()
