"""
Training hook for periodic memory profiling during training.

Usage in train_net.py:
    from src.utils.memory_hook import MemoryMonitorHook

    # Add to your hooks list
    hooks.append(MemoryMonitorHook(
        period=100,  # Profile every 100 iterations
        detailed=False,  # Set True for detailed profiling (slower)
    ))
"""

import torch
import logging
from typing import Optional
from detectron2.engine import HookBase


class MemoryMonitorHook(HookBase):
    """
    Hook for monitoring GPU memory usage during training.

    This hook logs memory statistics at regular intervals and can help
    identify memory leaks or unexpected memory growth during training.
    """

    def __init__(
        self,
        period: int = 100,
        detailed: bool = False,
        alert_threshold_gb: float = 20.0,
    ):
        """
        Args:
            period: Log memory stats every N iterations
            detailed: If True, log detailed breakdown (may impact performance)
            alert_threshold_gb: Alert if memory exceeds this threshold (GB)
        """
        self.period = period
        self.detailed = detailed
        self.alert_threshold = alert_threshold_gb * 1024**3  # Convert to bytes
        self.logger = logging.getLogger(__name__)

        # Track memory history
        self.memory_history = []
        self.peak_memory = 0

    def before_train(self):
        """Reset memory stats at start of training."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.peak_memory = 0
            self.memory_history = []
            self.logger.info("Memory monitoring started")

    def after_step(self):
        """Log memory stats periodically."""
        if not torch.cuda.is_available():
            return

        iteration = self.trainer.iter

        # Log every period iterations
        if iteration % self.period == 0:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()

            # Update peak
            if max_allocated > self.peak_memory:
                self.peak_memory = max_allocated

            # Store in history
            self.memory_history.append({
                'iteration': iteration,
                'allocated_gb': allocated / 1024**3,
                'reserved_gb': reserved / 1024**3,
                'max_allocated_gb': max_allocated / 1024**3,
            })

            # Log basic stats
            self.logger.info(
                f"[Memory] Iter {iteration:6d}: "
                f"Alloc={allocated/1024**3:.2f}GB, "
                f"Reserved={reserved/1024**3:.2f}GB, "
                f"Peak={max_allocated/1024**3:.2f}GB"
            )

            # Alert if exceeding threshold
            if allocated > self.alert_threshold:
                self.logger.warning(
                    f"Memory usage ({allocated/1024**3:.2f}GB) exceeds threshold "
                    f"({self.alert_threshold/1024**3:.2f}GB)!"
                )

            # Detailed breakdown if requested
            if self.detailed:
                self._log_detailed_stats()

    def _log_detailed_stats(self):
        """Log detailed memory breakdown."""
        if not torch.cuda.is_available():
            return

        # Get detailed memory stats
        stats = torch.cuda.memory_stats()

        allocated_bytes = stats.get('allocated_bytes.all.current', 0)
        reserved_bytes = stats.get('reserved_bytes.all.current', 0)
        active_bytes = stats.get('active_bytes.all.current', 0)
        inactive_bytes = stats.get('inactive_bytes.all.current', 0)

        num_alloc_retries = stats.get('num_alloc_retries', 0)
        num_ooms = stats.get('num_ooms', 0)

        self.logger.info(
            f"[Memory Detail] "
            f"Active={active_bytes/1024**3:.2f}GB, "
            f"Inactive={inactive_bytes/1024**3:.2f}GB, "
            f"Retries={num_alloc_retries}, "
            f"OOMs={num_ooms}"
        )

    def after_train(self):
        """Log final memory summary."""
        if not torch.cuda.is_available() or not self.memory_history:
            return

        self.logger.info("="*80)
        self.logger.info("MEMORY TRAINING SUMMARY")
        self.logger.info("="*80)

        # Compute statistics
        peak_iter = max(self.memory_history, key=lambda x: x['max_allocated_gb'])
        avg_allocated = sum(x['allocated_gb'] for x in self.memory_history) / len(self.memory_history)

        self.logger.info(f"Peak memory: {self.peak_memory/1024**3:.2f} GB at iter {peak_iter['iteration']}")
        self.logger.info(f"Average allocated: {avg_allocated:.2f} GB")
        self.logger.info(f"Final allocated: {self.memory_history[-1]['allocated_gb']:.2f} GB")

        # Check for memory leak indicators
        if len(self.memory_history) > 10:
            first_10_avg = sum(x['allocated_gb'] for x in self.memory_history[:10]) / 10
            last_10_avg = sum(x['allocated_gb'] for x in self.memory_history[-10:]) / 10
            growth = last_10_avg - first_10_avg

            if growth > 0.5:  # More than 500MB growth
                self.logger.warning(
                    f"Potential memory leak detected! "
                    f"Memory grew by {growth:.2f}GB from start to end of training."
                )

        self.logger.info("="*80)


class MemorySnapshotHook(HookBase):
    """
    Hook for taking detailed memory snapshots at specific iterations.

    Useful for debugging OOM errors or understanding memory allocation patterns.
    """

    def __init__(self, snapshot_iters: list, output_dir: str = "./memory_snapshots"):
        """
        Args:
            snapshot_iters: List of iterations at which to take snapshots
            output_dir: Directory to save snapshot files
        """
        self.snapshot_iters = set(snapshot_iters)
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)

    def after_step(self):
        """Take memory snapshot if at target iteration."""
        if not torch.cuda.is_available():
            return

        iteration = self.trainer.iter

        if iteration in self.snapshot_iters:
            self._take_snapshot(iteration)

    def _take_snapshot(self, iteration: int):
        """Take and save a memory snapshot."""
        try:
            snapshot_file = f"{self.output_dir}/memory_snapshot_iter_{iteration}.pickle"

            # Take snapshot
            torch.cuda.memory._dump_snapshot(snapshot_file)

            self.logger.info(f"Memory snapshot saved to {snapshot_file}")
            self.logger.info(
                f"Analyze with: python -m torch.cuda._memory_viz trace_plot {snapshot_file} -o snapshot_{iteration}.html"
            )

        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {e}")


def log_model_memory_footprint(model: torch.nn.Module, logger: Optional[logging.Logger] = None):
    """
    Log the memory footprint of model parameters and buffers.

    Args:
        model: PyTorch model
        logger: Logger instance (if None, uses print)
    """
    if logger is None:
        log_fn = print
    else:
        log_fn = logger.info

    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    total_memory = param_memory + buffer_memory

    log_fn("="*60)
    log_fn("MODEL MEMORY FOOTPRINT")
    log_fn("="*60)
    log_fn(f"Parameters: {param_memory/1024**2:.2f} MB ({sum(p.numel() for p in model.parameters())} elements)")
    log_fn(f"Buffers:    {buffer_memory/1024**2:.2f} MB ({sum(b.numel() for b in model.buffers())} elements)")
    log_fn(f"Total:      {total_memory/1024**2:.2f} MB")
    log_fn("="*60)

    # Breakdown by module type (optional)
    module_memory = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_type = type(module).__name__
            module_params = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
            module_buffers = sum(b.numel() * b.element_size() for b in module.buffers(recurse=False))
            module_total = module_params + module_buffers

            if module_total > 0:
                if module_type not in module_memory:
                    module_memory[module_type] = 0
                module_memory[module_type] += module_total

    log_fn("\nMemory by module type:")
    for module_type, memory in sorted(module_memory.items(), key=lambda x: x[1], reverse=True)[:10]:
        log_fn(f"  {module_type:30s}: {memory/1024**2:8.2f} MB")
