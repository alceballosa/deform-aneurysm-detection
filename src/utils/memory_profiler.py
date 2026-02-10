"""
Memory profiling utilities for identifying memory bottlenecks in training.

Usage:
    from src.utils.memory_profiler import MemoryProfiler, profile_model_memory

    # Option 1: Context manager for specific code blocks
    with MemoryProfiler("forward_pass"):
        output = model(input)

    # Option 2: Full model profiling
    profile_model_memory(model, sample_input, cfg)
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, List, Optional
import time


class MemoryStats:
    """Container for memory statistics."""

    def __init__(self, name: str):
        self.name = name
        self.allocated_start = 0
        self.allocated_end = 0
        self.reserved_start = 0
        self.reserved_end = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.time_ms = 0

    @property
    def allocated_diff(self) -> int:
        """Memory allocated during this operation (bytes)."""
        return self.allocated_end - self.allocated_start

    @property
    def reserved_diff(self) -> int:
        """Memory reserved during this operation (bytes)."""
        return self.reserved_end - self.reserved_start

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'name': self.name,
            'allocated_mb': self.allocated_diff / 1024**2,
            'reserved_mb': self.reserved_diff / 1024**2,
            'peak_allocated_mb': self.peak_allocated / 1024**2,
            'peak_reserved_mb': self.peak_reserved / 1024**2,
            'time_ms': self.time_ms,
        }

    def __str__(self) -> str:
        return (
            f"{self.name:30s} | "
            f"Alloc: {self.allocated_diff/1024**2:7.2f} MB | "
            f"Reserved: {self.reserved_diff/1024**2:7.2f} MB | "
            f"Peak: {self.peak_allocated/1024**2:7.2f} MB | "
            f"Time: {self.time_ms:6.1f} ms"
        )


class MemoryProfiler:
    """Context manager for profiling memory usage of code blocks."""

    _stats_history: List[MemoryStats] = []

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled and torch.cuda.is_available()
        self.stats = MemoryStats(name)

    def __enter__(self):
        if not self.enabled:
            return self

        torch.cuda.synchronize()
        self.stats.allocated_start = torch.cuda.memory_allocated()
        self.stats.reserved_start = torch.cuda.memory_reserved()
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        torch.cuda.synchronize()
        self.stats.allocated_end = torch.cuda.memory_allocated()
        self.stats.reserved_end = torch.cuda.memory_reserved()
        self.stats.peak_allocated = torch.cuda.max_memory_allocated()
        self.stats.peak_reserved = torch.cuda.max_memory_reserved()
        self.stats.time_ms = (time.time() - self.start_time) * 1000

        # Store in history
        MemoryProfiler._stats_history.append(self.stats)

    @classmethod
    def print_summary(cls):
        """Print summary of all profiled operations."""
        if not cls._stats_history:
            print("No memory profiling data collected.")
            return

        print("\n" + "="*100)
        print("MEMORY PROFILING SUMMARY")
        print("="*100)
        print(f"{'Operation':<30} | {'Allocated':>12} | {'Reserved':>13} | {'Peak':>12} | {'Time':>10}")
        print("-"*100)

        total_alloc = 0
        total_reserved = 0
        total_time = 0

        for stats in cls._stats_history:
            print(stats)
            total_alloc += stats.allocated_diff
            total_reserved += stats.reserved_diff
            total_time += stats.time_ms

        print("-"*100)
        print(f"{'TOTAL':<30} | {total_alloc/1024**2:10.2f} MB | {total_reserved/1024**2:11.2f} MB | "
              f"{'':>12} | {total_time:8.1f} ms")
        print("="*100 + "\n")

    @classmethod
    def clear_history(cls):
        """Clear profiling history."""
        cls._stats_history.clear()

    @classmethod
    def get_history(cls) -> List[MemoryStats]:
        """Get profiling history."""
        return cls._stats_history


def profile_forward_pass(model: nn.Module, batch: Dict, detailed: bool = True) -> Dict:
    """
    Profile memory usage during forward pass.

    Args:
        model: The model to profile
        batch: Input batch dictionary
        detailed: If True, profile individual components

    Returns:
        Dictionary with memory statistics
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return {}

    model.eval()  # Eval mode to avoid dropout randomness
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    MemoryProfiler.clear_history()

    print("\nProfiling forward pass...")

    with MemoryProfiler("total_forward"):
        # If detailed profiling requested, hook into submodules
        if detailed:
            hooks = []

            def make_hook(name):
                def hook(module, input, output):
                    # Just record that we passed through this module
                    # Actual memory tracking happens via context managers
                    pass
                return hook

            # Register hooks for major components
            for name, module in model.named_modules():
                if any(key in name for key in ['backbone', 'encoder', 'decoder', 'head']):
                    hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            with torch.no_grad():
                # Profile data loading
                with MemoryProfiler("input_data"):
                    inputs = batch.get('inputs', batch.get('image', None))
                    if inputs is not None and not inputs.is_cuda:
                        inputs = inputs.cuda()

                # Profile forward pass
                with MemoryProfiler("model_forward"):
                    output = model(batch)
        finally:
            if detailed:
                for hook in hooks:
                    hook.remove()

    MemoryProfiler.print_summary()

    # Return peak memory usage
    return {
        'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
    }


def profile_backward_pass(model: nn.Module, batch: Dict, loss_fn) -> Dict:
    """
    Profile memory usage during backward pass.

    Args:
        model: The model to profile
        batch: Input batch dictionary
        loss_fn: Loss function to compute gradients

    Returns:
        Dictionary with memory statistics
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return {}

    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    MemoryProfiler.clear_history()

    print("\nProfiling backward pass...")

    with MemoryProfiler("total_backward"):
        # Forward pass
        with MemoryProfiler("forward"):
            output = model(batch)

        # Loss computation
        with MemoryProfiler("loss_computation"):
            loss = loss_fn(output, batch)

        # Backward pass
        with MemoryProfiler("backward"):
            loss.backward()

    MemoryProfiler.print_summary()

    return {
        'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
    }


def profile_model_components(model: nn.Module, batch: Dict) -> Dict:
    """
    Profile memory usage of individual model components.

    This attempts to isolate backbone, encoder, decoder, and heads.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return {}

    print("\nProfiling individual components...")
    print("="*100)

    results = {}

    # Try to identify and profile major components
    model.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        # Profile backbone
        if hasattr(model, 'backbone'):
            torch.cuda.reset_peak_memory_stats()
            with MemoryProfiler("backbone"):
                features = model.backbone(batch['inputs'])
            results['backbone_mb'] = torch.cuda.max_memory_allocated() / 1024**2

        # Profile encoder (if exists)
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
            torch.cuda.reset_peak_memory_stats()
            # Note: This is approximate since encoder needs specific inputs
            print("  Encoder profiling requires full forward pass (integrated in total)")

        # Profile decoder (if exists)
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder'):
            print("  Decoder profiling requires full forward pass (integrated in total)")

    return results


def print_memory_summary():
    """Print current CUDA memory summary."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    max_reserved = torch.cuda.max_memory_reserved() / 1024**3

    print("\n" + "="*60)
    print("CUDA MEMORY SUMMARY")
    print("="*60)
    print(f"Current allocated:    {allocated:6.2f} GB")
    print(f"Current reserved:     {reserved:6.2f} GB")
    print(f"Peak allocated:       {max_allocated:6.2f} GB")
    print(f"Peak reserved:        {max_reserved:6.2f} GB")
    print("="*60 + "\n")


@contextmanager
def profile_memory_context(name: str = "operation"):
    """Simple context manager for one-off memory profiling."""
    profiler = MemoryProfiler(name)
    with profiler:
        yield profiler
    print(profiler.stats)
