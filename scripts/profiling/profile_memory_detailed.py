#!/usr/bin/env python3
"""
Detailed memory profiling with per-module granularity.
Uses forward hooks to track memory at each major component.
"""

import argparse
import torch
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))

from detectron2.config import get_cfg
from detectron2.utils.events import EventStorage


class DetailedMemoryProfiler:
    """Track memory usage per module with hooks."""

    def __init__(self):
        self.memory_stats = defaultdict(lambda: {'count': 0, 'total_mb': 0, 'peak_mb': 0})
        self.hooks = []
        self.enabled = True

    def register_hooks(self, model, prefix=""):
        """Register forward hooks on all modules."""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Skip if it's a container with children
            if len(list(module.children())) > 0:
                self.register_hooks(module, full_name)
            else:
                # Leaf module - register hook
                hook = module.register_forward_hook(self._make_hook(full_name))
                self.hooks.append(hook)

    def _make_hook(self, name):
        """Create a hook that measures memory after this module."""
        def hook(module, input, output):
            if not self.enabled or not torch.cuda.is_available():
                return

            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**2

            stats = self.memory_stats[name]
            stats['count'] += 1
            stats['total_mb'] += allocated
            stats['peak_mb'] = max(stats['peak_mb'], allocated)

        return hook

    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_summary(self, top_k=20):
        """Get top K modules by peak memory."""
        sorted_stats = sorted(
            self.memory_stats.items(),
            key=lambda x: x[1]['peak_mb'],
            reverse=True
        )

        print("\n" + "="*100)
        print(f"TOP {top_k} MODULES BY PEAK MEMORY")
        print("="*100)
        print(f"{'Module':<60} {'Peak (MB)':>12} {'Avg (MB)':>12} {'Count':>8}")
        print("-"*100)

        for name, stats in sorted_stats[:top_k]:
            avg_mb = stats['total_mb'] / stats['count'] if stats['count'] > 0 else 0
            print(f"{name:<60} {stats['peak_mb']:>12.2f} {avg_mb:>12.2f} {stats['count']:>8}")

        print("="*100)


def profile_with_hooks(cfg):
    """Profile memory with detailed per-module tracking."""

    print("="*100)
    print("DETAILED MEMORY PROFILING WITH HOOKS")
    print("="*100)

    # Create dummy batch
    from profile_memory import create_dummy_batch
    batch = create_dummy_batch(cfg)

    # Build model
    from src.models.deformable.def_parq_rec import PARQ_Deformable_R
    model = PARQ_Deformable_R(cfg)
    model = model.cuda()
    model.train()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\nRegistering hooks on model modules...")
    profiler = DetailedMemoryProfiler()
    profiler.register_hooks(model)
    print(f"Registered {len(profiler.hooks)} hooks")

    # Forward pass
    print("\nRunning forward pass...")
    with EventStorage() as storage:
        try:
            output = model(batch)
            print("✓ Forward pass complete")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

    # Print results
    profiler.get_summary(top_k=30)

    # Cleanup
    profiler.clear_hooks()

    # Overall stats
    print("\n" + "="*100)
    print("OVERALL MEMORY USAGE")
    print("="*100)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    current_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"Peak allocated:    {peak_mb:.2f} MB")
    print(f"Current allocated: {current_mb:.2f} MB")
    print("="*100)


def profile_manual_breakdown(cfg):
    """Manually profile major components by calling them separately."""

    print("\n" + "="*100)
    print("MANUAL COMPONENT BREAKDOWN")
    print("="*100)

    from profile_memory import create_dummy_batch
    batch = create_dummy_batch(cfg)

    from src.models.deformable.def_parq_rec import PARQ_Deformable_R
    model = PARQ_Deformable_R(cfg)
    model = model.cuda()
    model.train()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    results = {}

    # Step 1: Preprocess input
    print("\n1. Preprocessing input...")
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1024**2

    x, vessel_dists, cvs_dists = model.preprocess_train_input(batch)

    torch.cuda.synchronize()
    results['preprocess'] = {
        'allocated': (torch.cuda.memory_allocated() / 1024**2) - start_mem,
        'peak': torch.cuda.max_memory_allocated() / 1024**2
    }
    print(f"   Allocated: {results['preprocess']['allocated']:.2f} MB, Peak: {results['preprocess']['peak']:.2f} MB")

    # Step 2: Backbone
    if hasattr(model, 'backbone'):
        print("\n2. Backbone...")
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() / 1024**2

        if vessel_dists is not None:
            backbone_input = torch.cat([x, vessel_dists], dim=1)
        else:
            backbone_input = x

        if cvs_dists is not None:
            backbone_input = torch.cat([backbone_input, cvs_dists], dim=1)

        features = model.backbone(backbone_input)

        torch.cuda.synchronize()
        results['backbone'] = {
            'allocated': (torch.cuda.memory_allocated() / 1024**2) - start_mem,
            'peak': torch.cuda.max_memory_allocated() / 1024**2
        }
        print(f"   Allocated: {results['backbone']['allocated']:.2f} MB, Peak: {results['backbone']['peak']:.2f} MB")

        # Print output shapes safely
        try:
            if isinstance(features, (list, tuple)):
                shapes = [f.shape if hasattr(f, 'shape') else type(f) for f in features]
                print(f"   Output shapes: {shapes}")
            elif hasattr(features, 'shape'):
                print(f"   Output shape: {features.shape}")
            else:
                print(f"   Output type: {type(features)}")
        except Exception as e:
            print(f"   Could not print output shapes: {e}")

    # Step 3: Full forward (to get transformer)
    print("\n3. Full forward pass (includes encoder + decoder)...")
    torch.cuda.reset_peak_memory_stats()

    with EventStorage() as storage:
        output = model(batch)

    results['full_forward'] = {
        'peak': torch.cuda.max_memory_allocated() / 1024**2
    }
    print(f"   Peak: {results['full_forward']['peak']:.2f} MB")

    # Step 4: Backward pass
    print("\n4. Backward pass...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Need to do full forward + backward
    model.zero_grad()

    with EventStorage() as storage:
        output = model(batch)

    # Compute loss from output
    if isinstance(output, dict):
        loss_dict = output.get('loss_dict', {})
        if loss_dict:
            loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
        else:
            # Fallback: sum all tensors with gradients
            tensors = [v for v in output.values() if isinstance(v, torch.Tensor) and v.requires_grad]
            loss = sum(t.sum() for t in tensors) if tensors else None
    else:
        loss = None

    if loss is not None:
        start_mem = torch.cuda.memory_allocated() / 1024**2
        loss.backward()
        torch.cuda.synchronize()

        results['backward'] = {
            'allocated': (torch.cuda.memory_allocated() / 1024**2) - start_mem,
            'peak': torch.cuda.max_memory_allocated() / 1024**2
        }
        print(f"   Allocated: {results['backward']['allocated']:.2f} MB, Peak: {results['backward']['peak']:.2f} MB")
    else:
        print("   ⚠ Could not compute loss for backward pass")

    # Print summary
    print("\n" + "="*100)
    print("COMPONENT BREAKDOWN SUMMARY")
    print("="*100)
    print(f"{'Component':<30} {'Allocated (MB)':>15} {'Peak (MB)':>15}")
    print("-"*100)
    for comp, stats in results.items():
        alloc = stats.get('allocated', 'N/A')
        peak = stats['peak']
        alloc_str = f"{alloc:.2f}" if isinstance(alloc, float) else alloc
        print(f"{comp:<30} {alloc_str:>15} {peak:>15.2f}")
    print("="*100)

    # Additional analysis
    if 'backward' in results and 'full_forward' in results:
        total_training_peak = results['full_forward']['peak'] + results['backward'].get('allocated', 0)
        print(f"\n📊 Total training memory estimate: ~{total_training_peak:.2f} MB")
        print(f"   (Forward peak + Backward allocation)")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--method", default="hooks", choices=["hooks", "manual", "both"])
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Load config
    from src.config import add_config
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.method in ["hooks", "both"]:
        profile_with_hooks(cfg)

    if args.method in ["manual", "both"]:
        profile_manual_breakdown(cfg)


if __name__ == "__main__":
    main()
