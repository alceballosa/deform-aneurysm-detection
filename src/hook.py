import os
import subprocess

import torch
from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage


class PeriodicCudaCacheClearer(HookBase):
    def __init__(self, period):
        self._period = period

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0:
            torch.cuda.empty_cache()


class GPUUtilizationTracker(HookBase):
    """Periodically samples GPU utilization via nvidia-smi and reports the average."""

    def __init__(self, period=20, window=50):
        self._period = period
        self._window = window
        self._samples = []

    def _query_gpu_utilization(self):
        """Query utilization for GPUs visible to this process."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            values = []
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if visible is not None:
                indices = [int(x) for x in visible.split(",")]
                for idx in indices:
                    if idx < len(lines):
                        values.append(float(lines[idx]))
            else:
                values = [float(l) for l in lines]
            return sum(values) / len(values) if values else None
        except Exception:
            return None

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0:
            util = self._query_gpu_utilization()
            if util is not None:
                self._samples.append(util)
                if len(self._samples) > self._window:
                    self._samples = self._samples[-self._window:]
                storage = get_event_storage()
                avg_util = sum(self._samples) / len(self._samples)
                storage.put_scalar("[metric]gpu_util", avg_util, smoothing_hint=False)
