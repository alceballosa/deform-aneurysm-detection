import torch
from detectron2.engine import HookBase


class PeriodicCudaCacheClearer(HookBase):
    def __init__(self, period):
        self._period = period

    def after_step(self):
        return super().after_step()

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0:
            torch.cuda.empty_cache()
