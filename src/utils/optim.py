import itertools
import math
import warnings

import torch
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LRScheduler,
    OneCycleLR,
)


def maybe_add_grad_clip_and_accum(cfg, optimizer_class):
    class OptimizerWithOptionalFullGradClipAndAccumulate(optimizer_class):
        def __init__(self, *args, **kwargs):
            self.cfg = kwargs.pop("cfg", {})
            self.enable_grad_clip = cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            self.grad_clip_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            self.enable_grad_accumulate = cfg.SOLVER.GRAD_ACCUM.ENABLED
            self.grad_accumulate_steps = cfg.SOLVER.GRAD_ACCUM.STEPS
            self.step_count = 0
            super().__init__(*args, **kwargs)

        def step(self, closure=None):
            if self.enable_grad_clip:
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip_val)

            if self.enable_grad_accumulate:
                if self.step_count == self.grad_accumulate_steps:
                    super().step(closure=closure)
                    super().zero_grad()
                    self.step_count = 0
            else:
                super().step(closure=closure)

            if self.enable_grad_accumulate == False and self.enable_grad_clip == False:
                super().step(closure=closure)

    return OptimizerWithOptionalFullGradClipAndAccumulate


class CosineAnnealingWithPlateau(LRScheduler):
    """
    Reimplements the Cosine Annealing LR class with a plateau at
    the end. The plateau starts after the iteration defined by
    "T_flat" and lasts until the end of the training process.
    """

    def __init__(
        self, optimizer, T_flat, T_max, eta_min=0, last_epoch=-1, verbose="deprecated"
    ):
        self.T_max = T_max
        self.T_flat = T_flat
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    # TODO: configure the scheduler to flatten within the range
    # of the cosine annealing at the corresponding point

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        lr_adjust_factor = (
            self.T_flat if self.last_epoch > self.T_flat else self.last_epoch
        )

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos((lr_adjust_factor) * math.pi / self.T_flat))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (lr_adjust_factor - 1 - self.T_flat) % (2 * self.T_flat) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_flat)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * lr_adjust_factor / self.T_flat))
            / (1 + math.cos(math.pi * (lr_adjust_factor - 1) / self.T_flat))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        lr_adjust_factor = (
            self.T_flat if self.last_epoch > self.T_flat else self.last_epoch
        )
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * lr_adjust_factor / self.T_flat))
            / 2
            for base_lr in self.base_lrs
        ]


def retrieve_scheduler(
    optimizer, cycle_size, flat_iter, min_lr, max_lr, scheduler_name
):
    if scheduler_name == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, T_max=cycle_size, eta_min=min_lr)
    elif scheduler_name == "CosineAnnealingWithWarmRestarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=cycle_size)
    elif scheduler_name == "CosineAnnealingWithPlateau":
        return CosineAnnealingWithPlateau(
            optimizer, T_flat=flat_iter, T_max=cycle_size, eta_min=min_lr, verbose=False
        )
    elif scheduler_name == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=cycle_size,
            pct_start=0.05,
            anneal_strategy="cos",
        )
    elif scheduler_name == "CosineAnnealingWithDecay":
        pass
    elif scheduler_name == "CosineAnnealingWithWarmRestartsAndDecay":
        pass
    else:
        raise NotImplementedError(f"no scheduler type {scheduler_name}")
