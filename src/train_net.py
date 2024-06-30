# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
adapted from
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""


try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

import copy
import datetime
import itertools
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import bitsandbytes as bnb

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_dir[:-4])

import random

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import verify_results
from detectron2.modeling import build_model

# somehow work after import detectron2.data
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from src import dataset, models
from src.config import add_config
from src.dataset import (
    DATA_MAPPER_REGISTRY,
    build_test_loader,
    build_train_loader,
    setup_data_catalog,
)
from src.evaluator import CTAEvaluator
from src.hook import PeriodicCudaCacheClearer
from src.utils.optim import maybe_add_grad_clip_and_accum, retrieve_scheduler

did_training = False

torch.multiprocessing.set_sharing_strategy("file_system")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_inference_iters(cfg):
    model_weights = cfg.MODEL.WEIGHTS
    if model_weights:
        model_weights = model_weights.replace(".pth", "").split("/")[-1].split("_")[-1]
        if "final" in model_weights:
            model_weights = "final"
            return model_weights
        model_weights = math.ceil(int(model_weights) / 1000)
        return f"{model_weights}k"
    else:
        if did_training:
            return "final"
        else:
            raise ValueError("model weights not found")


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg)
        if cfg.CUSTOM.CLEAR_CUDA_CACHE_PERIOD:
            self.register_hooks(
                [PeriodicCudaCacheClearer(cfg.CUSTOM.CLEAR_CUDA_CACHE_PERIOD)]
            )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(
                cfg.OUTPUT_DIR, "inference_" + get_inference_iters(cfg)
            )
        return CTAEvaluator(
            cfg, dataset_name, distributed=True, output_dir=output_folder
        )

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        Mapper = DATA_MAPPER_REGISTRY.get(cfg.CUSTOM.DATASET_MAPPER)
        if cfg.CUSTOM.DATASET_MAPPER == "":
            mapper = None
        else:
            mapper = Mapper(cfg, mode="train")

        return build_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DATA_MAPPER_REGISTRY.get(cfg.CUSTOM.DATASET_MAPPER)(cfg, mode="val")
        return build_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        # TODO: here
        valid_schedulers = [
            "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts",
            "CosineAnnealingWithPlateau",
        ]

        if cfg.SOLVER.LR_SCHEDULER_NAME in valid_schedulers:
            return retrieve_scheduler(
                optimizer,
                cfg.SOLVER.SCHED_CYCLE,
                cfg.SOLVER.FLAT_ITER,
                cfg.SOLVER.MIN_LR,
                cfg.SOLVER.BASE_LR,
                cfg.SOLVER.LR_SCHEDULER_NAME,
            )

        else:
            return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                # if "backbone" in module_name:
                #     hyperparams["lr"] = (
                #         hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                #     )
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                params.append({"params": [value], **hyperparams})

        def maybe_add_grad_accumulation(cfg, optim):
            enable = cfg.SOLVER.GRAD_ACCUM.ENABLED and cfg.SOLVER.GRAD_ACCUM.STEPS > 2

            class GradAccumulationOptimizer(optim):
                _num_grad_accum = cfg.SOLVER.GRAD_ACCUM.STEPS
                _num_grad_accum_counter = 0
                # def __init__(self, *args, **kwargs):
                #     print(args)
                #     print(kwargs)
                #     input()
                #     super(GradAccumulationOptimizer, self).__init__(*args, **kwargs)

                #     self._num_grad_accum = cfg.SOLVER.GRAD_ACCUMULATION.STEPS
                #     self._num_grad_accum_counter = 0

                def step(self, closure=None):
                    self._num_grad_accum_counter += 1
                    if self._num_grad_accum_counter == self._num_grad_accum:
                        super().step(closure=closure)
                        super().zero_grad()
                        self._num_grad_accum_counter = 0

            return GradAccumulationOptimizer if enable else optim

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_grad_clip_and_accum(cfg, torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_grad_clip_and_accum(cfg, torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        elif optimizer_type == "ADAMW_8BIT":
            optimizer = maybe_add_grad_clip_and_accum(cfg, bnb.optim.AdamW8bit)(
                params, cfg.SOLVER.BASE_LR
            )
        elif optimizer_type == "LION_8BIT":
            optimizer = maybe_add_grad_clip_and_accum(cfg, bnb.optim.Lion8bit)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        # if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        #    optimizer = maybe_add_gradient_clipping(cfg, optimizer)

        # optimizer = maybe_add_grad_accumulation(cfg, optimizer)
        # optimizer = maybe_add_grad_clip_and_accum(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.RESUME = args.resume
    cfg.OUTPUT_DIR = os.path.join(
        cfg.OUTPUT_DIR, os.path.split(args.config_file)[-1].replace(".yaml", "")
    )

    # use weights from the training folder for eval on other datasets
    if "EXT" in cfg.MODEL.WEIGHTS:
        cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS.replace("_EXT", "")
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if "TI" in cfg.MODEL.WEIGHTS:
        cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS.replace("_TI", "")
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="src")
    return cfg


def main(args):
    global did_training
    cfg = setup(args)
    setup_data_catalog(cfg)

    seed = cfg.SEED
    seed_everything(seed)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        # model = torch.compile(model)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        # if comm.is_main_process():
        # verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    did_training = True

    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print("Command Line Args:", args)
    timeout = datetime.timedelta(hours=2)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        timeout=timeout,
        args=(args,),
    )
