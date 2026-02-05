# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
adapted from
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import torch.distributed as dist
from detectron2.config import CfgNode as CN
from torch import nn
import shutil

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
from detectron2.data import MapDataset
from torch.utils.data.distributed import DistributedSampler

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
from src.dataset.ds import CTA_Dataset
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
            results_folder = Path("./results")
            dataset_folder = Path(cfg.DATA.DIR.VAL.SCAN_DIR).parent.name

            output_folder = (
                results_folder
                / dataset_folder
                / cfg.MODEL.NAME
                / f"inference_{get_inference_iters(cfg)}"
            )
            output_folder.mkdir(parents=True, exist_ok=True)
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
    def build_lr_scheduler(cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

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


def cleanup():
    dist.destroy_process_group()


def get_base_config():
    cfg = CN()

    cfg.MODEL = CN()
    cfg.MODEL.META_ARCHITECTURE = None
    cfg.MODEL.DEVICE = "cuda:0"
    cfg.SOLVER = CN()
    cfg.SOLVER.AMP = CN()
    cfg.TEST = CN()
    cfg.VERSION = 2
    cfg.SEED = None
    cfg.OUTPUT_DIR = None
    cfg.DATASETS = CN()
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER = CN()
    cfg.DATALOADER.NUM_WORKERS = None
    cfg.SOLVER.STEPS = None
    cfg.SOLVER.MAX_ITER = None
    cfg.SOLVER.CHECKPOINT_PERIOD = None
    cfg.SOLVER.CLIP_GRADIENTS = CN()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = None
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = None
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = None
    cfg.SOLVER.WEIGHT_DECAY = 1
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    return cfg


def setup(args):
    """
    Create configs and perform basic setups.
    """
    # for poly lr schedule
    cfg = get_base_config()
    add_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.EVAL_ONLY = args.eval_only
    cfg.RESUME = args.resume
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME)
    cfg.MODEL.PATH_WEIGHTS = os.path.join(
        "./model_weights", cfg.MODEL.NAME, f"model_{cfg.MODEL.WEIGHTS}.pth"
    )
    # cfg.freeze()
    # Setup logger for "mask_former" module

    return cfg


def test_if_result_exists(cfg):
    """
    Verifies whether the result already exists.
    """
    results_folder = Path("./results")
    dataset_folder = Path(cfg.DATA.DIR.VAL.SCAN_DIR).parent.name

    output_folder = (
        results_folder
        / dataset_folder
        / cfg.MODEL.NAME
        / f"inference_{get_inference_iters(cfg)}"
    )
    if (output_folder / "predict.csv").exists():

        return True
    else:
        return False


def build_optimizer(cfg, model):
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
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
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


def build_lr_scheduler( cfg, optimizer):
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


import logging

def train(cfg, args):
    global did_training
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="src")
    setup_data_catalog(cfg)
    seed_everything(cfg.SEED)

    if cfg.EVAL_ONLY:
        model = Trainer.build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.PATH_WEIGHTS, resume=cfg.RESUME
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=cfg.RESUME)

    last_checkpoint = cfg.OUTPUT_DIR + "/last_checkpoint"
    is_resume = os.path.exists(last_checkpoint)

    if cfg.MODEL.TRANS_MODEL.USE_PRETRAINED_ENCODER == True and not is_resume:
        path_weights = cfg.MODEL.TRANS_MODEL.PRETRAINED_ENCODER_PATH
        encoder_weights = torch.load(path_weights)["model"]
        # rename all weights by prepending "module.backbone." to the key
        # also print all weight names
        encoder_weights = {
            f"module.backbone.{k.replace('model.module.','')}": v
            for k, v in encoder_weights.items()
        }

        # load all comaptible weights into trainer.model
        model_dict = trainer.model.state_dict()

        encoder_dict = {k: v for k, v in encoder_weights.items() if k in model_dict}
        # print all compatible weights and non-comaptible ones
        print("Loading encoder weights...\n\n\n")
        print("Compatible weights: ", encoder_dict.keys())
        print("\n")
        print(
            "Non-compatible weights: ",
            set(encoder_weights.keys()) - set(model_dict.keys()),
        )
        model_dict.update(encoder_dict)
        trainer.model.load_state_dict(model_dict)
        # TODO: fix resume for this case

    did_training = True

    return trainer.train()


def collate_fn(batch):
    return batch


def setup_multigpu():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MAIN_PROCESS_PORT"] = "12356"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # initialize the process group
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return rank, world_size


def setup_logger(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

def train(cfg):

    seed_everything(cfg.SEED)
    logger = setup_logger(cfg)
    rank, world_size = setup_multigpu()
    device_id = rank % torch.cuda.device_count()
    cfg.MODEL.DEVICE = f"cuda:{device_id}"
    ds = CTA_Dataset(cfg, "train")
    
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, shuffle=True)
        if world_size > 1
        else None
    )

    data_loader = torch.utils.data.DataLoader(
        ds,
        sampler=sampler,
        batch_size=cfg.SOLVER.SCANS_PER_BATCH // world_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda x: x,
    )
    model = build_model(cfg).cuda()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[cfg.MODEL.DEVICE], find_unused_parameters=False
    )
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    step = 0

    stats = {
        "step": [],
        "total_loss": [],
        "lr": [],
    }

    if cfg.RESUME:
        step, model, optimizer, lr_scheduler, stats = load_checkpoint(cfg, model, optimizer, lr_scheduler, stats)

    while step < cfg.SOLVER.MAX_ITER:
        for batch in data_loader:
            step += 1
            if step > cfg.SOLVER.MAX_ITER:
                break
            optimizer.zero_grad()
            loss_dict = model.forward(batch)
            loss_dict["total_loss"].backward()
            optimizer.step()
            lr_scheduler.step()
            dist.barrier()
            print_stats(logger, stats, step, loss_dict, optimizer.param_groups[0]["lr"], rank)
            if rank == 0 and (step + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                write_checkpoint(cfg, model, optimizer, lr_scheduler, step, stats)
                logger.info(f"Checkpoint saved at step {step}")

    cleanup()



def infer(cfg):

    logger = setup_logger(cfg)
    rank, world_size = setup_multigpu()
    device_id = rank % torch.cuda.device_count()
    cfg.MODEL.DEVICE = f"cuda:{device_id}"
    ds = CTA_Dataset(cfg, "val")
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, shuffle=False)
        if world_size > 1
        else None
    )

    data_loader = torch.utils.data.DataLoader(
        ds,
        sampler=sampler,
        batch_size=world_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda x: x,
    )
    model = build_model(cfg).cuda()

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[cfg.MODEL.DEVICE], find_unused_parameters=False
    )
    model = load_checkpoint_infer(cfg, model)
    model.module.training = False 
    output_folder = get_output_folder_infer(cfg)
    outputs = []
    total_samples = len(data_loader.dataset)
    samples_done = 0
    evaluator = CTAEvaluator(cfg, "eval", output_folder, distributed=True)
    for batch in data_loader:
        output = model.forward(batch)
        outputs.append(output)
        samples_done += 1
        evaluator.process(batch, output)
        print(f"Done with {samples_done} / {total_samples} samples")
        break
    
    evaluator.evaluate()
    cleanup()


def get_output_folder_infer(cfg):
    results_folder = Path("./results")
    dataset_folder = Path(cfg.DATA.DIR.VAL.SCAN_DIR).parent.name

    output_folder = (
        results_folder
        / dataset_folder
        / cfg.MODEL.NAME
        / f"inference_{get_inference_iters(cfg)}"
    )
    return output_folder 
import json

def write_checkpoint(cfg, model, optimizer, lr_scheduler, step, stats):
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"model_{str(step).zfill(7)}.pth" 
    # save model, optimizer and lr_scheduler state dicts in a single file
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "step": step,
        "stats": stats,
    }

    torch.save(checkpoint, checkpoint_path)
    shutil.copy(checkpoint_path, output_dir / "last_checkpoint")
    with open(output_dir / 'metrics.json', 'w') as fp:
        json.dump(stats, fp)


def load_checkpoint(cfg, model, optimizer, lr_scheduler, stats):
    output_dir = Path(cfg.OUTPUT_DIR)
    last_checkpoint = output_dir / "last_checkpoint"
    if not last_checkpoint.exists():
        return 0, model, optimizer, lr_scheduler, stats

    checkpoint = torch.load(last_checkpoint, map_location=cfg.MODEL.DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    step = checkpoint["step"] + 1
    stats = checkpoint["stats"]

    return step, model, optimizer, lr_scheduler, stats

def load_checkpoint_infer(cfg, model):
    output_dir = Path(cfg.OUTPUT_DIR)
    checkpoint = output_dir / f"model_{cfg.MODEL.WEIGHTS}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint}")

    checkpoint = torch.load(checkpoint, map_location=cfg.MODEL.DEVICE)
    model.load_state_dict(checkpoint["model"])    
    return model

def print_stats(logger, stats, step, loss_dict, lr, rank):
    # gather total loss value from all ranks

    total_loss = loss_dict["total_loss"].detach()
    dist.reduce(total_loss, 0, op=dist.ReduceOp.AVG, async_op=True).wait()
    if rank == 0:
        loss = total_loss.item()
        stats["total_loss"].append(loss)
        stats["step"].append(step)
        stats["lr"].append(lr)
        if step > 0 and step % 20 == 0:
            loss_stat = np.mean(np.array(stats["total_loss"][-20:]))
            lr_stat = np.mean(np.array(stats["lr"][-20:]))
            logger.info(f"iter: {step} | lr = {lr_stat} | total_loss = {loss_stat:.4f}")


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    if cfg.EVAL_ONLY:
        if test_if_result_exists(cfg):
            print(f"Inference run for {cfg.MODEL.WEIGHTS} done previously, aborting.")
        else:
            infer(cfg)
        sys.exit(0)
    train(cfg)
