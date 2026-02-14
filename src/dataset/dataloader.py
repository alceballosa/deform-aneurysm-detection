import itertools
import logging

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import (
    DatasetCatalog,
    DatasetFromList,
    MapDataset,
    MetadataCatalog,
    build_batch_data_loader,
)
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler, TrainingSampler
from detectron2.utils.logger import log_first_n
from tabulate import tabulate
from termcolor import colored


def _worker_init_fn(worker_id):
    """Ensure spawned dataloader workers inherit the file_system sharing strategy."""
    torch.multiprocessing.set_sharing_strategy("file_system")


def get_dataset_dicts(dataset_names):
    """
    Load and join classification dataset dicts

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names


    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    assert len(dataset_dicts), "No valid data found in {}.".format(
        ",".join(dataset_names)
    )
    return dataset_dicts


def _train_loader_from_config(cfg, mapper, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_dataset_dicts(cfg.DATASETS.TRAIN)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            # ! change shuffle later
            shuffle = True
            if cfg.CUSTOM.DEBUG:
                shuffle = False
            sampler = TrainingSampler(len(dataset), shuffle=shuffle)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.SCANS_PER_BATCH,
        "aspect_ratio_grouping": False,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)

    mp_context = "spawn" if num_workers > 0 else None
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        worker_init_fn=_worker_init_fn,
    )


def _test_loader_from_config(cfg, dataset_name, mapper):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    dataset = get_dataset_dicts([dataset_name])
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_test_loader_from_config)
def build_test_loader(dataset, *, mapper, num_workers=0):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    mp_context = "spawn" if num_workers > 0 else None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        pin_memory=True,
        multiprocessing_context=mp_context,
        worker_init_fn=_worker_init_fn,
    )
    return data_loader
