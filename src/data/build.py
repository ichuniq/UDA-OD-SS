"""Build the train and testing data loader."""
import numpy as np
import torch.utils.data
from tabulate import tabulate
from termcolor import colored

from detectron2.data import (
    load_proposals_into_dataset,
    DatasetCatalog,
    MetadataCatalog,
    get_detection_dataset_dicts,
)
from detectron2.data.build import (
    build_batch_data_loader,
)
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.utils.env import seed_all_rng
from detectron2.data.samplers import (
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.utils.comm import get_world_size

import bisect
import copy
import itertools
import json
import logging
import pickle

from .pair_sampler import PairTrainingSampler, PairDataLoader

"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_detection_train_loader",
]


def build_detection_train_loader(cfg, mapper=None, domain='source'):
    """
    A data loader is created by the following steps:
    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.
    The batched ``list[mapped_dict]`` is what this dataloader will yield.
    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be ``DatasetMapper(cfg, True)``.
    Returns:
        an infinite iterator of training data
    """
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    if cfg.DATASETS.CROSS_DOMAIN:
        dataset_train = cfg.DATASETS.TRAIN_SOURCE if domain =='source' else cfg.DATASETS.TRAIN_TARGET
    else:
        dataset_train = cfg.DATASETS.TRAIN
    
    dataset_dicts = get_detection_dataset_dicts(
        # cfg.DATASETS.TRAIN,
        dataset_train,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)
    # print("check dataset")
    # d=dataset[0]
    # print(d.keys())
    # print("====================")
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)
    # print("check dataset after mapping")
    # d=dataset[0]
    # print(d.keys())
    # print("====================")
    # print(d)
    # print(d['image'].size())
    
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    elif sampler_name == "PairTrainingSampler":
        sampler = PairTrainingSampler(cfg, dataset_dicts, images_per_worker)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )
        return PairDataLoader(cfg, data_loader)

    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
