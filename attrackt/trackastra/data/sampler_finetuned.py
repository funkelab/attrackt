""" "
Sampler utils for normal and distributed training
(e.g. to balance batch size).

"""

import logging
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
)

from .data import CTCData

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BalancedBatchSamplerFinetuned(BatchSampler):
    """samples batch indices such that the number of objects in each batch is balanced
    (so to reduce the number of paddings in the batch).


    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        n_pool: int = 10,
        num_samples: int | None = None,
        weight_by_ndivs: bool = False,
        weight_by_dataset: bool = False,
        drop_last: bool = False,
    ):
        """Setting n_pool =1 will result in a regular random batch sampler.

        weight_by_ndivs: if True, the probability of sampling an element is proportional to the number of divisions
        weight_by_dataset: if True, the probability of sampling an element is inversely proportional to the length of the dataset
        """
        if isinstance(dataset, CTCData):
            self.n_objects = dataset.n_objects
            self.n_divs = np.array(dataset.n_divs)
            self.n_sizes = np.ones(len(dataset)) * len(dataset)
        elif isinstance(dataset, ConcatDataset):
            self.n_objects = tuple(n for d in dataset.datasets for n in d.n_objects)
            self.n_divs = np.array(tuple(n for d in dataset.datasets for n in d.n_divs))
            self.n_sizes = np.array(
                tuple(len(d) for d in dataset.datasets for _ in range(len(d)))
            )
        else:
            raise NotImplementedError(
                f"BalancedBatchSampler: Unknown dataset type {type(dataset)}"
            )
        assert len(self.n_objects) == len(self.n_divs) == len(self.n_sizes)

        self.batch_size = batch_size
        self.n_pool = n_pool
        self.drop_last = drop_last
        self.num_samples = num_samples
        self.weight_by_ndivs = weight_by_ndivs
        self.weight_by_dataset = weight_by_dataset
        logger.debug(f"{weight_by_ndivs=}")
        logger.debug(f"{weight_by_dataset=}")

    def get_probs(self, idx):
        idx = np.array(idx)
        if self.weight_by_ndivs:
            probs = 1 + np.sqrt(self.n_divs[idx])
        else:
            probs = np.ones(len(idx))
        if self.weight_by_dataset:
            probs = probs / (self.n_sizes[idx] + 1e-6)

        probs = probs / (probs.sum() + 1e-10)
        return probs

    def sample_batches(self, idx: Iterable[int]):
        # we will split the indices into pools of size n_pool
        num_samples = self.num_samples if self.num_samples is not None else len(idx)
        # sample from the indices with replacement and given probabilites
        idx = np.random.choice(idx, num_samples, replace=True, p=self.get_probs(idx))

        n_pool = min(
            self.n_pool * self.batch_size,
            (len(idx) // self.batch_size) * self.batch_size,
        )

        batches = []
        for i in range(0, len(idx), n_pool):
            # the indices in the pool are sorted by their number of objects
            idx_pool = idx[i : i + n_pool]
            idx_pool = sorted(idx_pool, key=lambda i: self.n_objects[i])

            # such that we can create batches where each element has a similar number of objects
            jj = np.arange(0, len(idx_pool), self.batch_size)
            np.random.shuffle(jj)

            for j in jj:
                # dont drop_last, as this leads to a lot of lightning problems....
                # if j + self.batch_size > len(idx_pool):  # assume drop_last=True
                #     continue
                batch = idx_pool[j : j + self.batch_size]
                batches.append(batch)
        return batches

    def __iter__(self):
        idx = np.arange(len(self.n_objects))
        batches = self.sample_batches(idx)
        return iter(batches)

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples // self.batch_size
        else:
            return len(self.n_objects) // self.batch_size
