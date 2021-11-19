from torch._C import Value
from kge import Config, Configurable, Dataset
from kge.indexing import where_in

import random
import torch
from typing import Optional
import numpy as np
import numba
import time

from kge.temporal_dataset import TemporalDataset

SLOTS = [0, 1, 2, 3]
SLOT_STR = ["s", "p", "o", "t"]
S, P, O, T = SLOTS


class TemporalKgeSampler(Configurable):
    """Negative sampler."""

    def __init__(
        self, config: Config, configuration_key: str, dataset: TemporalDataset
    ):
        super().__init__(config, configuration_key)

        # load config
        self.num_samples = torch.zeros(4, dtype=torch.int)
        self.filter_positives = torch.zeros(4, dtype=torch.bool)
        self.vocabulary_size = torch.zeros(4, dtype=torch.int)
        self.shared = self.get_option("shared")
        self.shared_type = self.check_option("shared_type", ["naive", "default"])
        self.with_replacement = self.get_option("with_replacement")
        if not self.with_replacement and not self.shared:
            raise ValueError(
                "Without replacement sampling is only supported when "
                "shared negative sampling is enabled."
            )
        self.filtering_split = config.get("negative_sampling.filtering.split")
        if self.filtering_split == "":
            self.filtering_split = config.get("train.split")
        for slot in SLOTS:
            slot_str = SLOT_STR[slot]
            self.num_samples[slot] = self.get_option(f"num_samples.{slot_str}")
            self.filter_positives[slot] = self.get_option(f"filtering.{slot_str}")
            self.vocabulary_size[slot] = (
                dataset.num_relations()
                if slot == P
                else dataset.num_timestamps()
                if slot == T
                else dataset.num_entities()
            )
            # create indices for filtering here already if needed and not existing
            # otherwise every worker would create every index again and again
            if self.filter_positives[slot]:
                raise ValueError
        if any(self.filter_positives):
            raise ValueError
        self.dataset = dataset
        # auto config
        for slot, copy_from in [(S, O), (P, None), (O, S)]:
            if self.num_samples[slot] < 0:
                raise ValueError

    @staticmethod
    def create(
        config: Config, configuration_key: str, dataset: TemporalDataset
    ) -> "TemporalKgeSampler":
        """Factory method for sampler creation."""
        sampling_type = config.get(configuration_key + ".sampling_type")
        if sampling_type == "uniform":
            return KgeUniformSampler(config, configuration_key, dataset)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(configuration_key + ".sampling_type")

    def sample(
        self,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: Optional[int] = None,
    ) -> "BatchNegativeSample":
        """Obtain a set of negative samples for a specified slot.

        `positive_triples` is a batch_size x 3 tensor of positive triples. `slot` is
        either 0 (subject), 1 (predicate), or 2 (object). If `num_samples` is `None`,
        it is set to the default value for the slot configured in this sampler.

        Returns a `BatchNegativeSample` data structure that allows to retrieve or score
        all negative samples. In the simplest setting, this data structure holds a
        batch_size x num_samples tensor with the negative sample indexes (see
        `DefaultBatchNegativeSample`), but more efficient approaches may be used by
        certain samplers.

        """
        if num_samples is None:
            num_samples = self.num_samples[slot].item()

        if self.shared:
            # for shared sampling, we do not post-process; return right away
            return self._sample_shared(positive_triples, slot, num_samples)
        else:
            negative_samples = self._sample(positive_triples, slot, num_samples)

        # for non-shared smaples, we filter the positives (if set in config)
        if self.filter_positives[slot]:
            raise NotImplementedError

        return DefaultBatchNegativeSample(
            self.config,
            self.configuration_key,
            positive_triples,
            slot,
            num_samples,
            negative_samples,
        )

    def _sample(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ) -> torch.Tensor:
        raise NotImplementedError

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ) -> "BatchNegativeSample":
        raise NotImplementedError

    def _filter_and_resample(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class BatchNegativeSample(Configurable):
    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
    ):
        super().__init__(config, configuration_key)
        self.positive_triples = positive_triples
        self.slot = slot
        self.num_samples = num_samples
        self._implementation = self.check_option(
            "implementation", ["triple", "batch", "all"]
        )
        self.forward_time = 0.0
        self.prepare_time = 0.0

    def samples(self, indexes=None) -> torch.Tensor:
        raise NotImplementedError

    def unique_samples(self, indexes=None, return_inverse=False):
        samples = self.samples(indexes)
        return torch.unique(samples.view(-1), return_inverse=return_inverse)

    def to(self, device) -> "BatchNegativeSample":
        self.positive_triples = self.positive_triples.to(device)
        return self

    def score(self, model, indexes=None) -> torch.Tensor:
        self.forward_time = 0.0
        self.prepare_time = 0.0

        # the default implementation here is based on the set of all samples as provided
        # by self.samples(); get the relavant data
        slot = self.slot
        self.prepare_time -= time.time()
        negative_samples = self.samples(indexes)
        num_samples = self.num_samples
        triples = (
            self.positive_triples[indexes, :] if indexes else self.positive_triples
        )
        self.prepare_time += time.time()

        # go ahead and score
        device = self.positive_triples.device
        scores = None
        if self._implementation == "triple":
            # construct triples
            self.prepare_time -= time.time()
            triples_to_score = triples.repeat(1, num_samples).view(-1, 4)
            triples_to_score[:, slot] = negative_samples.contiguous().view(-1)
            self.prepare_time += time.time()

            # and score them
            self.forward_time -= time.time()
            chunk_size = len(negative_samples)
            scores = model.score_spo(
                triples_to_score[:, S],
                triples_to_score[:, P],
                triples_to_score[:, O],
                triples_to_score[:, T],
            ).view(chunk_size, -1)
            self.forward_time += time.time()
        else:
            raise ValueError

        return scores

    @staticmethod
    def _score_unique_targets(model, slot, triples, unique_targets) -> torch.Tensor:
        raise NotImplementedError


class DefaultBatchNegativeSample(BatchNegativeSample):
    """Default implementation that stores all negative samples as a tensor."""

    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
        samples: torch.Tensor,
    ):
        super().__init__(config, configuration_key, positive_triples, slot, num_samples)
        self._samples = samples

    def samples(self, indexes=None) -> torch.Tensor:
        return self._samples if indexes is None else self._samples[indexes]

    def to(self, device) -> "DefaultBatchNegativeSample":
        super().to(device)
        self._samples = self._samples.to(device)
        return self


class KgeUniformSampler(TemporalKgeSampler):
    def __init__(self, config: Config, configuration_key: str, dataset: Dataset):
        super().__init__(config, configuration_key, dataset)

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        return torch.randint(
            self.vocabulary_size[slot], (positive_triples.size(0), num_samples)
        )

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        raise NotImplementedError

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ):
        raise NotImplementedError

    @numba.njit
    def _filter_and_resample_numba(
        negative_samples, pairs, positives_index, batch_size, voc_size
    ):
        raise NotImplementedError
