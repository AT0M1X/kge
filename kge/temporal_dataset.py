from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.tensor import Tensor
from kge.config import Config
from kge.dataset import Dataset
from kge.misc import module_base_dir


class TemporalDataset(Dataset):
    def __init__(self, config, folder=None):
        super().__init__(config, folder=folder)

        # read number of timestamps from config if present
        try:
            self._num_timestamps: int = config.get("dataset.num_timestamps")
            if self._num_timestamps < 0:
                self._num_timestamps = None
        except KeyError:
            self._num_timestamps: int = None

    @staticmethod
    def create(
        config: Config, preload_data: bool = True, folder: Optional[str] = None
    ) -> TemporalDataset:
        """Loads a dataset.

        If preload_data is set, loads entity, relation and time maps as well as all splits.
        Otherwise, this data is lazy loaded on first use.
        """
        name = config.get("dataset.name")

        root_modules = list({m.split(".")[0] for m in config.get("modules")})
        if folder is None:
            for m in root_modules:
                folder = os.path.join(module_base_dir(m), "data", name)
                if os.path.isfile(os.path.join(folder, "dataset.yaml")):
                    break
            else:
                raise ValueError(f"Dataset with name {name} could not be found.")

        config.log(f"Loading configuration of dataset {name} from {folder} ...")
        config.load(os.path.join(folder, "dataset.yaml"))

        dataset = TemporalDataset(config, folder)
        if preload_data:
            dataset.entity_ids()
            dataset.relation_ids()
            dataset.timestamp_ids()
            for split in ["train", "valid", "test"]:
                dataset.split(split)
        return dataset

    @staticmethod
    def create_from(
        checkpoint: Dict,
        config: Config = None,
        dataset: Optional[Dataset] = None,
        preload_data=False,
    ) -> TemporalDataset:
        """Creates dataset based on a checkpoint.

        If a dataset is provided, only (!) its meta data will be updated with the values
        from the checkpoint. No further checks are performed.

        Args:
            checkpoint: loaded checkpoint
            config: config (should match the one of checkpoint if set)
            dataset: dataset to update
            preload_data: preload data

        Returns: created/updated dataset
        """
        if config is None:
            config = Config.create_from(checkpoint)
        if dataset is None:
            dataset = TemporalDataset.create(config, preload_data)
        if "dataset" in checkpoint:
            dataset_checkpoint = checkpoint["dataset"]
            if (
                "dataset.meta" in dataset_checkpoint
                and dataset_checkpoint["meta"] is not None
            ):
                dataset._meta.update(dataset_checkpoint["meta"])
            dataset._num_entities = dataset_checkpoint["num_entities"]
            dataset._num_relations = dataset_checkpoint["num_relations"]
            # TODO num_times
        return dataset

    def save_to(self, checkpoint: Dict, meta_keys: Optional[List[str]] = None) -> Dict:
        checkpoint = super().save_to()
        checkpoint["num_timestamps"] = self.num_timestamps
        return checkpoint

    @staticmethod
    def _load_triples(filename: str, delimiter="\t", use_pickle=False) -> Tensor:
        """Actually quadruples, but to avoid changing too much code, we don't change it."""
        if use_pickle:
            # check if there is a pickled, up-to-date version of the file
            pickle_suffix = Dataset._to_valid_filename(f"-{delimiter}.pckl")
            pickle_filename = filename + pickle_suffix
            quadruples = Dataset._pickle_load_if_uptodate(
                None, pickle_filename, filename
            )
            if quadruples is not None:
                return quadruples

        # numpy loadtxt is very slow, use pandas instead
        quadruples = pd.read_csv(
            filename, sep=delimiter, dtype=np.int32, header=None, usecols=range(4)
        ).to_numpy()

        quadruples = torch.from_numpy(quadruples)
        if use_pickle:
            Dataset._pickle_dump_atomic(quadruples, pickle_filename)
        return quadruples

    def load_triples(self, key: str) -> Tensor:
        """Load or return the triples with the specified key. Actually quadruples."""
        if key not in self._triples:
            self.ensure_available(key)
            filename = self.config.get(f"dataset.files.{key}.filename")
            filetype = self.config.get(f"dataset.files.{key}.type")
            if filetype != "triples":
                raise ValueError(
                    "Unexpected file type: "
                    f"dataset.files.{key}.type='{filetype}', expected 'triples'"
                )
            triples = TemporalDataset._load_triples(
                os.path.join(self.folder, filename),
                use_pickle=self.config.get("dataset.pickle"),
            )
            self.config.log(f"Loaded {len(triples)} {key} triples")
            self._triples[key] = triples

        return self._triples[key]

    def shallow_copy(self):
        """Returns a dataset that shares the underlying splits and indexes.

        Changes to splits and indexes are also reflected on this and the copied dataset.
        """
        copy = TemporalDataset(self.config, self.folder)
        copy._num_entities = self.num_entities()
        copy._num_relations = self.num_relations()
        copy._num_timestamps = self.num_timestamps()
        copy._triples = self._triples
        copy._meta = self._meta
        copy._indexes = self._indexes
        copy.index_functions = self.index_functions
        return copy

    def index(self, key: str) -> Any:
        raise NotImplementedError(
            "Something tried to use the index method on TemporalDataset."
        )

    def timestamp_ids(
        self, indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode indexes to time_ids.

        See `Dataset#map_indexes` for a description of the `indexes` argument.
        """
        return self.map_indexes(indexes, "time_ids")

    def num_timestamps(self) -> int:
        "Return the number of entities in this dataset."
        if not self._num_timestamps:
            self._num_timestamps = len(self.timestamp_ids())
        return self._num_timestamps
