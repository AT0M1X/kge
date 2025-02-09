import argparse
import numpy as np
from os import path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Iterable
import yaml
import os

### DEFAULT PREPROCESS UTILS ##########################################################


@dataclass
class RawSplit:
    """Track data and meta-data of a dataset split.

    Attributes:
        file (str): File with tab-separated raw quadruples (can be labeled)
        collect_entities (bool): If true, entities contained in this split will be
            collected during preprocessing.
        collect_relations (bool): If true, relations contained in this split will be
            collected during preprocessing.
        field_map (dict): Mapping of "S", "P", "O" and "T" to the subject/predicate/object field
            in the raw quadruples.
        splits (list[Split]): List of Split's, i.e, the final
            splits that ultimately will be written from this split.
        data (list): List of quadruples represented with raw ids; created during
            preprocessing
        size (str): Number of quadruples; determined during preprocessing.
        entities (dict): Entities contained in this split if collected. Keys refer to
            raw id's and values to a dense index assigned during pre-processing.
        relations (dict): See entities.

    """

    # fields defined by user
    file: str
    collect_entities: bool = False
    collect_relations: bool = False
    collect_time: bool = False
    field_map: Dict[str, int] = None
    splits: List["Split"] = field(default_factory=list)

    # fields defined during pre-processing
    data: List[str] = None
    size: int = None
    entities: Dict = field(default_factory=dict)
    relations: Dict = field(default_factory=dict)
    time: Dict = field(default_factory=dict)

    def write_splits(self, entities: dict, relations: dict, times: dict, folder):
        for split in self.splits:
            split.prepare(folder)
        for n, t in enumerate(self.data):
            for split in self.splits:
                split.process_quadruple(t, entities, relations, times, n=n)
        for split in self.splits:
            split.file.close()

    def update_config(self, config: Dict) -> Dict:
        """Update a dataset config with meta data of the derived splits."""
        for split in self.splits:
            for key, val in split.options.items():
                config[f"files.{split.key}.{key}"] = val
        return config


@dataclass
class Split:
    """The final libKGE split derived and written from a base RawSplit.

    Attributes:
        raw_split (RawSplit): The parent split.
        key (str): The key in the dataset.yaml file.
        options (dict): Arbitrary dict of options. Key_ value_ pairs will be added to
            to dataset.yaml under the respective key entry of the split.

    """

    raw_split: RawSplit = None
    key: str = None
    options: Dict = None

    def prepare(self, folder: str):
        self.file = open(path.join(folder, self.options["filename"]), "w")
        self.options["size"] = 0

    def process_quadruple(
        self, quadruple: list, entities: dict, relations: dict, times: dict, **kwargs
    ):
        write_quadruple(
            self.file,
            entities,
            relations,
            times,
            quadruple,
            self.raw_split.field_map["S"],
            self.raw_split.field_map["P"],
            self.raw_split.field_map["O"],
            self.raw_split.field_map["T"],
        )
        self.options["size"] += 1


@dataclass
class FilteredSplit(Split):
    """A filtered derived split.

    Attributes:
        filter_with (RawSplit): The RawSplit of which entities and relations shall be taken
            for filtering. The derived split exclusively contains quadruples where
            entities and relations are known in the filter_with RawSplit.
    """

    filter_with: RawSplit = None

    def process_quadruple(
        self, quadruple: list, entities: dict, relations: dict, times: dict, **kwargs
    ):
        S, P, O, T = (
            self.raw_split.field_map["S"],
            self.raw_split.field_map["P"],
            self.raw_split.field_map["O"],
            self.raw_split.field_map["T"],
        )
        if (
            quadruple[S] in self.filter_with.entities
            and quadruple[O] in self.filter_with.entities
            and quadruple[P] in self.filter_with.relations
            and quadruple[T] in self.filter_with.time
        ):
            super().process_quadruple(quadruple, entities, relations, times, **kwargs)


@dataclass
class SampledSplit(Split):
    """A derived sub-sample RawSplit.

    Attributes:
        sample_size (int): Size of the subsample.
        sample (Iterable[int]): Randomly selected quadruple indexes with size
            sample_size; determined in  prepare().

    """

    sample_size: int = None
    sample: Iterable[int] = None

    def prepare(self, folder: str):
        super().prepare(folder)
        self.sample = np.random.choice(
            len(self.raw_split.data), self.sample_size, False
        )

    def process_quadruple(
        self, quadruple: list, entities: dict, relations: dict, times: dict, **kwargs
    ):
        if kwargs["n"] in self.sample:
            super().process_quadruple(quadruple, entities, relations, times, **kwargs)


@dataclass
class RawDataset:
    """A raw relational dataset.

    Contains the RawSplits of the dataset to be processed and the final config;
    is generated automatically in analyze_raw_splits().

    Attributes:
        raw_splits (list[RawSplit]): List of Splits.
        entity_map (dict): Distinct entities over all splits. Keys refer to
           raw entity-id's and values to the dense index.
        relation_map (dict): See all entities.
        config (dict): Raw dictionary holding the dataset config options.
        folder (str): Path to the dataset folder.

    """

    raw_splits: List[RawSplit]
    entity_map: Dict[str, int]
    relation_map: Dict[str, int]
    time_map: Dict[str, int]
    config: Dict[str, str]
    folder: str


def process_splits(raw_dataset: RawDataset):
    for raw_split in raw_dataset.raw_splits:
        raw_split.write_splits(
            entities=raw_dataset.entity_map,
            relations=raw_dataset.relation_map,
            times=raw_dataset.time_map,
            folder=raw_dataset.folder,
        )
        # add the collected meta data to the config
        raw_split.update_config(raw_dataset.config)


def analyze_raw_splits(raw_splits: List[RawSplit], folder: str) -> RawDataset:
    """Read a collection of raw splits and create a RawDataset.

    Args:
        raw_splits (list[Splits]): List of RawSplits.
        folder (str): Folder of the raw_dataset containing the files.
    """
    # read data and collect entities and relations
    entity_map: Dict[str, int] = {}
    relation_map: Dict[str, int] = {}
    time_map: Dict[str, int] = {}
    for raw_split in raw_splits:
        with open(path.join(folder, raw_split.file), "r") as f:
            raw_split.data = list(map(lambda s: s.strip().split("\t"), f.readlines()))
            S, P, O, T = (
                raw_split.field_map["S"],
                raw_split.field_map["P"],
                raw_split.field_map["O"],
                raw_split.field_map["T"],
            )
            for q in raw_split.data:
                if q[S] not in entity_map:
                    entity_map[q[S]] = len(entity_map)
                if q[P] not in relation_map:
                    relation_map[q[P]] = len(relation_map)
                if q[O] not in entity_map:
                    entity_map[q[O]] = len(entity_map)
                if q[T] not in time_map:
                    time_map[q[T]] = len(time_map)
                if raw_split.collect_entities:
                    raw_split.entities[q[S]] = entity_map[q[S]]
                    raw_split.entities[q[O]] = entity_map[q[O]]
                if raw_split.collect_relations:
                    raw_split.relations[q[P]] = relation_map[q[P]]
                if raw_split.collect_time:
                    raw_split.time[q[T]] = time_map[q[T]]
            raw_split.size = len(raw_split.data)
            print(f"Found {raw_split.size} quadruples in {raw_split.file}")
    print(f"{len(relation_map)} distinct relations")
    print(f"{len(entity_map)} distinct entities")
    print(f"{len(time_map)} distinct timestamps")

    config = dict(
        name=folder,
        num_entities=len(entity_map),
        num_relations=len(relation_map),
        num_timestamps=len(time_map),
    )

    raw_dataset = RawDataset(
        raw_splits=raw_splits,
        entity_map=entity_map,
        relation_map=relation_map,
        time_map=time_map,
        config=config,
        folder=folder,
    )

    # write entity/relation/time maps and update config
    write_maps(raw_dataset)

    return raw_dataset


def write_maps(raw_dataset: RawDataset):
    """Write entity, relation and time maps and update config with respective keys."""
    print("Writing relation, entity and time maps...")
    store_map(
        raw_dataset.relation_map, path.join(raw_dataset.folder, "relation_ids.del")
    )
    store_map(raw_dataset.entity_map, path.join(raw_dataset.folder, "entity_ids.del"))
    store_map(raw_dataset.time_map, path.join(raw_dataset.folder, "time_ids.del"))
    for obj in ["entity", "relation", "time"]:
        raw_dataset.config[f"files.{obj}_ids.filename"] = f"{obj}_ids.del"
        raw_dataset.config[f"files.{obj}_ids.type"] = "map"


def store_map(symbol_map: Dict, filename: str):
    """Write a map file."""
    with open(filename, "w") as f:
        for symbol, index in symbol_map.items():
            f.write(f"{index}\t{symbol}\n")


def write_quadruple(f, ent, rel, tim, q, S, P, O, T):
    """Write a quadruple to a file."""
    f.write(
        str(ent[q[S]])
        + "\t"
        + str(rel[q[P]])
        + "\t"
        + str(ent[q[O]])
        + "\t"
        + str(tim[q[T]])
        + "\n"
    )


def write_dataset_yaml(config: Dict, folder: str):
    """Write a dataset.yaml file given a config dictionary and a folder path."""
    print(yaml.dump(dict(dataset=config)))
    with open(path.join(folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=config)))


@dataclass
class LabeledSplit(Split):
    label: int = None

    def process_triple(self, triple, entities, relations, **kwargs):
        if int(triple[3]) == self.label:
            super().process_quadruple(triple, entities, relations, **kwargs)


@dataclass
class FilteredLabeledSplit(FilteredSplit):
    label: int = None

    def process_triple(self, triple: List, entities: Dict, relations: Dict, **kwargs):
        if int(triple[3]) == self.label:
            super().process_quadruple(triple, entities, relations, **kwargs)


# default CLI
def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--subject-field", "-S", action="store", default=0, type=int)
    parser.add_argument("--predicate-field", "-P", action="store", default=1, type=int)
    parser.add_argument("--object-field", "-O", action="store", default=2, type=int)
    parser.add_argument("--time-field", "-T", action="store", default=3, type=int)
    return parser


def create_raw_dataset(
    train_raw, valid_raw, test_raw, args, create_splits=True
) -> RawDataset:
    # read data and collect entity and relation maps
    raw_dataset: RawDataset = analyze_raw_splits(
        raw_splits=[train_raw, valid_raw, test_raw],
        folder=args.folder,
    )

    if create_splits:
        # register all splits to be derived from the raw splits
        # arbitrary options may be added to the raw_dataset config in the process
        train = Split(
            raw_split=train_raw,
            key="train",
            options={
                "type": "triples",
                "filename": "train.del",
                "split_type": "train",
            },
        )
        train_sample = SampledSplit(
            raw_split=train_raw,
            key="train_sample",
            sample_size=len(valid_raw.data),
            options={
                "type": "triples",
                "filename": "train_sample.del",
                "split_type": "train",
            },
        )
        train_raw.splits.extend([train, train_sample])

        valid = Split(
            raw_split=valid_raw,
            key="valid",
            options={
                "type": "triples",
                "filename": "valid.del",
                "split_type": "valid",
            },
        )

        valid_wo_unseen = FilteredSplit(
            raw_split=valid_raw,
            key="valid_without_unseen",
            filter_with=train_raw,
            options={
                "type": "triples",
                "filename": "valid_without_unseen.del",
                "split_type": "valid",
            },
        )
        valid_raw.splits.extend([valid, valid_wo_unseen])

        test = Split(
            raw_split=test_raw,
            key="test",
            options={
                "type": "triples",
                "filename": "test.del",
                "split_type": "test",
            },
        )
        test_wo_unseen = FilteredSplit(
            raw_split=test_raw,
            key="test_without_unseen",
            filter_with=train_raw,
            options={
                "type": "triples",
                "filename": "test_without_unseen.del",
                "split_type": "test",
            },
        )
        test_raw.splits.extend([test, test_wo_unseen])

    return raw_dataset


def update_string_files(raw_dataset: RawDataset, args):
    """update config with entity string files"""
    string_files = {
        "entity_strings": "entity_strings.del",
        "relation_strings": "relation_strings.del",
    }

    for string in string_files.keys():
        if os.path.exists(os.path.join(args.folder, string_files[string])):
            raw_dataset.config[f"files.{string}.filename"] = string_files.get(string)
            raw_dataset.config[f"files.{string}.type"] = "idmap"
