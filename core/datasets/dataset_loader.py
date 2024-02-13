import datetime
import typing
from random import random

from datasets import load_dataset, Dataset, Split, DatasetDict
from typing import List, Iterator, Union, Callable, Any

from tokenizer.core.utilities.merger import Merger


class DatasetLoader:
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size

    def load_hf_dataset(self, dataset_name: str, name: typing.Optional[str] = None, split: Union[str, Split] = "train", shuffle: bool = True, select_column: str = "text", rename_to: str = None) -> DatasetDict:
        dataset = load_dataset(dataset_name, name=name, split=split)
        dataset = dataset.select_columns(select_column)

        if rename_to is not None:
            dataset = dataset.rename_column(select_column, rename_to)

        if shuffle:
            dataset = dataset.shuffle(seed=int(datetime.datetime.now().timestamp()))
        return dataset

    def load_custom_dataset(self, generator_function: Callable[..., Any], shuffle: bool = True) -> Iterator:
        dataset = Dataset.from_generator(generator_function)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
        return dataset

    def merge_datasets(self, *datasets: Iterator, merger: typing.Callable = Merger.shuffled_stream_merged_datasets) -> Iterator:
        return merger(*datasets)