import datetime
import random
from itertools import chain
from typing import Iterable
from datasets import concatenate_datasets, Dataset, DatasetDict


class Merger(object):

    @staticmethod
    def stream_merged_datasets(*datasets: Iterable[Dataset]) -> Iterable[Dataset]:
        for sample in chain(*datasets):
            yield sample

    @staticmethod
    def shuffled_stream_merged_datasets(*datasets: Iterable[Dataset], buffer_size: int = 10000):
        buffer = []
        for row in Merger.stream_merged_datasets(*datasets):
                buffer.append(row)
                if len(buffer) >= buffer_size:
                    random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

        random.shuffle(buffer)
        for item in buffer:
            yield item

    @staticmethod
    def merge(*datasets: Iterable[Dataset], shuffle: bool = True, limit: int = 10000000) -> DatasetDict:
        concatenated = concatenate_datasets(list(datasets), axis=0)

        if shuffle:
            seed = int(round(datetime.datetime.now().timestamp()))
            concatenated = concatenated.shuffle(seed=seed)

        length = min(concatenated.num_rows, limit)
        concatenated = concatenated.select(range(length))

        return concatenated