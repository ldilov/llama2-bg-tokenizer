from typing import Iterator, Union
from datasets import Dataset
from tokenizer.core.parsers.text_parser import TextParser


def dataset_iterator(dataset: Union[Dataset, Iterator], alphabet: str) -> Iterator[str]:
    parser = TextParser()
    for item in dataset:
        txt = str(item)
        txt = parser.parse(txt)

        if txt and len(txt.strip()):
            yield txt