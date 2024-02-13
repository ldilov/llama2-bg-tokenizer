import datetime
import gc
import json
import os
import typing
import tokenizers
from pathlib import Path
from transformers import LlamaTokenizerFast
from datasets import Dataset, DatasetDict
from typing import List, Union, Dict
from textdistance import levenshtein
from tokenizers import pre_tokenizers, normalizers, decoders, trainers
from tokenizers.decoders import Decoder
from tokenizers.models import BPE
from tokenizers.normalizers import Prepend, Replace, NFKC, Normalizer
from tokenizers.pre_tokenizers import Punctuation, PreTokenizer
from tokenizers.processors import TemplateProcessing, PostProcessor
from tokenizer.core.utilities.constants import ALPHABET
from tokenizer.core.datasets.dataset_iterator import dataset_iterator
from tokenizer.core.logger import logger
from tokenizer.core.timer import Timer
from tokenizer.core.tokenizer import GanioTokenizer
from tokenizer.core.utilities.merger import Merger
from tokenizer.core.utilities.statistics import analyze_dataset_for_dynamic_tokens, adjust_min_frequency
from tokenizer.core.utilities.validator.schema import schema


class TokenizerTrainer:
    def __init__(self, config: Dict[str, Union[str, int, Dict]]):
        self.__config = config
        self.__special_tokens = [tokenizers.AddedToken(**value) for key, value in
                                 self.__config['added_tokens_decoder'].items()]
        self.__replacement_char = "▁"
        self.__tokenizer = self._initialize_tokenizer()

        self.__date = datetime.datetime.now().strftime("%Y-%m-%d")

        self.__min_frequency = None
        self.__inner_trainer = None

        from tokenizer.core.utilities.validator.validate_config import validate_config
        validate_config(config, schema=schema)

    def _initialize_tokenizer(self) -> tokenizers.Tokenizer:
        bpe = self.__build_bpe()

        tokenizer = tokenizers.Tokenizer(bpe)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Punctuation()])
        tokenizer.normalizer = normalizers.Sequence([
            Prepend("▁"),
            Replace(r" ", "▁"),
            NFKC(),
        ])

        tokenizer.decoder = decoders.Sequence([
            decoders.Replace(self.__replacement_char, " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(" ", left=1)
        ])

        tokenizer.post_processor = TemplateProcessing(
            single=self.__config['template']['single'],
            pair=self.__config['template']['pair'],
            special_tokens=[(self.__config['bos_token'],
                             self.__config['additional_special_tokens'].index(self.__config['bos_token']))]
        )

        tokenizer.add_special_tokens(tokens=self.__special_tokens)

        return tokenizer

    def train(self, datasets: List[Union[Dataset, DatasetDict]], limit: int = 50000):
        """Train the tokenizer using the provided datasets."""
        self.__tokenizer.enable_padding(
            pad_token=self.__config['pad_token'],
            pad_type_id=self.__config['pad_type_id'],
            pad_to_multiple_of=self.__config['pad_to_multiple_of'],
            direction=self.__config['padding_side']
        )

        merged_dataset = Merger.merge(*datasets, limit=limit)

        cut_off = 5000
        if merged_dataset.num_rows <= cut_off:
            cut_off = int(merged_dataset.num_rows / 3)

        test_ds = merged_dataset['text'][:cut_off]
        train_ds = merged_dataset['text'][cut_off:]
        merged_dataset = None

        if self.__inner_trainer is None:
            self.__build_bpe_trainer(train_ds)

        gc.collect()

        with Timer("Training time:"):
            self.__tokenizer.train_from_iterator(trainer=self.__inner_trainer,
                                                 iterator=dataset_iterator(train_ds, ALPHABET))
            if test_ds:
                self.evaluate_tokenizer(test_ds)

    def evaluate_tokenizer(self, holdout_dataset: List[str]):
        # Make a temporary copy of the tokenizer to avoid disrupting current training
        temp_tokenizer = tokenizers.Tokenizer(self.__tokenizer.model)
        temp_tokenizer.normalizer = self.__tokenizer.normalizer
        temp_tokenizer.pre_tokenizer = self.__tokenizer.pre_tokenizer
        temp_tokenizer.decoder = self.__tokenizer.decoder
        temp_tokenizer.post_processor = self.__tokenizer.post_processor

        results = self.evaluate(temp_tokenizer, holdout_dataset)
        result_txt = f"Evaluation - Vocab Size: {results['vocab_size']}, Errors: {results['round_trip_errors']}, Loss: {results['loss']}"

        logs_path = Path(__file__).parent.parent / 'scripts' / 'logs' / f"eval_{self.__date}.log"
        logs_path.touch(exist_ok=True, mode=os.O_CREAT)

        try:
            logs_path.write_text(data=result_txt, encoding='utf-8', newline='\n', errors='ignore')
        except PermissionError:
            pass

        logger.info(result_txt)

    def evaluate(self, tokenizer, data):
        tokenized_samples = [tokenizer.encode(example).ids for example in data]

        round_trip_errors = 0
        distance = 0.0
        for original, tokens in zip(data, tokenized_samples):
            detokenized = tokenizer.decode(tokenizer.encode(original).ids)
            original_length = len(original)
            if original != detokenized and original_length > 0:
                round_trip_errors += 1
                distance += (levenshtein(original, detokenized) / original_length)

        loss = distance / round_trip_errors if round_trip_errors > 0 else 0.0

        results = {
            'round_trip_errors': round_trip_errors,
            'vocab_size': len(tokenizer.get_vocab()),
            'loss': loss
        }

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save the tokenizer and its configuration to the given path."""
        if not self.__tokenizer:
            raise ValueError("Tokenizer has not been trained.")

        model_file = str(Path(path))
        self.__save_to_file(model_file)
        file_tokenizer = self.__load_from_file(model_file)

        kwargs = {self.__config["special_tokens_attr"][int(key)]: value for key, value in
                  enumerate(self.__config['additional_special_tokens'])}
        kwargs["pad_token"] = self.__config["pad_token"]

        pretrained_model_path = str(Path(model_file).parent / "llama")
        pretrained_tokenizer_fast = LlamaTokenizerFast(tokenizer_object=file_tokenizer, **kwargs)
        pretrained_tokenizer_fast.save_pretrained(pretrained_model_path, legacy_format=False)

        pretrained_model_path = str(Path(model_file).parent / "ganio")
        pretrained_tokenizer_fast = GanioTokenizer(tokenizer_object=file_tokenizer, **kwargs)
        pretrained_tokenizer_fast.save_pretrained(pretrained_model_path, legacy_format=False)

    def __getattr__(self, name: str) -> typing.Any:
        """
        Forward attribute access to self.__tokenizer when an attribute
        is not found in the GanioTokenizer instance.
        """

        def method(*args, **kwargs):
            return getattr(self.__tokenizer, name)(*args, **kwargs)

        if hasattr(self.__tokenizer, name):
            return method
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @staticmethod
    def load(path: Union[Path, str], loader: typing.Callable = tokenizers.Tokenizer.from_file) -> 'GanioTokenizer':
        """Load a tokenizer and its configuration from the given path."""

        tokenizer = loader(str(Path(path) / "tokenizer.json"))

        with open(str(Path(path) / "tokenizer_config.json"), 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)

        custom_tokenizer = GanioTokenizer(config)
        custom_tokenizer.__tokenizer = tokenizer

        return custom_tokenizer

    def __load_from_file(self, path: Union[Path, str]) -> tokenizers.Tokenizer:
        file_tokenizer = tokenizers.Tokenizer.from_file(str(path))
        return file_tokenizer

    def __apply_dynamic_min_frequency(self, datasets: Union[Dataset, List[str]],
                                      default_min_frequency: typing.Optional[int] = None) -> None:
        if default_min_frequency is None:
            default_min_frequency = self.__config['min_frequency']

        self.__min_frequency = adjust_min_frequency(datasets, default_min_frequency)

    def __apply_dynamic_tokens(self, datasets, threshold=0.001):
        if self.__tokenizer and self.__special_tokens:
            dynamic_tokens = analyze_dataset_for_dynamic_tokens(datasets, threshold=threshold)
            self.__tokenizer.add_tokens(dynamic_tokens)
            self.__special_tokens.extend(dynamic_tokens)

    def __build_bpe_trainer(self, train_ds: Union[Dataset, List[str]]) -> None:
        self.__apply_dynamic_tokens(train_ds)
        self.__apply_dynamic_min_frequency(train_ds)

        self.__inner_trainer = trainers.BpeTrainer(
            vocab_size=self.__config['metadata']['vocab_size'],
            max_token_length=self.__config['max_length'],
            special_tokens=self.__special_tokens,
            show_progress=True,
            min_frequency=self.__min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

    def __build_bpe(self):
        return BPE(byte_fallback=self.__config.get('byte_fallback', True),
                   dropout=self.__config.get('dropout', 0.05),
                   unk_token=self.__config.get('unk_token', '<unk>'),
                   fuse_unk=self.__config.get('fuse_unk', True),
                   cache_capacity=32)

    def __save_to_file(self, path: Union[Path, str]) -> None:
        if self.__tokenizer is not None:
            self.__tokenizer.save(path)

    @property
    def tokenizer(self) -> tokenizers.Tokenizer:
        return self.__tokenizer

    @property
    def decoder(self) -> Decoder:
        return self.__tokenizer.decoder

    @property
    def pre_tokenizer(self) -> PreTokenizer:
        return self.__tokenizer.pre_tokenizer

    @property
    def normalizer(self) -> Normalizer:
        return self.__tokenizer.normalizer

    @property
    def post_processor(self) -> PostProcessor:
        return self.__tokenizer.post_processor

    @property
    def replacement_char(self) -> str:
        return self.__replacement_char

    @replacement_char.setter
    def replacement_char(self, value: Union[str, int]) -> None:
        self.__replacement_char = str(value)