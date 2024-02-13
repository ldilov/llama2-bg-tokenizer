import gc
import heapq
import numpy as np
from itertools import chain, islice
from typing import Union, List
from datasets import Dataset, DatasetDict
from tokenizer.core.logger import logger
from multiprocessing import Pool, cpu_count
from collections import Counter


def process_chunk(texts):
    from tokenizer.core.parsers.text_parser import TextParser

    counter = Counter()
    text_parser = TextParser()

    for text in texts:
        text = text_parser.parse(text)
        if len(text) > 1:
            counter.update(text.split())
    return counter


def count_frequencies(texts):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    return counter


def chunks_generator(data, chunk_size):
    """Yield successive n-sized chunks from data."""
    iterator = iter(data)
    for first in iterator:
        yield chain([first], islice(iterator, chunk_size - 1))


def analyze_dataset_for_dynamic_tokens(datasets, threshold=0.0005, top_k=10):
    total = len(datasets)
    if isinstance(datasets[0], (Dataset, DatasetDict)):
        datasets = (text for dataset in datasets for text in dataset['text'])
    else:
        datasets = iter(datasets)

    cpu_cores = max(1, cpu_count() - 2)
    chunk_size = int(total / cpu_cores) + 1

    dataset_chunks = chunks_generator(datasets, chunk_size)
    with Pool(processes=cpu_cores) as pool:
        counters = pool.map(process_chunk, dataset_chunks)

    total_counter = Counter()
    for counter in counters:
        total_counter.update(counter)

    total_words = sum(total_counter.values())
    dynamic_tokens = [(token, count) for token, count in total_counter.items() if count / total_words <= threshold]
    top_k_rare_tokens = heapq.nlargest(top_k, dynamic_tokens, key=lambda x: x[1])
    top_k_tokens = [token for token, count in top_k_rare_tokens]

    logger.info(f"Total dynamic tokens: {len(top_k_tokens)}")

    counters = None
    gc.collect()

    return top_k_tokens


def adjust_min_frequency(datasets: Union[List[str], List[List[str]]], default_min_frequency: int = 1) -> int:
    if all(isinstance(dataset, list) for dataset in datasets):
        datasets = [text for sublist in datasets for text in sublist]

    cpu_cores = max(1, cpu_count() - 2)
    chunk_size = int(len(datasets) / cpu_cores) + 1
    datasets_chunks = [datasets[i:i + chunk_size] for i in range(0, len(datasets), chunk_size)]

    # Process each chunk in parallel to count frequencies
    with Pool(processes=cpu_cores) as pool:
        results = pool.map(count_frequencies, datasets_chunks)

    # Aggregate results from all chunks
    total_counter = Counter()
    for counter in results:
        total_counter.update(counter)

    # Calculate statistical metrics
    frequencies = np.array(list(total_counter.values()))
    mean_freq = np.mean(frequencies)
    std_freq = np.std(frequencies)
    median_freq = np.median(frequencies)
    skewness = ((3 * (mean_freq - median_freq)) / std_freq) if std_freq else 0

    # Dynamic adjustment based on statistical analysis
    if skewness > 1:  # Highly skewed distribution
        adjusted_min_frequency = max(1, int(mean_freq - std_freq))
    elif mean_freq > 2 and std_freq < mean_freq:
        adjusted_min_frequency = int(max(mean_freq - std_freq, default_min_frequency))
    else:
        adjusted_min_frequency = default_min_frequency

    # Ensure the adjusted frequency is sensible
    adjusted_min_frequency = min(adjusted_min_frequency, int(mean_freq), default_min_frequency)

    logger.info(f"Adjusted min_frequency set to: {adjusted_min_frequency}")
    gc.collect()

    return adjusted_min_frequency