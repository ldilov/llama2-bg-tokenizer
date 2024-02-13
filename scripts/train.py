import json
from pathlib import Path

from tokenizer.core.datasets.dataset_loader import DatasetLoader

if __name__ == "__main__":
    from tokenizer.core.trainer import TokenizerTrainer
    config_file = Path(__file__).parent.parent / "config" / "tokenizer_config.json"
    config = None

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if config is None:
        raise ValueError("Config can't be loaded!")

    loader = DatasetLoader(buffer_size=100000)

    # Load predefined and custom datasets
    oscar_dataset = loader.load_hf_dataset("oscar", "unshuffled_deduplicated_bg")
    dataset_oscar = loader.load_hf_dataset("oscar", name="unshuffled_deduplicated_bg")
    dataset_poems = loader.load_hf_dataset(dataset_name="Dilyana56/bulgarian_poems")
    dataset_opus = loader.load_hf_dataset("anuragshas/bg_opus100_processed")
    dataset_reasoning = loader.load_hf_dataset("reasoning_bg", name="philosophy-12th", select_column="question", rename_to="text")
    dataset_fake_news = loader.load_hf_dataset("clickbait_news_bg", select_column="content", rename_to="text")

    # Merge datasets

    trainer = TokenizerTrainer(config=config)
    trainer.train([oscar_dataset, dataset_poems, dataset_opus, dataset_reasoning, dataset_fake_news], 550000)

    trainer.save(Path(__file__).parent / "saved_models" / "tokenizer.json")