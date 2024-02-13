import unittest

from unittest.mock import patch, MagicMock
from tokenizer.core.datasets.dataset_loader import DatasetLoader


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DatasetLoader(buffer_size=10000)

    def test_load_hf_dataset(self):
        with patch("tokenizer.core.datasets.dataset_loader.load_dataset", return_value=MagicMock(name='dataset', shuffle=lambda: 'shuffled_dataset')) as mock_load:
            dataset_iterator = self.loader.load_hf_dataset("dummy_dataset", split="train", shuffle=True)
            mock_load.assert_called_once_with("dummy_dataset", split="train", name=None)
            self.assertIsNotNone(dataset_iterator)

    @patch("datasets.Dataset.from_generator")
    def test_load_custom_dataset(self, mock_from_generator):
        mock_from_generator.return_value = MagicMock(shuffle=MagicMock(return_value="shuffled_custom_dataset"))
        generator_function = MagicMock()

        dataset_iterator = self.loader.load_custom_dataset(generator_function, shuffle=True)
        mock_from_generator.assert_called_once_with(generator_function)
        self.assertEqual(dataset_iterator, "shuffled_custom_dataset")

    @patch("tokenizer.core.utilities.merger.Merger.shuffled_stream_merged_datasets")
    def test_merge_datasets(self, mock_shuffled_stream_merged_datasets):
        mock_shuffled_stream_merged_datasets.return_value = iter(["merged_datasets"])
        datasets = [MagicMock(), MagicMock()]
        merged_iterator = list(self.loader.merge_datasets(*datasets, merger=mock_shuffled_stream_merged_datasets))
        mock_shuffled_stream_merged_datasets.assert_called_once_with(*datasets)
        self.assertIn("merged_datasets", merged_iterator)


if __name__ == "__main__":
    unittest.main()