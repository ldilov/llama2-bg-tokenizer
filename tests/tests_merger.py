import unittest
from datasets import Dataset

from tokenizer.core.utilities.merger import Merger


class TestMerger(unittest.TestCase):
    def setUp(self):
        self.dataset1 = Dataset.from_dict({"text": ["sample1", "sample2"]})
        self.dataset2 = Dataset.from_dict({"text": ["sample3", "sample4"]})

    def test_stream_merged_datasets(self):
        merged_stream = Merger.stream_merged_datasets(self.dataset1, self.dataset2)
        merged_list = list(merged_stream)
        expected_texts = ["sample1", "sample2", "sample3", "sample4"]
        self.assertEqual([sample["text"] for sample in merged_list], expected_texts)

    def test_shuffled_stream_merged_datasets(self):
        merged_stream = Merger.shuffled_stream_merged_datasets(self.dataset1, self.dataset2, buffer_size=2)
        merged_list = list(merged_stream)
        expected_texts = set(["sample1", "sample2", "sample3", "sample4"])
        self.assertEqual(set([sample["text"] for sample in merged_list]), expected_texts)

    def test_merge_with_shuffling(self):
        merged_dataset = Merger.merge(self.dataset1, self.dataset2, shuffle=True)
        self.assertEqual(set(merged_dataset["text"]), set(["sample1", "sample2", "sample3", "sample4"]))

    def test_merge_without_shuffling(self):
        merged_dataset = Merger.merge(self.dataset1, self.dataset2, shuffle=False)
        expected_texts = ["sample1", "sample2", "sample3", "sample4"]
        self.assertEqual(merged_dataset["text"], expected_texts)


if __name__ == "__main__":
    unittest.main()