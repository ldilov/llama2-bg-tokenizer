import unittest

from tokenizer.core.datasets.dataset_iterator import dataset_iterator


class TestDatasetIterator(unittest.TestCase):

    def test_remove_non_alphabet_characters(self):
        alphabet = "abc "
        dataset = ["abcd efgh!", "ijklmnop"]
        expected_results = ["abcd efgh!", "ijklmnop"]

        results = list(dataset_iterator(dataset, alphabet))
        self.assertEqual(results, expected_results)

    def test_ignore_empty_strings(self):
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        dataset = ["hello", "   ", ""]
        expected_results = ["hello"]

        results = list(dataset_iterator(dataset, alphabet))
        self.assertEqual(results, expected_results)

    def test_preserve_space_within_text(self):
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        dataset = ["hello world"]
        expected_results = ["hello world"]

        results = list(dataset_iterator(dataset, alphabet))
        self.assertEqual(results, expected_results)

    def test_strip_texts(self):
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        dataset = ["  hello world  "]
        expected_results = ["hello world"]

        results = list(dataset_iterator(dataset, alphabet))
        self.assertEqual(results, expected_results)


if __name__ == '__main__':
    unittest.main()