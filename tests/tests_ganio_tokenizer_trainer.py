import unittest
from unittest import mock
from pathlib import Path
from unittest.mock import patch, MagicMock

from datasets import Dataset
from tokenizers.models import BPE

from tokenizer.core.trainer import TokenizerTrainer
from tokenizer.core.utilities.validator.schema import schema
from tokenizer.tests.mocks.tokenizer_config import TokenizerConfig

CONFIG = TokenizerConfig


class TestGanioTokenizerTrainer(unittest.TestCase):

    def test_init(self):
        with patch("tokenizer.core.utilities.validator.validate_config.validate_config", return_value=None) as mock_validate_config:
            with mock.patch.object(TokenizerTrainer, '_TokenizerTrainer__build_bpe', return_value=BPE()) as mock_bpe:
                TokenizerTrainer(CONFIG)
                mock_validate_config.assert_called_once_with(CONFIG, schema=schema)
                mock_bpe.assert_called_once()

    def test_train(self):
        tokenizer = TokenizerTrainer(CONFIG)
        datasets = [Dataset.from_dict({"text": ["sample1", "sample2"]})]

        tokenizer.tokenizer.train_from_iterator = MagicMock(return_value=None)
        tokenizer.train(datasets)
        tokenizer.tokenizer.train_from_iterator.assert_called()

    @patch("tokenizer.core.trainer.TokenizerTrainer.save")
    def test_save(self, mock_save):
        tokenizer = TokenizerTrainer(CONFIG)
        path = "test_path"
        tokenizer.save(path)
        mock_save.assert_called()

    @patch("transformers.PreTrainedTokenizerFast.save_pretrained")
    def test_load_from_file_and_presave(self, mock_save_pretrained):
        tokenizer = TokenizerTrainer(CONFIG)

        with mock.patch.object(TokenizerTrainer, '_TokenizerTrainer__save_to_file', return_value=None) as savef:
            with mock.patch.object(TokenizerTrainer, '_TokenizerTrainer__load_from_file', return_value=tokenizer.tokenizer) as lf:
                path = "test_path"
                tokenizer.save(path)
                mock_save_pretrained.assert_called()

    def test_getattr(self):
        tokenizer = TokenizerTrainer(CONFIG)
        with patch.object(tokenizer.tokenizer, "encode", return_value="test") as mock_method:
            result = tokenizer.encode()
            self.assertEqual(result, "test")
            mock_method.assert_called_once()

    @patch("tokenizers.Tokenizer.from_file")
    def test_load_mock_from_file(self, mock_from_file):
        path = Path("test_path")
        try:
            TokenizerTrainer.load(path, loader=mock_from_file)
        except:
            pass
        finally:
            mock_from_file.assert_called()

    @patch("tokenizer.core.trainer.TokenizerTrainer.load", return_value=CONFIG)
    def test_load_mock_load(self, mock_json_load):
        path = Path("test_path")
        try:
            TokenizerTrainer.load(path)
        except:
            pass
        finally:
            mock_json_load.assert_called()


if __name__ == "__main__":
    unittest.main()