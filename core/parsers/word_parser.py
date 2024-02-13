import re
from .base_parser import BaseParser


class WordParser(BaseParser):
    def parse(self, text: str) -> str:
        """
        Parses the input text, focusing on word-level processing.

        Args:
            text (str): The input text to parse.

        Returns:
            str: The parsed text, with word-specific modifications applied.
        """
        super(WordParser, self).parse(text)

    def clean(self, text: str):
        cleaned_word = WordParser.extract_backtick(text)

        if WordParser.is_nonsense(cleaned_word):
            cleaned_word = None

        return cleaned_word

    @staticmethod
    def extract_backtick(input_data: str):
        input_data = input_data.split('\t')[1]
        input_data = re.sub(BaseParser.NOT_ALPHABET_REGEX, '', input_data)

        if not input_data.endswith('\n'):
            input_data = input_data + '\n'

        return input_data

    @staticmethod
    def is_nonsense(word: str):
        pattern = re.compile(r'^([^\d\.!?])\1{3,}$')
        match = pattern.search(word)
        return bool(match)