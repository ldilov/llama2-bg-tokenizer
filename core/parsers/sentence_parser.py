import re
from .base_parser import BaseParser


class SentenceParser(BaseParser):
    def parse(self, text: str) -> str:
        """
        Parses the input text, focusing on sentence-level processing.

        Args:
            text (str): The input text to parse.

        Returns:
            str: The parsed text, with sentence-specific modifications applied.
        """

        super(SentenceParser, self).parse(text)
        return text

    def clean(self, text: str):
        text = text.split('\t')[-1]
        return re.sub(BaseParser.NOT_ALPHABET_REGEX, '', text)