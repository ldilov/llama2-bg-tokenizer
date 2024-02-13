import re
from abc import ABC, abstractmethod

from tokenizer.core.utilities.constants import ALPHABET


class BaseParser(ABC):
    NOT_ALPHABET_REGEX = re.compile(r'[^' + ''.join(ALPHABET) + r']*', re.UNICODE)

    ALPHABET_REGEX = re.compile(r'[' + ''.join(ALPHABET) + r']*', re.UNICODE)

    def parse(self, text: str) -> str:
        """
        Parses the given text and returns the processed text.

        Args:
            text (str): The input text to parse.

        Returns:
            str: The parsed text.
        """
        return self.clean(text)

    @abstractmethod
    def clean(self, text: str) -> str:
        pass