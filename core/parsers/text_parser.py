import re

from tokenizer.core.parsers.base_parser import BaseParser
from tokenizer.core.utilities.constants import ALPHABET


class TextParser(BaseParser):
    UINT32_MAX = 4294967295
    COMMON_YEARS = set(range(1900, 2025))
    COMMON_AGES = set(range(1, 100))
    NON_ALPHABET_REGEX = rf"[^{r''.join(ALPHABET)}\sa-zA-Z" + r"\{\}\[\]\(\)]+"

    def clean(self, text: str) -> str:
        text = re.sub(r'https?://\S+', '<url>', text, flags=re.IGNORECASE | re.UNICODE)
        text = re.sub(TextParser.NON_ALPHABET_REGEX, '', text)
        text = self.__bucket_numbers(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def __bucket_numbers(self, text: str, buckets: int = 4) -> str:
        bucket_size = int((2 * self.UINT32_MAX) // buckets)

        def get_bucket_token(match: re.Match[str]) -> str:
            number = int(match.group())
            if number in TextParser.COMMON_YEARS:
                return str(number)
            elif number in TextParser.COMMON_AGES:
                return str(number)
            else:
                bucket_index = max(0, min(int(number / bucket_size), buckets))
                return f'<number_{bucket_index}>'

        return re.sub(pattern=r'\d+', repl=get_bucket_token, string=text, flags=re.UNICODE | re.IGNORECASE)