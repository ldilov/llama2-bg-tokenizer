import time
from tokenizer.core.logger import logger


class Timer:
    def __init__(self, msg="Execution took:"):
        self.msg = msg

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        logger.info(f"{self.msg} {end-self.start:.4f} seconds")