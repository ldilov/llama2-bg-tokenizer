import logging
import time

# Color customization
ANSI_RESET = "\u001B[0m"
ANSI_GREEN = "\u001B[32m"
ANSI_YELLOW = "\u001B[33m"
ANSI_RED = "\u001B[31m"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    f"%(asctime)s - %(levelname)s - %(name)s : {ANSI_GREEN}%(message)s{ANSI_RESET}"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)