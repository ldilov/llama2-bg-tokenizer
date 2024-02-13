from pathlib import Path
from datasets import Features, Value


class WortschatzDatasetGenerator:
    """Custom Wortschatz Unviersity dataset builder"""

    FEATURES = Features({'text': Value(dtype='string', id='text'), 'id': Value(dtype='int64', id='id')})

    def __call__(self):
        id_ = 0
        data_dir = Path(__file__).parent.parent / "parsed"
        filepaths = list(data_dir.glob('*.txt'))
        for filepath in filepaths:
            with filepath.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield {"id": id_, "text": line.strip()}
                        id_ += 1