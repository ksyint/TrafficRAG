import json
from pathlib import Path


def load_query(path):
    return json.loads(Path(path).read_text())
