import json
import os
from . import paths


def append_jsonl(filename: str, record: dict) -> None:
    base = paths.logs_dir()
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
