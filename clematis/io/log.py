import json, os
from .paths import logs_dir

def append_jsonl(filename: str, record: dict) -> None:
    path = os.path.join(logs_dir(), filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
