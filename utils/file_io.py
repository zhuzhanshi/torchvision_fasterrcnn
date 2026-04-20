import json
import os
from pathlib import Path


def ensure_dir(path):
    if path is None:
        raise ValueError("Directory path is None.")
    if str(path).strip() == "":
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def dump_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_text(text: str, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_jsonl(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
