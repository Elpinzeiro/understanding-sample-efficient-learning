"""Logging and JSON files. Writes go through a per-process temp file and a rename, so a reader (or another worker)
never sees half a file."""
import datetime
import json
import os
from pathlib import Path


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def tag_of(repo):
    """'google/gemma-3-12b-it' -> 'gemma-3-12b-it' (the folder name of a model everywhere)."""
    return repo.split("/")[-1]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, obj, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)
