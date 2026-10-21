from __future__ import annotations
import json
from pathlib import Path
import torch
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
def prepare_frontend_cache_write(cache_dir):
    cache_path = Path(cache_dir)
    if cache_path.exists() and any(cache_path.iterdir()):
        raise FileExistsError("Cache output already exists")
    (cache_path / "frames").mkdir(parents=True, exist_ok=True)
def save_frontend_cache_frame(cache_dir, frame_index, payload):
    relative_path = Path("frames") / f"{frame_index:04d}.pt"
    output_path = Path(cache_dir) / relative_path
    if output_path.exists():
        raise FileExistsError("Cache output already exists")
    torch.save({
        key: value.detach().cpu().contiguous()
        if torch.is_tensor(value) else value
        for key, value in payload.items()
    }, output_path)
    return relative_path.as_posix()
def write_frontend_cache_manifest(cache_dir, metadata):
    output_path = Path(cache_dir) / "manifest.json"
    with output_path.open("x", encoding="utf-8") as output_file:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            **metadata,
        }, output_file, indent=2)
def load_frontend_cache_manifest(cache_dir):
    manifest_path = Path(cache_dir) / "manifest.json"
    with manifest_path.open(encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    return manifest
def load_frontend_cache_frame(cache_dir, frame_record):
    frame_path = Path(cache_dir) / frame_record["file"]
    payload = torch.load(frame_path, map_location="cpu", weights_only=True)
    return payload
