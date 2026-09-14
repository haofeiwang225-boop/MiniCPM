"""Download and verify the unmodified official release_v6 JSONL shards."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
DATASET_REPO = "livecodebench/code_generation_lite"
DATA_DIR = Path(__file__).resolve().parent / "data" / "release_v6"
# ALLOWED_FILES['release_v6'] in the official dataset builder at this revision.
# SHA256 values are the official Hugging Face LFS object identifiers.
SHARDS = {
    "test.jsonl": "2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f",
    "test2.jsonl": "095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3",
    "test3.jsonl": "28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79",
    "test4.jsonl": "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da",
    "test5.jsonl": "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214",
    "test6.jsonl": "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(data_dir, name, expected):
    path = data_dir / name
    if path.exists():
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}; existing file left untouched")
        return
    partial = path.with_suffix(".jsonl.partial")
    url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{name}?download=true"
    subprocess.run([
        "curl", "-L", "--fail", "--retry", "3", "--connect-timeout", "30",
        "--max-time", "3600", "-C", "-", "-sS", url, "-o", str(partial),
    ], check=True)
    if sha256(partial) != expected:
        raise ValueError(f"Checksum mismatch: {partial}")
    partial.rename(path)


def index_data(data_dir):
    """Verify raw files, indexing IDs/offsets without retaining decoded test cases."""
    index = []
    counts = {}
    for name, expected in SHARDS.items():
        path = data_dir / name
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}")
        count = 0
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                index.append((row["question_id"], name, offset))
                count += 1
        counts[name] = count
    if len({item[0] for item in index}) != len(index):
        raise ValueError("Duplicate official question IDs")
    return sorted(index, key=lambda item: item[0]), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda pair: download(args.data_dir, *pair), SHARDS.items()))
    index, counts = index_data(args.data_dir)
    print(json.dumps({"dataset_revision": DATASET_REVISION,
                      "release_v6_count": len(index), "shard_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
