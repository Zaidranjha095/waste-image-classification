"""Download the resized TrashNet images into data/dataset-resized/.

The zip is about 41 MB and is stored directly on GitHub (not Git LFS).
Full images are kept local and are not committed. Cite TrashNet if you use them.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "dataset-resized.zip"
URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def find_root(data_dir: Path) -> Path | None:
    for candidate in (
        data_dir / "dataset-resized",
        data_dir / "dataset-resized" / "dataset-resized",
        data_dir,
    ):
        if all((candidate / name).is_dir() for name in CLASSES):
            return candidate
    return None


def main() -> None:
    existing = find_root(DATA_DIR)
    if existing is not None:
        counts = {name: len(list((existing / name).glob("*"))) for name in CLASSES}
        print(f"Dataset already present at {existing}")
        print(counts)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    if ZIP_PATH.stat().st_size < 1_000_000:
        sys.exit("Download is too small to be the TrashNet zip.")
    print(f"Extracting {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)
    root = find_root(DATA_DIR)
    if root is None:
        sys.exit("Extracted zip does not contain the six TrashNet class folders.")
    counts = {name: len(list((root / name).glob("*"))) for name in CLASSES}
    print(root)
    print(counts)


if __name__ == "__main__":
    main()
