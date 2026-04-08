"""Build the Street dataset using fixed local source paths.

Process:
1. Copy street label annotations into repository annotations/street/labels.
2. Build YOLOv8 train/valid/test split in dataset/street_split.
3. Remove unannotated source images from the Street_jpg folder.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tools.dataset_builder.build_bench_dataset import build_dataset

IMAGES_DIR = Path('/Users/abtinzandi/Downloads/Street_jpg')
SOURCE_ANNOTATIONS_DIR = Path('/Users/abtinzandi/Downloads/street(1)')
SOURCE_LABELS_DIR = SOURCE_ANNOTATIONS_DIR / 'labels'
REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = REPO_ROOT / 'annotations' / 'street'
ANNOTATIONS_LABELS_DIR = ANNOTATIONS_DIR / 'labels'
OUTPUT_DIR = REPO_ROOT / 'dataset' / 'street_split'


def sync_street_annotations() -> int:
    """Copy `.txt` labels from source labels tree into annotations/street/labels."""
    ANNOTATIONS_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(SOURCE_LABELS_DIR.rglob('*.txt')):
        rel = src.relative_to(SOURCE_LABELS_DIR)
        dst = ANNOTATIONS_LABELS_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> None:
    copied = sync_street_annotations()
    summary = build_dataset(
        images_dir=IMAGES_DIR,
        labels_dir=ANNOTATIONS_LABELS_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        delete_unannotated=True,
        train_ref='../train/images',
        val_ref='../valid/images',
        test_ref='../test/images',
        class_names=['street'],
        nc=None,
    )

    print('Street annotation sync complete:')
    print(f'- copied_labels: {copied}')
    print(f'- annotations_dir: {ANNOTATIONS_DIR}')
    print('Street dataset build complete:')
    for key, value in summary.items():
        print(f'- {key}: {value}')


if __name__ == '__main__':
    main()
