"""Build the Pole YOLO dataset from a CVAT YOLO export + image folder.

Similarly to fire hydrant, this tool:
1. Syncs .txt labels from the source into annotations/pole/labels.
2. Builds train/valid/test splits in dataset/pole_split using shared build_dataset logic.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tools.dataset_builder.build_bench_dataset import build_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths identified from the environment
DEFAULT_IMAGES_DIR = Path(r"C:\Users\parsa\OneDrive\Desktop\Abtin\Pole")
DEFAULT_SOURCE_LABELS_DIR = Path(
    r"C:\Users\parsa\OneDrive\Desktop\Abtin\annotations_pole\labels\train"
)
ANNOTATIONS_DIR = REPO_ROOT / "annotations" / "pole"
ANNOTATIONS_LABELS_DIR = ANNOTATIONS_DIR / "labels"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset" / "pole_split"


def sync_pole_labels(source_labels_dir: Path, dest_labels_dir: Path) -> int:
    """Copy .txt labels from the local source into annotations/pole/labels."""
    dest_labels_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(source_labels_dir.rglob("*.txt")):
        rel = src.relative_to(source_labels_dir)
        dst = dest_labels_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/valid/test YOLO dataset for Pole."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--source-labels-dir",
        type=Path,
        default=DEFAULT_SOURCE_LABELS_DIR,
        help="Source labels directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--no-sync-annotations",
        action="store_true",
        help="Skip syncing labels into the repo.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--delete-unannotated",
        action="store_true",
        help="Delete images under --images-dir that have no matching label.",
    )
    parser.add_argument("--train-ref", default="../train/images")
    parser.add_argument("--val-ref", default="../valid/images")
    parser.add_argument("--test-ref", default="../test/images")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["fire_hydrant", "pole"],
        help="Class names for data.yaml. Note: pole is index 1 in the source labels.",
    )
    parser.add_argument("--nc", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.expanduser().resolve()
    source_labels_dir = args.source_labels_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    copied = 0
    if args.no_sync_annotations:
        labels_dir = source_labels_dir
    else:
        copied = sync_pole_labels(source_labels_dir, ANNOTATIONS_LABELS_DIR)
        labels_dir = ANNOTATIONS_LABELS_DIR

    summary = build_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        delete_unannotated=args.delete_unannotated,
        train_ref=args.train_ref,
        val_ref=args.val_ref,
        test_ref=args.test_ref,
        class_names=args.class_names,
        nc=args.nc,
    )

    if not args.no_sync_annotations:
        print("Pole label sync into repository:")
        print(f"- copied_labels: {copied}")
        print(f"- annotations_dir: {ANNOTATIONS_DIR}")
    print("Pole dataset build complete:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
