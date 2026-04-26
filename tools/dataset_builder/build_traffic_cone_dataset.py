"""Build the Traffic Cone YOLO dataset from a CVAT YOLO export + image folders.

This tool:
1. Syncs .txt labels from the source into annotations/traffic_cone/labels.
2. Builds train/valid/test splits in dataset/traffic_cone_split using shared build_dataset logic.
3. Scans the parent Abtin folder to find images spread across subdirectories.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tools.dataset_builder.build_bench_dataset import build_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths identified from the environment
DEFAULT_IMAGES_DIR = Path(r"C:\Users\parsa\OneDrive\Desktop\Abtin\Traffic Cone")
DEFAULT_SOURCE_LABELS_DIR = Path(
    r"C:\Users\parsa\OneDrive\Desktop\Abtin\Traffic cone annotation\labels\train"
)
ANNOTATIONS_DIR = REPO_ROOT / "annotations" / "traffic_cone"
ANNOTATIONS_LABELS_DIR = ANNOTATIONS_DIR / "labels"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset" / "traffic_cone_split"


def sync_traffic_cone_labels(source_labels_dir: Path, dest_labels_dir: Path) -> int:
    """Copy .txt labels from the local source into annotations/traffic_cone/labels."""
    dest_labels_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    # Search for all .txt files in the source labels directory
    for src in sorted(source_labels_dir.rglob("*.txt")):
        # We want to flatten them or keep structure? 
        # Usually they are flat in the CVAT export
        rel = src.relative_to(source_labels_dir)
        dst = dest_labels_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/valid/test YOLO dataset for Traffic Cone."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing source images (Abtin root).",
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
    parser.add_argument("--train-ref", default="train/images")
    parser.add_argument("--val-ref", default="valid/images")
    parser.add_argument("--test-ref", default="test/images")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["fire_hydrant", "pole", "traffic_cone"],
        help="Class names for data.yaml. Note: traffic_cone is index 2.",
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
        copied = sync_traffic_cone_labels(source_labels_dir, ANNOTATIONS_LABELS_DIR)
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
        print("Traffic Cone label sync into repository:")
        print(f"- copied_labels: {copied}")
        print(f"- annotations_dir: {ANNOTATIONS_DIR}")
    print("Traffic Cone dataset build complete:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
