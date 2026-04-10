"""Build the Dustbin YOLO dataset from a CVAT YOLO export + image folder.

CVAT "YOLO 1.1" exports typically contain ``labels/train/*.txt`` (and optional
``val``/``test``) plus a small ``data.yaml`` that differs from our training layout.

This tool:
1. Optionally copies label ``.txt`` files into ``annotations/dustbin/labels`` in
   the repository (same idea as ``build_street_dataset``).
2. Builds ``train`` / ``valid`` / ``test`` splits with ``images`` and ``labels``
   subfolders and writes ``data.yaml`` in the Ultralytics-style format used by
   ``build_bench_dataset`` (``train:``, ``val:``, ``test:``, ``nc``, ``names``).

Default paths match the local Dustbin + CVAT export layout; override with CLI args.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tools.dataset_builder.build_bench_dataset import build_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IMAGES_DIR = Path(r"C:\Users\ariya\Desktop\Dustbin convert area\Dustbin")
DEFAULT_SOURCE_LABELS_DIR = Path(
    r"C:\Users\ariya\Desktop\Dustbin convert area\dustbin_annotation_by_cvat\labels"
)
ANNOTATIONS_DIR = REPO_ROOT / "annotations" / "dustbin"
ANNOTATIONS_LABELS_DIR = ANNOTATIONS_DIR / "labels"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset" / "dustbin_split"


def sync_dustbin_labels(source_labels_dir: Path, dest_labels_dir: Path) -> int:
    """Copy ``.txt`` labels from the CVAT export tree into ``annotations/dustbin/labels``."""
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
        description="Build train/valid/test YOLO dataset for Dustbin (CVAT YOLO export + images)."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing source images (recursive scan).",
    )
    parser.add_argument(
        "--source-labels-dir",
        type=Path,
        default=DEFAULT_SOURCE_LABELS_DIR,
        help="CVAT YOLO export labels root (e.g. .../dustbin_annotation_by_cvat/labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output dataset directory (train/valid/test + data.yaml).",
    )
    parser.add_argument(
        "--no-sync-annotations",
        action="store_true",
        help="Do not copy labels into the repo under annotations/dustbin/labels; use --source-labels-dir directly.",
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
        default=["dustbin"],
        help="Class names for data.yaml (single-class dustbin by default).",
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
        copied = sync_dustbin_labels(source_labels_dir, ANNOTATIONS_LABELS_DIR)
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
        print("Dustbin label sync into repository:")
        print(f"- copied_labels: {copied}")
        print(f"- annotations_dir: {ANNOTATIONS_DIR}")
    print("Dustbin dataset build complete:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
