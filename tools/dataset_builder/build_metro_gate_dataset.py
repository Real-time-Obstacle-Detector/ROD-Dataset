"""Build the Metro Gate YOLO dataset from a CVAT YOLO export with HEIC images."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_builder.build_bench_dataset import build_dataset
from tools.heic_convertor.converter import convert_heic_folder

DEFAULT_EXPORT_DIR = Path(
    "/Users/abtinzandi/Downloads/"
    "job_3936959_annotations_2026_04_30_22_42_06_ultralytics_yolo_detection"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset" / "new_datasets" / "Metro Gates"
ANNOTATIONS_DIR = REPO_ROOT / "annotations" / "metro_gate"
ANNOTATIONS_LABELS_DIR = ANNOTATIONS_DIR / "labels"


def sync_labels(source_labels_dir: Path, dest_labels_dir: Path) -> int:
    """Copy YOLO labels from the export into annotations/metro_gate/labels."""
    dest_labels_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in sorted(source_labels_dir.rglob("*.txt")):
        destination = dest_labels_dir / source.name
        shutil.copy2(source, destination)
        copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Metro Gate YOLO dataset.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.expanduser().resolve()
    images_dir = export_dir / "Metro Gates"
    source_labels_dir = export_dir / "labels" / "train"
    output_dir = args.output_dir.expanduser().resolve()

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not source_labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {source_labels_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    copied_labels = sync_labels(source_labels_dir, ANNOTATIONS_LABELS_DIR)

    with tempfile.TemporaryDirectory(prefix="metro_gate_jpg_") as temp_dir:
        converted_dir = Path(temp_dir)
        converted_images = convert_heic_folder(
            input_dir=images_dir,
            output_dir=converted_dir,
            recursive=True,
            quality=args.quality,
        )

        summary = build_dataset(
            images_dir=converted_dir,
            labels_dir=ANNOTATIONS_LABELS_DIR,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            delete_unannotated=False,
            train_ref="../train/images",
            val_ref="../valid/images",
            test_ref="../test/images",
            class_names=["metro_gate"],
            nc=None,
        )

    print("Metro Gate dataset build complete:")
    print(f"- copied_labels: {copied_labels}")
    print(f"- converted_images: {converted_images}")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
