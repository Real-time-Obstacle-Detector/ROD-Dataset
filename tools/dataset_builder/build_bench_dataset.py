"""Build a YOLOv8 dataset split from annotated Bench images.

Workflow:
1. Scan image directory and label directory.
2. Remove unannotated images (optional) or skip them.
3. Split matched image/label pairs into train/val/test.
4. Copy files to output directory with train/valid/test layout.
5. Write ``data.yaml`` for YOLOv8 training.
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Pair:
    """Represents one matched image and label file."""

    image_path: Path
    label_path: Path


def find_images(images_dir: Path) -> List[Path]:
    """Return all supported images under ``images_dir`` (recursive)."""
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_label_index(labels_dir: Path) -> Dict[str, Path]:
    """Build a case-insensitive index from image stem to label file path.

    Args:
        labels_dir: Directory containing YOLO ``.txt`` label files.

    Returns:
        Mapping from lowercase stem (e.g., ``img_0001``) to label path.

    Raises:
        ValueError: If duplicate label stems are found.
    """
    index: Dict[str, Path] = {}
    for label_path in sorted(labels_dir.rglob("*.txt")):
        key = label_path.stem.lower()
        if key in index:
            raise ValueError(
                f"Duplicate label stem found for '{label_path.stem}': "
                f"{index[key]} and {label_path}"
            )
        index[key] = label_path
    return index


def match_pairs(images: Iterable[Path], label_index: Dict[str, Path]) -> Tuple[List[Pair], List[Path]]:
    """Match images with labels by filename stem.

    Returns:
        matched_pairs: Images that have a corresponding label file.
        unannotated_images: Images with no matching label file.
    """
    matched_pairs: List[Pair] = []
    unannotated_images: List[Path] = []

    for image_path in images:
        label_path = label_index.get(image_path.stem.lower())
        if label_path is None:
            unannotated_images.append(image_path)
            continue
        matched_pairs.append(Pair(image_path=image_path, label_path=label_path))

    return matched_pairs, unannotated_images


def remove_files(paths: Iterable[Path]) -> int:
    """Delete files and return how many were removed."""
    removed = 0
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()
            removed += 1
    return removed


def split_pairs(
    pairs: Sequence[Pair],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    """Shuffle and split pairs by ratio.

    The training split receives the larger remainder by design so that
    ``train`` gets the biggest portion when counts are not perfectly divisible.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)

    # Assign leftover samples to train so it remains the largest split.
    remainder = total - (train_count + val_count + test_count)
    train_count += remainder

    train_split = shuffled[:train_count]
    val_split = shuffled[train_count : train_count + val_count]
    test_split = shuffled[train_count + val_count :]
    return train_split, val_split, test_split


def ensure_split_dirs(output_dir: Path) -> None:
    """Create YOLOv8-style output directories for images and labels."""
    for split_name in ("train", "valid", "test"):
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)


def copy_split(split_pairs_list: Sequence[Pair], split_name: str, output_dir: Path) -> None:
    """Copy image/label files for one split into output structure."""
    for pair in split_pairs_list:
        image_dest = output_dir / split_name / "images" / pair.image_path.name
        label_dest = output_dir / split_name / "labels" / f"{pair.image_path.stem}.txt"
        shutil.copy2(pair.image_path, image_dest)
        shutil.copy2(pair.label_path, label_dest)


def write_data_yaml(
    output_dir: Path,
    train_ref: str,
    val_ref: str,
    test_ref: str,
    class_names: Sequence[str],
    nc: int | None,
) -> None:
    """Write YOLOv8 ``data.yaml`` in the requested format."""
    data_yaml_path = output_dir / "data.yaml"
    resolved_nc = len(class_names) if nc is None else nc
    if resolved_nc != len(class_names):
        raise ValueError(
            f"Invalid class configuration: nc={resolved_nc} but names has {len(class_names)} entries."
        )
    names_literal = "[" + ", ".join(repr(name) for name in class_names) + "]"
    content = (
        f"train: {train_ref}\n"
        f"val: {val_ref}\n"
        f"test: {test_ref}\n\n"
        f"nc: {resolved_nc}\n"
        f"names: {names_literal}\n"
    )
    data_yaml_path.write_text(content, encoding="utf-8")


def build_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    delete_unannotated: bool,
    train_ref: str,
    val_ref: str,
    test_ref: str,
    class_names: Sequence[str],
    nc: int | None,
) -> dict:
    """Build the dataset and return a summary dictionary."""
    images = find_images(images_dir)
    label_index = build_label_index(labels_dir)
    matched_pairs, unannotated_images = match_pairs(images, label_index)

    removed_count = 0
    if delete_unannotated:
        removed_count = remove_files(unannotated_images)

    ensure_split_dirs(output_dir)
    train_pairs, val_pairs, test_pairs = split_pairs(
        matched_pairs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    copy_split(train_pairs, "train", output_dir)
    copy_split(val_pairs, "valid", output_dir)
    copy_split(test_pairs, "test", output_dir)
    write_data_yaml(
        output_dir=output_dir,
        train_ref=train_ref,
        val_ref=val_ref,
        test_ref=test_ref,
        class_names=class_names,
        nc=nc,
    )

    return {
        "total_images_found": len(images),
        "annotated_pairs": len(matched_pairs),
        "unannotated_images": len(unannotated_images),
        "unannotated_removed": removed_count,
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "test_count": len(test_pairs),
        "data_yaml": str((output_dir / "data.yaml").resolve()),
        "output_dir": str(output_dir.resolve()),
    }


def parse_args() -> argparse.Namespace:
    """Parse command line options for dataset generation."""
    parser = argparse.ArgumentParser(description="Build train/val/test dataset from Bench annotations.")
    parser.add_argument(
        "--images-dir",
        default="/Users/abtinzandi/Downloads/Bench_jpg",
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--labels-dir",
        default="/Users/abtinzandi/Desktop/ROD/ROD-Dataset/annotations/bench/labels",
        help="Directory containing YOLO .txt labels.",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/abtinzandi/Desktop/ROD/ROD-Dataset/dataset/bench_split",
        help="Output dataset directory.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--delete-unannotated",
        action="store_true",
        help="Delete unannotated images from the source images directory.",
    )
    parser.add_argument(
        "--train-ref",
        default="../train/images",
        help="Value for the 'train:' field in data.yaml.",
    )
    parser.add_argument(
        "--val-ref",
        default="../valid/images",
        help="Value for the 'val:' field in data.yaml.",
    )
    parser.add_argument(
        "--test-ref",
        default="../test/images",
        help="Value for the 'test:' field in data.yaml.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["bench"],
        help="Class names for data.yaml names list.",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=None,
        help="Number of classes for data.yaml (default: inferred from --class-names).",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset build from CLI arguments."""
    args = parse_args()
    summary = build_dataset(
        images_dir=Path(args.images_dir).expanduser().resolve(),
        labels_dir=Path(args.labels_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
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

    print("Dataset build complete:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
