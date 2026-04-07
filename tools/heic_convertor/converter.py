"""HEIC to JPG conversion utility.

This module provides a small, documented utility to convert HEIC/HEIF images
from an input directory into JPG files in an output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# Register HEIC/HEIF support for Pillow before opening images.
register_heif_opener()

# File extensions treated as HEIC/HEIF images.
HEIC_EXTENSIONS = {".heic", ".heif"}


def find_heic_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    """Return all HEIC/HEIF files found under ``input_dir``.

    Args:
        input_dir: Directory to search for source images.
        recursive: If True, scan all nested subdirectories.

    Returns:
        A sorted list of paths to HEIC/HEIF files.
    """
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in HEIC_EXTENSIONS
    ]
    return sorted(files)


def build_output_path(source_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Build the output JPG path for a source file.

    The relative directory structure from ``input_dir`` is preserved in
    ``output_dir``. The extension is replaced with ``.jpg``.

    Args:
        source_path: Source HEIC/HEIF image path.
        input_dir: Base input directory.
        output_dir: Base output directory.

    Returns:
        Destination JPG path.
    """
    relative = source_path.relative_to(input_dir)
    return output_dir / relative.with_suffix(".jpg")


def convert_heic_to_jpg(source_path: Path, output_path: Path, quality: int = 100) -> None:
    """Convert one HEIC/HEIF image file to JPG with high-quality settings.

    This function applies EXIF orientation, converts to RGB color space,
    and saves JPEG with:
    - quality=100
    - subsampling=0 (best chroma fidelity)
    - EXIF and ICC profile copied when available

    Args:
        source_path: Input HEIC/HEIF file path.
        output_path: Output JPG file path.
        quality: JPEG quality value (1-100). Use 100 for maximum quality.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)
        image_rgb = image.convert("RGB")

        save_kwargs = {
            "format": "JPEG",
            "quality": quality,
            "subsampling": 0,
            "optimize": False,
            "progressive": False,
        }

        # Preserve metadata where available.
        exif_bytes = image.info.get("exif")
        icc_profile = image.info.get("icc_profile")
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        image_rgb.save(output_path, **save_kwargs)


def convert_heic_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    recursive: bool = True,
    quality: int = 100,
) -> int:
    """Convert all HEIC/HEIF images in a folder to JPG.

    Args:
        input_dir: Source folder containing HEIC images.
        output_dir: Destination folder where JPG files are saved.
        recursive: If True, process nested folders.
        quality: JPEG quality value (1-100).

    Returns:
        Number of converted image files.

    Raises:
        ValueError: If quality is outside 1..100.
        FileNotFoundError: If input directory does not exist.
        NotADirectoryError: If input path is not a directory.
    """
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")
    if not (1 <= quality <= 100):
        raise ValueError("quality must be between 1 and 100")

    sources = find_heic_files(input_path, recursive=recursive)

    for source in sources:
        destination = build_output_path(source, input_path, output_path)
        convert_heic_to_jpg(source, destination, quality=quality)

    return len(sources)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Convert HEIC/HEIF images in a folder to JPG."
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir_flag",
        default=None,
        help="Input directory containing HEIC/HEIF files.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir_flag",
        default=None,
        help="Output directory for converted JPG files.",
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/Users/abtinzandi/Downloads/Bench",
        help="Input directory containing HEIC/HEIF files.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="/Users/abtinzandi/Downloads/Bench_jpg",
        help="Output directory for converted JPG files.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only process the top-level input directory.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=100,
        help="JPEG quality (1-100). Default: 100",
    )
    args = parser.parse_args()

    # Flags take precedence over positional values when both are provided.
    if args.input_dir_flag:
        args.input_dir = args.input_dir_flag
    if args.output_dir_flag:
        args.output_dir = args.output_dir_flag

    return args


def main() -> None:
    """CLI entrypoint for HEIC folder conversion."""
    args = parse_args()
    converted_count = convert_heic_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=not args.non_recursive,
        quality=args.quality,
    )
    print(f"Converted {converted_count} file(s) to JPG.")


if __name__ == "__main__":
    main()
