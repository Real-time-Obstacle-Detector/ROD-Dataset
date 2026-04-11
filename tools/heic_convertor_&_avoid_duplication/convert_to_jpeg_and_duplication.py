"""
Batch converter: HEIC + JPEG → all JPEG, with duplicate removal.
Duplicates like 6551(1).jpg, 6551(2).jpg are removed, keeping 6551.jpg.

Usage: python convert_to_jpeg_and_duplication.py /path/to/your/folder
"""

import sys
import re
import os
from pathlib import Path
from PIL import Image
import pillow_heif

# Register HEIC support with Pillow
pillow_heif.register_heif_opener()


def convert_folder(folder_path: str, quality: int = 95):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)

    # ── Step 1: Convert all HEIC → JPG ───────────────────────────────────────
    print("=" * 45)
    print("  STEP 1: Converting HEIC → JPEG")
    print("=" * 45)

    extensions = {".heic", ".heif", ".jpg", ".jpeg"}
    files = [f for f in folder.rglob("*") if f.suffix.lower() in extensions]

    if not files:
        print("⚠️  No image files found.")
        sys.exit(0)

    converted, skipped, errors = 0, 0, 0

    for file in files:
        output_path = file.with_suffix(".jpg")

        if file.suffix.lower() == ".jpg":
            skipped += 1
            continue

        try:
            img = Image.open(file)
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")

            img.save(output_path, "JPEG", quality=quality, optimize=True)

            if file.suffix.lower() in {".heic", ".heif"}:
                file.unlink()
                print(f"  ✅ Converted : {file.name} → {output_path.name}")
            elif file.suffix.lower() == ".jpeg":
                file.rename(output_path)
                print(f"  🔄 Renamed   : {file.name} → {output_path.name}")

            converted += 1

        except Exception as e:
            print(f"  ❌ Error with {file.name}: {e}")
            errors += 1

    print(f"\n  Converted: {converted}  |  Already .jpg: {skipped}  |  Errors: {errors}")

    # ── Step 2: Remove duplicates ─────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("  STEP 2: Removing duplicates")
    print("=" * 45)

    # Pattern matches filenames like: 6551(1).jpg or photo (2).jpg
    duplicate_pattern = re.compile(r"^(.+?)\s*\(\d+\)$", re.IGNORECASE)

    all_jpgs = list(folder.rglob("*.jpg"))

    # Group duplicates by their base name (without the (N) suffix)
    # key = (parent_folder, clean_base_name)
    groups: dict = {}
    for f in all_jpgs:
        match = duplicate_pattern.match(f.stem)
        if match:
            base = match.group(1).strip()
            key = (f.parent, base.lower())
            groups.setdefault(key, {"original": None, "dupes": []})
            groups[key]["dupes"].append(f)
        else:
            key = (f.parent, f.stem.lower())
            groups.setdefault(key, {"original": None, "dupes": []})
            groups[key]["original"] = f

    removed = 0
    for key, group in groups.items():
        original = group["original"]
        dupes = group["dupes"]

        if not dupes:
            continue  # no duplicates for this file

        if original and original.exists():
            # Original exists — delete all the (1), (2)... copies
            for dupe in dupes:
                try:
                    dupe.unlink()
                    print(f"  🗑  Removed  : {dupe.name}  (kept {original.name})")
                    removed += 1
                except Exception as e:
                    print(f"  ❌ Could not remove {dupe.name}: {e}")
        else:
            # No clean original — keep the lowest-numbered duplicate, delete the rest
            dupes_sorted = sorted(dupes, key=lambda f: (
                int(re.search(r"\((\d+)\)", f.stem).group(1))
                if re.search(r"\((\d+)\)", f.stem) else 0
            ))
            keep = dupes_sorted[0]
            # Rename it to the clean base name
            parent, base = key
            clean_name = parent / f"{base}.jpg"
            keep.rename(clean_name)
            print(f"  🔄 Promoted  : {keep.name} → {clean_name.name}")

            for dupe in dupes_sorted[1:]:
                try:
                    dupe.unlink()
                    print(f"  🗑  Removed  : {dupe.name}")
                    removed += 1
                except Exception as e:
                    print(f"  ❌ Could not remove {dupe.name}: {e}")

    if removed == 0:
        print("  ✔ No duplicates found.")

    # ── Summary ───────────────────────────────────────────────────────────────
    remaining = len(list(folder.rglob("*.jpg")))
    print("\n" + "=" * 45)
    print(f"  ✅ Converted/renamed : {converted}")
    print(f"  🗑  Duplicates removed: {removed}")
    print(f"  📁 Final .jpg count  : {remaining}")
    print("=" * 45)
    print("  Done! Folder is clean and ready for annotation.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_jpeg_and_duplication.py /path/to/folder")
        print("Example (Mac):     python convert_to_jpeg_and_duplication.py ~/Downloads/my_images")
        print("Example (Windows): python convert_to_jpeg_and_duplication.py \"C:\\Users\\You\\Downloads\\my_images\"")
        sys.exit(1)

    convert_folder(sys.argv[1])
