# HEIC converter and duplicate cleanup

This script converts HEIC/HEIF (and normalizes `.jpeg`) to `.jpg` under a folder tree, then removes Windows-style duplicate filenames such as `photo (1).jpg` when a base `photo.jpg` exists.

## Step 1 — Install dependencies

From a terminal (PowerShell or Command Prompt), run:

```bash
pip install pillow-heif Pillow
```

Use `python -m pip install pillow-heif Pillow` if `pip` is not on your PATH.

## Step 2 — Run the script

1. Open a terminal and go to this folder (the one that contains `convert_to_jpeg_and_duplication.py`).
2. Run:

```bash
python convert_to_jpeg_and_duplication.py PATH
```

Replace `PATH` with the full path to the folder you want to process.

**How to get `PATH` on Windows:** In File Explorer, single-click the folder you want (so it is selected), then click the address bar, copy the path (`Ctrl+C`), and paste it after the command as `PATH`. If the path contains spaces, wrap it in double quotes, for example:

```bash
python convert_to_jpeg_and_duplication.py "D:\Photos\my batch"
```

**Warning:** The script deletes original `.heic`/`.heif` files after a successful conversion and removes duplicate `.jpg` files. Work on a copy of your data if you are unsure.
