"""
Build Kaggle-ready CSV files for the ROD (Real-time Obstacle Detector) dataset.

Outputs:
    - class_distribution.csv : per-class counts across train/valid/test splits
    - images_manifest.csv    : one row per image with split, filename, annotation count, classes present
"""
import csv
import os
from collections import Counter

DATASET_DIR = "/Users/abtinzandi/Documents/projects/ROD/ROD-Dataset/dataset/dataset"
OUTPUT_DIR = DATASET_DIR

CLASS_NAMES = [
    "Bike", "Building", "Car", "Person", "Stairs", "Traffic sign", "Electrical Pole",
    "Road", "Motorcycle", "Dustbin", "Dog", "Manhole", "Tree", "Guard rail",
    "Pedestrian crosswalk", "Truck", "Bus", "Bench", "Traffic Cone", "Fire hydrant",
    "Teraffic Barrel", "Plant Pot", "Electrical Box", "Chair", "Bicycle Rack",
]
SPLITS = ["train", "valid", "test"]


def parse_label_file(path):
    """Return list of class ids in a YOLO label file."""
    classes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                classes.append(int(parts[0]))
            except ValueError:
                continue
    return classes


def collect():
    per_split_class_counts = {s: Counter() for s in SPLITS}
    manifest_rows = []

    for split in SPLITS:
        labels_dir = os.path.join(DATASET_DIR, split, "labels")
        images_dir = os.path.join(DATASET_DIR, split, "images")
        if not os.path.isdir(labels_dir):
            continue
        for fname in sorted(os.listdir(labels_dir)):
            if not fname.endswith(".txt"):
                continue
            label_path = os.path.join(labels_dir, fname)
            cids = parse_label_file(label_path)
            per_split_class_counts[split].update(cids)
            image_name = fname[:-4] + ".jpg"
            image_exists = os.path.isfile(os.path.join(images_dir, image_name))
            classes_present = sorted(set(cids))
            manifest_rows.append({
                "split": split,
                "image_filename": image_name if image_exists else "",
                "label_filename": fname,
                "num_objects": len(cids),
                "classes_present": ";".join(str(c) for c in classes_present),
                "class_names_present": ";".join(CLASS_NAMES[c] for c in classes_present),
            })
    return per_split_class_counts, manifest_rows


def write_class_distribution(counts):
    out_path = os.path.join(OUTPUT_DIR, "class_distribution.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "train", "valid", "test", "total"])
        totals = [0, 0, 0, 0]  # train, valid, test, all
        for cid, name in enumerate(CLASS_NAMES):
            t = counts["train"].get(cid, 0)
            v = counts["valid"].get(cid, 0)
            te = counts["test"].get(cid, 0)
            tot = t + v + te
            totals[0] += t
            totals[1] += v
            totals[2] += te
            totals[3] += tot
            w.writerow([cid, name, t, v, te, tot])
        w.writerow(["", "TOTAL", totals[0], totals[1], totals[2], totals[3]])
    print(f"[+] wrote {out_path}")


def write_manifest(rows):
    out_path = os.path.join(OUTPUT_DIR, "images_manifest.csv")
    fieldnames = ["split", "image_filename", "label_filename", "num_objects", "classes_present", "class_names_present"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[+] wrote {out_path}  ({len(rows)} rows)")


def main():
    counts, rows = collect()
    write_class_distribution(counts)
    write_manifest(rows)


if __name__ == "__main__":
    main()
