import os
import shutil


def process_and_merge_dataset(dataset_paths, final_dataset_path, index_mapping):
    """
    Updates label indices in a new dataset and merges its images and labels
    into the final ROD dataset.

    Arguments:
    - dataset_paths: dict with keys ('train', 'test', 'valid'), each a sub-dict
                     with 'labels' and 'images' folder paths.
    - final_dataset_path: path to the final dataset (with train/test/valid/{images,labels}).
    - index_mapping: dict {original_class_idx_str: final_class_idx_str}.
    """

    for dataset_type, paths in dataset_paths.items():
        src_label_folder = paths["labels"]
        src_image_folder = paths["images"]

        final_labels_folder = os.path.join(final_dataset_path, dataset_type, "labels")
        final_images_folder = os.path.join(final_dataset_path, dataset_type, "images")

        for filename in os.listdir(src_label_folder):
            label_file_path = os.path.join(src_label_folder, filename)

            if not (os.path.isfile(label_file_path) and label_file_path.endswith(".txt")):
                continue

            with open(label_file_path, "r") as file:
                lines = file.readlines()

            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] in index_mapping:
                    parts[0] = index_mapping[parts[0]]
                modified_lines.append(" ".join(parts) + "\n")

            final_label_path = os.path.join(final_labels_folder, filename)
            with open(final_label_path, "w") as file:
                file.writelines(modified_lines)
            print(f"Updated labels and copied: {final_label_path}")

            image_file_path = os.path.join(src_image_folder, filename.replace(".txt", ".jpg"))
            if os.path.isfile(image_file_path):
                final_image_path = os.path.join(final_images_folder, os.path.basename(image_file_path))
                shutil.copy2(image_file_path, final_image_path)
                print(f"Copied image: {final_image_path}")


# Chair Detection: ['chair'] -> Chair (23)
NEW_DATASET_PATH = "/Users/abtinzandi/Documents/projects/ROD/ROD-Dataset/dataset/new_datasets/Chair Detection.v1i.yolov8"
FINAL_DATASET_PATH = "/Users/abtinzandi/Documents/projects/ROD/ROD-Dataset/dataset/dataset"
INDEX_MAPPING = {"0": "23"}

dataset_paths = {
    "train": {
        "labels": os.path.join(NEW_DATASET_PATH, "train", "labels"),
        "images": os.path.join(NEW_DATASET_PATH, "train", "images"),
    },
    "test": {
        "labels": os.path.join(NEW_DATASET_PATH, "test", "labels"),
        "images": os.path.join(NEW_DATASET_PATH, "test", "images"),
    },
    "valid": {
        "labels": os.path.join(NEW_DATASET_PATH, "valid", "labels"),
        "images": os.path.join(NEW_DATASET_PATH, "valid", "images"),
    },
}

process_and_merge_dataset(dataset_paths, FINAL_DATASET_PATH, INDEX_MAPPING)
