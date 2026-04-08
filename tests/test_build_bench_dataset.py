"""Tests for tools/dataset_builder/build_bench_dataset.py."""

from pathlib import Path

from tools.dataset_builder.build_bench_dataset import (
    Pair,
    build_dataset,
    ensure_split_dirs,
    match_pairs,
    split_pairs,
    write_data_yaml,
)


def test_match_pairs_separates_unannotated(tmp_path: Path) -> None:
    images = [tmp_path / "A.jpg", tmp_path / "B.jpg", tmp_path / "C.jpg"]
    label_index = {"a": Path("/tmp/A.txt"), "c": Path("/tmp/C.txt")}

    matched, unannotated = match_pairs(images, label_index)

    assert len(matched) == 2
    assert {p.image_path.stem for p in matched} == {"A", "C"}
    assert [p.stem for p in unannotated] == ["B"]


def test_split_pairs_ratios_and_remainder_goes_to_train() -> None:
    pairs = [Pair(Path(f"/tmp/{i}.jpg"), Path(f"/tmp/{i}.txt")) for i in range(11)]

    train, val, test = split_pairs(
        pairs,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=1,
    )

    assert len(train) == 9
    assert len(val) == 1
    assert len(test) == 1


def test_ensure_split_dirs_creates_train_valid_test_structure(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    ensure_split_dirs(output_dir)

    for split_name in ("train", "valid", "test"):
        assert (output_dir / split_name / "images").is_dir()
        assert (output_dir / split_name / "labels").is_dir()


def test_write_data_yaml_uses_requested_format(tmp_path: Path) -> None:
    write_data_yaml(
        output_dir=tmp_path,
        train_ref="../train/images",
        val_ref="../valid/images",
        test_ref="../test/images",
        class_names=["bench"],
        nc=1,
    )
    expected = (
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n\n"
        "nc: 1\n"
        "names: ['bench']\n"
    )
    assert (tmp_path / "data.yaml").read_text(encoding="utf-8") == expected


def test_build_dataset_outputs_new_yolov8_layout(tmp_path: Path) -> None:
    images_dir = tmp_path / "images_src"
    labels_dir = tmp_path / "labels_src"
    output_dir = tmp_path / "dataset_out"
    images_dir.mkdir()
    labels_dir.mkdir()

    for stem in ("A", "B", "C"):
        (images_dir / f"{stem}.jpg").write_bytes(b"img")
        (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 1 1\n", encoding="utf-8")

    summary = build_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=1,
        delete_unannotated=False,
        train_ref="../train/images",
        val_ref="../valid/images",
        test_ref="../test/images",
        class_names=["bench"],
        nc=None,
    )

    assert summary["annotated_pairs"] == 3
    assert summary["train_count"] == 3
    assert summary["val_count"] == 0
    assert summary["test_count"] == 0
    assert (output_dir / "data.yaml").exists()
    assert any((output_dir / "train" / "images").iterdir())
    assert any((output_dir / "train" / "labels").iterdir())
