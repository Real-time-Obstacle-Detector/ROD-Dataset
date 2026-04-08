"""Tests for tools/dataset_builder/build_bench_dataset.py."""

from pathlib import Path

from tools.dataset_builder.build_bench_dataset import match_pairs, split_pairs, Pair


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
