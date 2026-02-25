"""Tests for sharding: path listing, partitioning, single-shard write."""

from __future__ import annotations

from pathlib import Path

import pytest
import pyarrow.parquet as pq
import webdataset as wds

from prism.sharding.metadata import write_metadata_parquet
from prism.sharding.paths import list_image_paths
from prism.sharding.reader import list_shards
from prism.sharding.writer import partition_paths, write_one_shard, write_shards


def test_list_image_paths_empty(tmp_path: Path) -> None:
    assert list_image_paths(tmp_path) == []


def test_list_image_paths_class_subdirs(tmp_path: Path) -> None:
    # Layout: root/class_a/1.jpg, root/class_a/2.jpg, root/class_b/3.png
    (tmp_path / "class_a").mkdir()
    (tmp_path / "class_b").mkdir()
    (tmp_path / "class_a" / "1.jpg").write_bytes(b"x")
    (tmp_path / "class_a" / "2.jpg").write_bytes(b"y")
    (tmp_path / "class_b" / "3.png").write_bytes(b"z")
    (tmp_path / "class_b" / "readme.txt").write_bytes(b"ignore")
    result = list_image_paths(tmp_path, with_labels=True)
    assert len(result) == 3
    paths = [p for p, _ in result]
    assert (tmp_path / "class_a" / "1.jpg") in paths
    assert (tmp_path / "class_a" / "2.jpg") in paths
    assert (tmp_path / "class_b" / "3.png") in paths
    labels = [lbl for _, lbl in result]
    assert "class_a" in labels and "class_b" in labels


def test_list_image_paths_respects_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.gif").write_bytes(b"y")
    (tmp_path / "c.png").write_bytes(b"z")
    result = list_image_paths(tmp_path, extensions=[".jpg", ".png"], with_labels=False)
    assert len(result) == 2
    suffixes = {p.suffix.lower() for p, _ in result}
    assert suffixes == {".jpg", ".png"}


def test_list_image_paths_excludes_sidecars(tmp_path: Path) -> None:
    (tmp_path / "n02106662").mkdir()
    (tmp_path / "n02106662" / "img.jpg").write_bytes(b"x")
    (tmp_path / "n02106662" / "n02106662_boxes.txt").write_bytes(b"0 0 1 1")
    result = list_image_paths(tmp_path)
    assert len(result) == 1
    assert result[0][0].name == "img.jpg"


def test_list_image_paths_split(tmp_path: Path) -> None:
    (tmp_path / "train" / "c1").mkdir(parents=True)
    (tmp_path / "val" / "c1").mkdir(parents=True)
    (tmp_path / "train" / "c1" / "a.jpg").write_bytes(b"x")
    (tmp_path / "val" / "c1" / "b.jpg").write_bytes(b"y")
    result_train = list_image_paths(tmp_path, split="train")
    result_val = list_image_paths(tmp_path, split="val")
    assert len(result_train) == 1 and result_train[0][0].name == "a.jpg"
    assert len(result_val) == 1 and result_val[0][0].name == "b.jpg"


def test_partition_paths() -> None:
    # 5500 items, shard_size 2000 -> 3 chunks (2000, 2000, 1500)
    paths_with_labels = [(Path(f"p{i}"), f"c{i % 10}") for i in range(5500)]
    chunks = list(partition_paths(paths_with_labels, 2000))
    assert len(chunks) == 3
    assert len(chunks[0]) == 2000
    assert len(chunks[1]) == 2000
    assert len(chunks[2]) == 1500


def test_write_one_shard_and_read_back(tmp_path: Path) -> None:
    # Create 10 tiny image files
    samples = []
    for i in range(10):
        p = tmp_path / "class_x" / f"img_{i:02d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100)  # minimal JPEG-like
        samples.append((p, "class_x"))
    out_tar = tmp_path / "out.tar"
    n = write_one_shard(samples, out_tar)
    assert n == 10
    assert out_tar.is_file()
    # Read back with WebDataset
    count = 0
    for sample in wds.WebDataset(str(out_tar)):
        count += 1
        assert "__key__" in sample
        assert "jpg" in sample or "cls" in sample
    assert count == 10


def test_write_shards_sequential(tmp_path: Path) -> None:
    root = tmp_path / "imgs"
    root.mkdir()
    for i in range(5):
        (root / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
    paths_with_labels = list_image_paths(root, with_labels=False)
    written = write_shards(paths_with_labels, tmp_path / "shards", shard_size=2, shard_prefix="s")
    assert len(written) == 3  # 2+2+1
    for w in written:
        assert w.exists()
    assert (tmp_path / "shards" / "s_00000.tar").exists()
    assert (tmp_path / "shards" / "s_00001.tar").exists()
    assert (tmp_path / "shards" / "s_00002.tar").exists()


def test_list_shards(tmp_path: Path) -> None:
    (tmp_path / "shard_00000.tar").write_bytes(b"")
    (tmp_path / "shard_00001.tar").write_bytes(b"")
    (tmp_path / "other.txt").write_bytes(b"")
    shards = list_shards(tmp_path)
    assert len(shards) == 2
    assert all(p.suffix == ".tar" for p in shards)
    names = sorted(p.name for p in shards)
    assert names == ["shard_00000.tar", "shard_00001.tar"]


def test_write_metadata_parquet(tmp_path: Path) -> None:
    """Metadata Parquet has shard, shard_idx, sample_idx, __key__, cls, source; loadable by dd.read_parquet."""
    samples = [
        (Path("train/c1/a.jpg"), "c1"),
        (Path("train/c1/b.jpg"), "c1"),
        (Path("train/c2/c.jpg"), "c2"),
    ]
    out = write_metadata_parquet(samples, shard_size=2, output_dir=tmp_path, split_name="train")
    assert out == tmp_path / "metadata" / "part.0.parquet"
    assert out.is_file()
    table = pq.read_table(out)
    assert table.column_names == ["shard", "shard_idx", "sample_idx", "__key__", "cls", "source"]
    assert len(table) == 3
    assert table.column("shard")[0].as_py() == "train_00000.tar"
    assert table.column("shard")[2].as_py() == "train_00001.tar"
    assert table.column("sample_idx")[0].as_py() == 0
    assert table.column("sample_idx")[1].as_py() == 1
    assert table.column("__key__")[0].as_py() == "a"
    assert table.column("cls")[0].as_py() == "c1"
    assert "train/c1/a.jpg" in table.column("source")[0].as_py()
