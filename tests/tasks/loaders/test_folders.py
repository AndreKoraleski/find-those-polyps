import http.server
import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from find_those_polyps.data import LocalFiles, RemoteFiles, read_index
from find_those_polyps.registry import DatasetCard, Download
from find_those_polyps.tasks.loaders.folders import (
    FolderClassification,
    MaskSegmentation,
    TableClassification,
)

Server = tuple[str, Path, http.server.ThreadingHTTPServer]


def _png(value: int, mask: bool = False) -> bytes:
    array = np.full((6, 8) if mask else (6, 8, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _tree(root: Path, files: dict[str, bytes]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


@pytest.fixture
def classes(tmp_path: Path) -> LocalFiles:
    files = {
        f"set/{folder}/{video}_{frame}.png": _png(40 * index)
        for index, folder in enumerate(["a", "b", "c"])
        for video in ("v1", "v2")
        for frame in range(3)
    }
    files["set/b/clash.png"] = _png(1)
    files["set/c/clash.png"] = _png(2)
    files["set/ignored/x.png"] = _png(3)
    files["split.csv"] = b"file;fold\nv1_0.png;1\nv2_0.png;0\nclash.png;1\n"
    _tree(tmp_path / "toy", files)
    return LocalFiles("toy", tmp_path / "toy")


def test_folder_classes_merge_and_ignore_folders(classes: LocalFiles) -> None:
    loader = FolderClassification(classes, root="set", classes=[["a"], ["b", "c"]])
    labels = loader.labels()
    assert len(labels) == 20
    assert labels["a/v1_0"] == 0
    assert labels["b/clash"] == labels["c/v2_2"] == 1
    assert "x" not in labels
    sample = loader.load("b/clash")
    assert sample.target == 1
    assert sample.path == "toy/set/b/clash.png"


def test_groups_and_official_test(classes: LocalFiles) -> None:
    test = {"file": "split.csv", "delimiter": ";", "name": "file", "column": "fold", "value": "1"}
    loader = FolderClassification(
        classes,
        root="set",
        classes=[["a"], ["b"]],
        group=r"^(?:\w/)?(v\d)_",
        official_test=test | {"source": "toy"},
    )
    assert loader.groups()["a/v2_1"] == "v2"
    official = loader.official_test()
    assert official == (["a/v1_0", "b/v1_0", "clash"], "toy")
    assert FolderClassification(classes, root="set", classes=[["a"]]).official_test() is None


def test_mask_pairs_by_stem(tmp_path: Path) -> None:
    files = {f"img/p_BN3_{i}.jpg": _png(9) for i in range(3)}
    files |= {f"msk/p_BN3_{i}.png": _png(255 * (i > 0), mask=True) for i in range(2)}
    _tree(tmp_path / "seg", files)
    loader = MaskSegmentation(
        LocalFiles("seg", tmp_path / "seg"), images="img", masks="msk", group=r"^(.+_BN\d+)_\d+$"
    )
    assert loader.ids() == ["p_BN3_0", "p_BN3_1"]
    assert set(loader.groups().values()) == {"p_BN3"}
    assert loader.load("p_BN3_1").target.sum() == 48  # type: ignore[union-attr]


def _tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


INNER = {f"images/i{i}.png": _png(i) for i in range(4)} | {
    f"masks/i{i}.png": _png(255, mask=True) for i in range(4)
}


def test_local_tarballs_are_expanded_once(tmp_path: Path) -> None:
    _tree(tmp_path / "ins", {"pack/data.tar.gz": _tarball(INNER)})
    files = LocalFiles("ins", tmp_path / "ins")
    loader = MaskSegmentation(
        files, images="pack/images", masks="pack/masks", expand=["pack/data.tar.gz"]
    )
    assert len(loader.ids()) == 4
    assert files.expand("pack/data.tar.gz") == [f"pack/{name}" for name in INNER]
    assert ".expanded" not in " ".join(files.names())


@pytest.mark.parametrize("cache", [True, False])
def test_remote_tarballs_stream_into_cache_or_memory(
    range_server: Server, tmp_path: Path, cache: bool
) -> None:
    url, directory, _ = range_server
    with zipfile.ZipFile(directory / "ins.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("pack/data.tar.gz", _tarball(INNER))
        bundle.writestr("pack/readme.txt", "hello")
    card = DatasetCard.model_validate(
        {
            "key": "ins",
            "name": "Toy",
            "labelled": True,
            "modality": "m",
            "licence": "l",
            "access": "open",
            "url": url,
            "citation": "c",
            "provenance": "p",
            "family": "f",
            "downloads": [
                Download(
                    url=f"{url}/ins.zip",
                    filename="ins.zip",
                    index=read_index(f"{url}/ins.zip").digest,
                )
            ],
        }
    )
    files = RemoteFiles(card, tmp_path / "ins" if cache else None)
    loader = MaskSegmentation(
        files, images="pack/images", masks="pack/masks", expand=["pack/data.tar.gz"]
    )
    samples = list(loader.stream(loader.ids()))
    assert sorted(s.id for s in samples) == ["i0", "i1", "i2", "i3"]
    assert (tmp_path / "ins" / "pack/images/i2.png").exists() == cache
    if cache:
        again = RemoteFiles(card, tmp_path / "ins")
        assert "pack/masks/i3.png" in again.names()


def test_nested_class_folders_with_path_ids_and_external_test(tmp_path: Path) -> None:
    files = {
        f"Images/Center-{c}/{m}/{k}/{k}-{lesion}/{n}.tiff": _png(c)
        for c in (1, 2)
        for m in ("WLE-Set", "EUS-Set")
        for k in ("Lipoma", "NET")
        for lesion in range(2)
        for n in range(2)
    }
    _tree(tmp_path / "sel", files)
    loader = FolderClassification(
        LocalFiles("sel", tmp_path / "sel"),
        root="Images",
        classes=[["Lipoma"], ["NET"]],
        depth=2,
        include="/WLE-Set/",
        id_mode="path",
        group=r"^(.+)/[^/]+$",
        official_test={"match": r"^Center-2/", "source": "external"},
    )
    labels = loader.labels()
    assert len(labels) == 16
    assert labels["Center-1/WLE-Set/NET/NET-1/0"] == 1
    assert loader.groups()["Center-1/WLE-Set/NET/NET-1/0"] == "Center-1/WLE-Set/NET/NET-1"
    official = loader.official_test()
    assert official is not None
    assert len(official[0]) == 8
    assert all(key.startswith("Center-2/") for key in official[0])


def test_table_labels_with_columns_one_hot_and_plain_lists(tmp_path: Path) -> None:
    files = {f"imgs/p{p}/f{p}{i}.jpg": _png(i) for p in range(2) for i in range(3)}
    files["labels.csv"] = b"pid,filename,grade,set\n" + b"".join(
        f"{p},f{p}{i}.jpg,{'AB'[i % 2]},{'Test' if p else 'Train'}\n".encode()
        for p in range(2)
        for i in range(3)
    )
    files["hot.csv"] = b"name,s0,s1\nf00.jpg,1,0\nf01.jpg,0,1\nf02.jpg,0,0\n"
    files["test.txt"] = b"f00\nf11\n"
    _tree(tmp_path / "tab", files)
    files_view = LocalFiles("tab", tmp_path / "tab")
    loader = TableClassification(
        files_view,
        tables=["labels.csv"],
        key="filename",
        root="imgs",
        label="grade",
        classes=[["A"], ["B"]],
        group_column="pid",
        official_test={
            "file": "labels.csv",
            "name": "filename",
            "column": "set",
            "value": "Test",
            "source": "t",
        },
    )
    assert loader.labels() == {"f00": 0, "f01": 1, "f02": 0, "f10": 0, "f11": 1, "f12": 0}
    assert loader.groups()["f12"] == "1"
    assert loader.official_test() == (["f10", "f11", "f12"], "t")
    sample = loader.load("f01")
    assert (sample.target, sample.group, sample.path) == (1, "0", "tab/imgs/p0/f01.jpg")
    hot = TableClassification(
        files_view,
        tables=["hot.csv"],
        key="name",
        root="imgs",
        one_hot=[["s0"], ["s1"]],
        official_test={"file": "test.txt", "header": False, "source": "list"},
    )
    assert hot.labels() == {"f00": 0, "f01": 1}
    assert hot.official_test() == (["f00"], "list")
    with pytest.raises(ValueError, match="either"):
        TableClassification(files_view, tables=["hot.csv"], key="name")


def test_masks_found_by_renamed_stems(tmp_path: Path) -> None:
    files = {f"img/img- ({i}).png": _png(9) for i in range(3)}
    files |= {f"ann/ann- ({i}).png": _png(255, mask=True) for i in range(3)}
    _tree(tmp_path / "wce", files)
    loader = MaskSegmentation(
        LocalFiles("wce", tmp_path / "wce"), images="img", masks="ann", mask_name=("^img-", "ann-")
    )
    assert loader.ids() == ["img- (0)", "img- (1)", "img- (2)"]
    assert loader.load("img- (2)").target.all()  # type: ignore[union-attr]


def test_table_rows_with_conflicting_classes_are_excluded(tmp_path: Path) -> None:
    files = {f"imgs/f{i}.jpg": _png(i) for i in range(3)}
    files["a.csv"] = b"name,label\nf0.jpg,x\nf1.jpg,x\n"
    files["b.csv"] = b"name,label\nf1.jpg,y\nf2.jpg,y\n"
    _tree(tmp_path / "conf", files)
    loader = TableClassification(
        LocalFiles("conf", tmp_path / "conf"),
        tables=["a.csv", "b.csv"],
        key="name",
        root="imgs",
        label="label",
        classes=[["x"], ["y"]],
    )
    assert loader.labels() == {"f0": 0, "f2": 1}
    assert loader.conflicts == ["f1"]


def test_table_rows_without_the_group_column_are_their_own_groups(tmp_path: Path) -> None:
    files = {f"imgs/f{i}.jpg": _png(i) for i in range(4)}
    files["a.csv"] = b"name,label,pid\nf0.jpg,x,p1\nf1.jpg,x\n"
    files["b.csv"] = b"name,label\nf2.jpg,y\nf3.jpg,y\n"
    _tree(tmp_path / "grp", files)
    loader = TableClassification(
        LocalFiles("grp", tmp_path / "grp"),
        tables=["a.csv", "b.csv"],
        key="name",
        root="imgs",
        label="label",
        classes=[["x"], ["y"]],
        group_column="pid",
    )
    assert loader.groups() == {"f0": "p1", "f1": "f1", "f2": "f2", "f3": "f3"}
    assert loader.load("f3").group == "f3"


def test_tarball_members_with_windows_separators_cannot_escape(tmp_path: Path) -> None:
    _tree(tmp_path / "bad", {"pack.tar.gz": _tarball({r"..\..\escaped.txt": b"x"})})
    with pytest.raises(ValueError, match="unsafe"):
        LocalFiles("bad", tmp_path / "bad").expand("pack.tar.gz")
    assert not (tmp_path / "escaped.txt").exists()
