import hashlib
import zipfile
from pathlib import Path

import pytest

from find_those_polyps.data import ManualAccessError, extract, fetch
from find_those_polyps.data.__main__ import main
from find_those_polyps.registry import DatasetCard, Download


def _archive(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("images/a.txt", "a")
    return source, hashlib.md5(source.read_bytes()).hexdigest()


def _card(url: str, checksum: str, access: str = "open") -> DatasetCard:
    return DatasetCard(
        key="toy",
        name="Toy",
        labelled=True,
        modality="m",
        licence="CC BY 4.0",
        access=access,  # type: ignore[arg-type]
        url="https://example.org",
        citation="c",
        provenance="p",
        family="toy",
        downloads=(Download(url=url, filename="toy.zip", checksum=f"md5:{checksum}"),),
    )


def test_fetch_verifies_extracts_and_is_idempotent(tmp_path: Path) -> None:
    source, digest = _archive(tmp_path)
    root = tmp_path / "data"
    destination = fetch(_card(source.as_uri(), digest), root)
    assert (destination / "images" / "a.txt").read_text() == "a"
    assert (destination / ".complete").exists()
    source.unlink()
    assert fetch(_card("file:///missing", digest), root) == destination


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    source, _ = _archive(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        fetch(_card(source.as_uri(), "0" * 32), tmp_path / "data")


def test_restricted_datasets_are_not_fetched(tmp_path: Path) -> None:
    with pytest.raises(ManualAccessError, match=r"example.org"):
        fetch(_card("file:///x", "0", access="request"), tmp_path)


def test_extract_refuses_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        extract(archive, tmp_path / "out")


def test_cli_lists_and_rejects_unknown_keys(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    assert "limuc" in capsys.readouterr().out
    assert main(["fetch", "nonexistent"]) == 1
