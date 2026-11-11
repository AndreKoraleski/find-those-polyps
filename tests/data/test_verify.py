import hashlib
import http.server
import zipfile
from pathlib import Path

import pytest

from find_those_polyps.data import __main__ as cli
from find_those_polyps.data import read_index
from find_those_polyps.data.verify import Outcome, verify, verify_download
from find_those_polyps.registry import DatasetCard, Download, load_registry

Server = tuple[str, Path, http.server.ThreadingHTTPServer]
NOTES = b"id,label\nk0,1\n"


def _publish(directory: Path, n: int = 6) -> None:
    with zipfile.ZipFile(directory / "set.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for i in range(n):
            bundle.writestr(f"images/{i:02d}.bin", bytes([i]) * (50 + 11 * i))
    (directory / "notes.csv").write_bytes(NOTES)


def _card(url: str, archive: str = "set.zip") -> DatasetCard:
    index = read_index(f"{url}/set.zip")
    items = (
        Download(url=f"{url}/{archive}", filename="set.zip", size=index.size, index=index.digest),
        Download(
            url=f"{url}/notes.csv",
            filename="notes.csv",
            checksum="sha256:" + hashlib.sha256(NOTES).hexdigest(),
            size=len(NOTES),
            extract=False,
        ),
    )
    return DatasetCard.model_validate(
        {
            "key": "toy",
            "name": "Toy",
            "labelled": True,
            "modality": "m",
            "licence": "CC BY 4.0",
            "access": "open",
            "url": url,
            "citation": "c",
            "provenance": "p",
            "family": "f",
            "downloads": items,
        }
    )


@pytest.fixture
def card(range_server: Server) -> DatasetCard:
    url, directory, _ = range_server
    _publish(directory)
    return _card(url)


def test_pinned_files_are_ok(card: DatasetCard) -> None:
    verdicts = list(verify([card]))
    assert [v.filename for v in verdicts] == ["set.zip", "notes.csv"]
    assert all(v.outcome is Outcome.OK and not v.detail for v in verdicts)
    assert str(verdicts[0]) == "ok           toy/set.zip"


def test_changed_archives_report_what_differs(card: DatasetCard, range_server: Server) -> None:
    _, directory, _ = range_server
    _publish(directory, n=7)
    verdict = verify_download(card.key, card.downloads[0])
    assert verdict.outcome is Outcome.CHANGED
    assert f"size {card.downloads[0].size} is now" in verdict.detail
    assert f"index {card.downloads[0].index} is now sha256:" in verdict.detail


def test_changed_plain_files_are_detected(card: DatasetCard, range_server: Server) -> None:
    _, directory, _ = range_server
    (directory / "notes.csv").write_bytes(NOTES.replace(b"1", b"2"))
    verdict = verify_download(card.key, card.downloads[1])
    assert verdict.outcome is Outcome.CHANGED
    assert "checksum sha256:" in verdict.detail
    assert "size" not in verdict.detail


def test_archives_replaced_by_other_content_are_changed(
    card: DatasetCard, range_server: Server
) -> None:
    _, directory, _ = range_server
    (directory / "set.zip").write_bytes(b"<html>moved</html>")
    verdict = verify_download(card.key, card.downloads[0])
    assert verdict.outcome is Outcome.CHANGED
    assert "not a zip archive" in verdict.detail


def test_dead_urls_are_unreachable_after_retries(
    range_server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, directory, _ = range_server
    _publish(directory)
    card = _card(url, archive="moved.zip")
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)
    verdict = verify_download(card.key, card.downloads[0])
    assert verdict.outcome is Outcome.UNREACHABLE
    assert "404" in verdict.detail
    assert delays == [2, 4, 8, 16]


def test_cli_reports_every_file_and_fails_on_changes(
    card: DatasetCard,
    range_server: Server,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, directory, _ = range_server
    registry = load_registry().model_copy(update={"datasets": (card,)})
    monkeypatch.setattr(cli, "load_registry", lambda: registry)
    assert cli.main(["verify"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "ok           toy/set.zip",
        "ok           toy/notes.csv",
    ]
    (directory / "notes.csv").write_bytes(b"changed\n")
    assert cli.main(["verify", "toy"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("ok ")
    assert lines[1].startswith("changed ")
    assert cli.main(["verify", "missing"]) == 1
    assert "unknown dataset" in capsys.readouterr().err
