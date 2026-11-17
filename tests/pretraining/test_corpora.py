import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from find_those_polyps.pretraining import Archive, CorpusCard, corpus, load_corpora
from find_those_polyps.registry import load_registry


def _card(**changes: object) -> dict[str, object]:
    archives = [
        {
            "unit": f"v{i}",
            "url": f"https://example.org/v{i}.tar.gz",
            "filename": f"v{i}.tar.gz",
            "checksum": "md5:" + "0" * 32,
            "size": 10,
            "frames": 100,
            "fps": 25.0,
        }
        for i in range(3)
    ]
    card: dict[str, object] = {
        "key": "toy",
        "name": "Toy",
        "modality": "colonoscopy",
        "licence": "CC BY 4.0",
        "url": "https://example.org",
        "citation": "c",
        "provenance": "p",
        "family": "toy",
        "frame_pattern": r"v\d_(\d+)\.jpg",
        "archives": archives,
        "held_out": [{"units": ["v2"], "reason": "test", "evidence": "https://example.org/split"}],
    }
    return card | changes


def test_held_out_units_are_never_selected() -> None:
    card = CorpusCard.model_validate(_card())
    assert [archive.unit for archive in card.selected()] == ["v0", "v1"]
    assert card.excluded == {"v2"}


def test_frames_are_named_by_the_pattern() -> None:
    card = CorpusCard.model_validate(_card())
    assert card.frame_index("v0_frames/v0_12.jpg") == 12
    assert card.frame_index("v0_12.jpg") == 12
    assert card.frame_index("v0_frames/readme.txt") is None
    assert card.frame_index("v0_frames/v0_12.jpg.bak") is None


@pytest.mark.parametrize(
    ("fps", "native", "stride"),
    [(1.0, 29.97, 30), (1.0, 25.0, 25), (5.0, 60.0, 12), (None, 25.0, 1), (1.0, None, 1)],
)
def test_strides_follow_the_recording_rate(
    fps: float | None, native: float | None, stride: int
) -> None:
    archive = Archive(unit="v", url="u", filename="v.tar.gz", checksum="md5:00", size=1, fps=native)
    assert archive.stride(fps) == stride
    assert archive.stride(1000.0) == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"held_out": [{"units": ["v9"], "reason": "r", "evidence": "e"}]}, "unknown units"),
        ({"frame_pattern": r"v\d_\d+\.jpg"}, "capture the frame index"),
        ({"key": "Toy"}, "key"),
    ],
)
def test_inconsistent_cards_are_refused(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        CorpusCard.model_validate(_card(**changes))


def test_repeated_units_and_unpinned_archives_are_refused() -> None:
    card = _card()
    archives = card["archives"]
    assert isinstance(archives, list)
    with pytest.raises(ValidationError, match="unit twice"):
        CorpusCard.model_validate(card | {"archives": [*archives, archives[0]]})
    with pytest.raises(ValidationError, match="checksum"):
        CorpusCard.model_validate(card | {"archives": [archives[0] | {"checksum": ""}]})


def test_corpus_keys_are_unique(tmp_path: Path) -> None:
    path = tmp_path / "corpora.json"
    path.write_text(json.dumps([_card(), _card()]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_corpora(path)


def test_packaged_corpora_are_pinned_and_apart_from_the_benchmark() -> None:
    cards = load_corpora()
    datasets = {card.key for card in load_registry().datasets}
    assert cards
    for card in cards:
        assert card.key not in datasets
        assert card.selected()
        for archive in card.archives:
            assert archive.url.startswith("https://")
            assert archive.frames and archive.fps
    with pytest.raises(KeyError, match="known corpora"):
        corpus("absent")


def test_real_colon_holds_out_its_official_test_videos() -> None:
    card = corpus("real-colon-train")
    assert len(card.archives) == 60
    assert card.excluded == {
        f"00{study}-0{video}" for study in range(1, 5) for video in (13, 14, 15)
    }
    selected = card.selected()
    assert len(selected) == 48
    assert sum(archive.frames or 0 for archive in card.archives) == 2757723
    assert sum(archive.frames or 0 for archive in selected) == 2166076
    assert card.frame_index("001-001_frames/001-001_8903.jpg") == 8903
    assert card.licence == "CC BY 4.0"
