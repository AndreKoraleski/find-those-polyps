"""Synthetic datasets, tasks and encoders shared by the test suite."""

import functools
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from find_those_polyps.data import DatasetFiles, LocalFiles
from find_those_polyps.models import Encoder
from find_those_polyps.tasks import (
    Budget,
    Loader,
    Sample,
    SplitFile,
    Task,
    TaskKind,
    TaskSpec,
)


class MemoryLoader(Loader):
    """Loader serving samples held in memory."""

    samples: dict[str, Sample]

    def available(self) -> bool:
        return True

    def ids(self) -> list[str]:
        return list(self.samples)

    def groups(self) -> dict[str, str]:
        return {key: sample.group for key, sample in self.samples.items()}

    def load(self, sample_id: str) -> Sample:
        return self.samples[sample_id]


class ColourLoader(MemoryLoader):
    """Noisy images whose dominant channel is the class."""

    def __init__(self, files: DatasetFiles, n: int = 90, **options: Any) -> None:
        super().__init__(files, **options)
        rng = np.random.default_rng(0)
        self.samples = {}
        for i in range(n):
            label = i % 3
            image = rng.integers(0, 100, size=(24 + i % 5, 32, 3))
            image[..., label] += 120
            self.samples[f"c{i}"] = Sample(
                id=f"c{i}",
                image=image.astype(np.uint8),
                target=label,
                group=f"g{i // 2}",
                path=f"colour/c{i}.png",
            )


class TagLoader(MemoryLoader):
    """Images tagged with every channel brighter than a threshold."""

    def __init__(self, files: DatasetFiles, n: int = 80, **options: Any) -> None:
        super().__init__(files, **options)
        rng = np.random.default_rng(2)
        self.samples = {}
        for i in range(n):
            tags = np.array([(i >> bit) & 1 for bit in range(3)], dtype=np.int64)
            image = rng.integers(0, 60, size=(16, 16, 3))
            image[..., tags == 1] += 150
            self.samples[f"t{i}"] = Sample(
                id=f"t{i}", image=image.astype(np.uint8), target=tags, group=f"g{i}"
            )


class SquareLoader(MemoryLoader):
    """Dark images containing one bright square, segmented as foreground."""

    def __init__(self, files: DatasetFiles, n: int = 40, **options: Any) -> None:
        super().__init__(files, **options)
        rng = np.random.default_rng(1)
        self.samples = {}
        for i in range(n):
            image = rng.integers(0, 40, size=(48, 64, 3)).astype(np.uint8)
            mask = np.zeros((48, 64), dtype=np.int64)
            top, left = rng.integers(0, 24), rng.integers(0, 32)
            image[top : top + 20, left : left + 24] = 220
            mask[top : top + 20, left : left + 24] = 1
            self.samples[f"s{i}"] = Sample(
                id=f"s{i}", image=image, target=mask, group=f"g{i}", path=f"square/s{i}.png"
            )


def make_task(loader: MemoryLoader, spec: TaskSpec) -> Task:
    ids = loader.ids()
    n = len(ids)
    split = SplitFile(
        test_source="synthetic",
        train=tuple(ids[: n // 2]),
        val=tuple(ids[n // 2 : 3 * n // 4]),
        test=tuple(ids[3 * n // 4 :]),
    )
    return Task(spec=spec, loader=loader, split=split)


def make_spec(**fields: Any) -> TaskSpec:
    base: dict[str, Any] = {
        "description": "Synthetic task.",
        "loader": "memory",
        "splits": "synthetic.json",
    }
    return TaskSpec.model_validate(base | fields)


@pytest.fixture
def colour_task(tmp_path: Path) -> Task:
    return make_task(
        ColourLoader(LocalFiles("synthetic", tmp_path)),
        make_spec(
            id="cls/colour",
            dataset="colour",
            kind=TaskKind.CLASSIFICATION,
            classes=("red", "green", "blue"),
            primary_metric="balanced_accuracy",
            budgets=(Budget(kind="full"), Budget.model_validate("shots:2")),
        ),
    )


@pytest.fixture
def ordinal_task(tmp_path: Path) -> Task:
    return make_task(
        ColourLoader(LocalFiles("synthetic", tmp_path)),
        make_spec(
            id="ordinal/colour",
            dataset="colour",
            kind=TaskKind.ORDINAL,
            classes=("low", "mid", "high"),
            primary_metric="quadratic_kappa",
        ),
    )


@pytest.fixture
def tag_task(tmp_path: Path) -> Task:
    return make_task(
        TagLoader(LocalFiles("synthetic", tmp_path)),
        make_spec(
            id="multilabel/tags",
            dataset="tags",
            kind=TaskKind.MULTILABEL,
            classes=("red", "green", "blue"),
            primary_metric="macro_ap",
            budgets=(Budget(kind="full"), Budget.model_validate("frac:0.5")),
        ),
    )


@pytest.fixture
def square_task(tmp_path: Path) -> Task:
    return make_task(
        SquareLoader(LocalFiles("synthetic", tmp_path)),
        make_spec(
            id="seg/square",
            dataset="square",
            kind=TaskKind.SEGMENTATION,
            classes=("background", "square"),
            primary_metric="dice",
        ),
    )


class MeanColour(Encoder):
    """Mean colour as global feature and a pooled image as dense feature."""

    name = "mean-colour"
    description = "Channel means of the resized image."
    image_size = 32

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def postprocess(self, output: torch.Tensor) -> torch.Tensor:
        return output.mean(dim=(2, 3))

    def postprocess_dense(self, output: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(output, 4)


class Noise(Encoder):
    """Features independent of the input."""

    name = "noise"
    description = "Random features unrelated to the image."
    image_size = 8

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator().manual_seed(int(images.sum().item() * 1000))
        return torch.randn(len(images), 3, generator=generator)


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler honouring single byte range requests."""

    def log_message(self, message: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        header = self.headers.get("Range")
        if header is None or self.server.ignore_ranges:  # type: ignore[attr-defined]
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        start_text, _, end_text = header.removeprefix("bytes=").partition("-")
        start, end = int(start_text), min(int(end_text or len(data) - 1), len(data) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start : end + 1])


@pytest.fixture
def range_server(tmp_path: Path) -> Iterator[tuple[str, Path, http.server.ThreadingHTTPServer]]:
    """HTTP server publishing a directory, its base URL and the directory itself."""
    directory = tmp_path / "www"
    directory.mkdir()
    handler = functools.partial(RangeHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.ignore_ranges = False  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", directory, server
    server.shutdown()
    server.server_close()
