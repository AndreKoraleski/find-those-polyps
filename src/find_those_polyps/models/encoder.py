"""Encoder interface implemented by every benchmarked model.

The benchmark treats an encoder as a black box computing
`postprocess(forward(stack(preprocess(image) for image in batch)))`. Each stage may be overridden,
so no input resolution, normalization or output format is imposed beyond the shape of the final
features, (B, D) for global and (B, D, h, w) for dense features.
"""

import ast
import hashlib
import inspect
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from find_those_polyps.models.transforms import resize


def _code(source: str) -> str:
    """Syntax tree of source code without docstrings, so that documentation never matters.

    Comments, blank lines and formatting vanish in the tree as well. Fragments that do not parse
    on their own, such as a lambda cut from a longer line, are kept as written.

    Args:
        source: Source code of a class, function or line.

    Returns:
        A dump of the syntax tree, or `source` itself when it does not parse.
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def _source(stage: Any) -> str:
    """Code of a processing stage, or a description that changes whenever it may change.

    The source of the stage itself is preferred, then the source of its class followed by its
    representation, and finally its representation alone when no source can be read. Docstrings,
    comments and formatting are ignored, so that editing documentation keeps cached features.

    Args:
        stage: Module, function or other callable taking part in the pipeline.

    Returns:
        Text identifying the code of the stage.
    """
    for target in (stage, type(stage)):
        try:
            code = _code(inspect.getsource(target))
            return code + (repr(stage) if target is not stage else "")
        except (OSError, TypeError):
            continue
    return repr(stage)


def _weights(modules: Sequence[nn.Module]) -> Iterator[bytes]:
    """Names and raw bytes of every parameter and buffer, in a stable order.

    Args:
        modules: Modules whose state dictionaries are read in turn, each sorted by entry name.

    Yields:
        The encoded name of each entry followed by the bytes of its tensor on the CPU.
    """
    for module in modules:
        for name, tensor in sorted(module.state_dict().items()):
            yield name.encode()
            yield tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


class EncoderCard(BaseModel):
    """Mandatory metadata identifying an encoder on the leaderboard.

    Attributes:
        name: Leaderboard key of at most 100 characters, made of letters, digits and the
            characters `.`, `_`, `+` and `-`, starting with a letter or digit.
        description: One sentence summary of 1 to 200 characters.
        pretrained_on: Keys of the datasets and corpora seen in pretraining, empty when there were
            none and `None` when undeclared.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$", max_length=100)
    description: str = Field(min_length=1, max_length=200)
    pretrained_on: tuple[str, ...] | None = None


class Encoder(nn.Module, ABC):
    """Frozen image encoder evaluated as a black box.

    Subclasses implement `forward` and set `name` and `description`. Every task whose dataset
    overlaps `pretrained_on` is skipped. The default `preprocess` resizes to `image_size` without
    normalization and the default `postprocess` returns the output of `forward` unchanged, which
    must then be a tensor of shape (B, D). Dense tasks additionally require `postprocess_dense`.

    Attributes:
        name: Identifier keying the leaderboard, validated by `EncoderCard`.
        description: One sentence summary of the encoder.
        pretrained_on: Keys of the datasets and corpora seen in pretraining, empty when there were
            none and `None`, the default, when undeclared.
        image_size: Side in pixels of the square input of the default `preprocess`, 224 by
            default.
        preprocess_device: Device on which the default `preprocess` resizes, the host when `None`,
            the default. Resizing on the accelerator is faster for large images but slightly
            different, so a device enters the fingerprint and keeps its features apart.
    """

    name: str
    description: str
    pretrained_on: tuple[str, ...] | None = None
    image_size: int = 224
    preprocess_device: str | None = None

    def preprocess(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """Map an RGB image to a model input tensor.

        Args:
            image: Image of shape (H, W, 3) with values in [0, 255].

        Returns:
            By default a float tensor of shape (3, `image_size`, `image_size`) with values in
            [0, 1].
        """
        return resize(image, self.image_size, device=self.preprocess_device)

    @abstractmethod
    def forward(self, images: torch.Tensor) -> Any:
        """Raw model output for a batch of preprocessed images.

        Args:
            images: Stacked outputs of `preprocess`, of shape (B, ...) on the evaluation device.

        Returns:
            Any output that `postprocess` and `postprocess_dense` know how to read.
        """

    def postprocess(self, output: Any) -> torch.Tensor:
        """Global embedding computed from the raw model output.

        Args:
            output: Output of `forward` for a batch of B images.

        Returns:
            A tensor of shape (B, D), by default `output` itself.

        Raises:
            TypeError: If `output` is not a tensor and the method was not overridden.
        """
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{self.name}: override postprocess for non-tensor output")
        return output

    def postprocess_dense(self, output: Any) -> torch.Tensor:
        """Spatial feature map computed from the raw model output, required for dense tasks.

        Args:
            output: Output of `forward` for a batch of B images.

        Returns:
            A tensor of shape (B, D, h, w).

        Raises:
            NotImplementedError: If the encoder does not override it to provide dense features.
        """
        raise NotImplementedError(f"{self.name} does not provide dense features")

    @property
    def has_dense(self) -> bool:
        """Whether the encoder provides dense features through `postprocess_dense`."""
        return type(self).postprocess_dense is not Encoder.postprocess_dense

    def fingerprint(self) -> str:
        """Hash of the name, the input size, the weights and the code of every processing stage.

        Features cached on disk are keyed by it, so that changing the weights or the code of the
        encoder never reuses stale features. Stages whose code cannot be read are hashed by their
        representation, which may differ between processes and then disables the reuse. A
        `preprocess_device` enters the hash when set, leaving the fingerprints of encoders that
        preprocess on the host unchanged.

        Returns:
            The hexadecimal SHA-256 digest of those contents.
        """
        header = f"{self.name}:{self.image_size}"
        if self.preprocess_device is not None:
            header += f":{torch.device(self.preprocess_device).type}"
        digest = hashlib.sha256(header.encode())
        for source in self._sources():
            digest.update(source.encode())
        for chunk in _weights(self._weighted()):
            digest.update(chunk)
        return digest.hexdigest()

    def _sources(self) -> list[str]:
        """Code of the encoder classes between this one and the interface, most derived first."""
        classes = type(self).__mro__[: type(self).__mro__.index(Encoder)]
        return [_source(cls) for cls in classes]

    def _weighted(self) -> list[nn.Module]:
        """Modules whose parameters and buffers enter the fingerprint."""
        return [self]

    @property
    def card(self) -> EncoderCard:
        """Validated metadata of the encoder.

        Raises:
            pydantic.ValidationError: If the name or the description is missing or malformed.
        """
        return EncoderCard(
            name=getattr(self, "name", ""),
            description=getattr(self, "description", ""),
            pretrained_on=self.pretrained_on,
        )


class TorchEncoder(Encoder):
    """Encoder assembled from a module and optional processing callables.

    Stages left as `None` fall back to the defaults of `Encoder`. The README shows how DINOv2 is
    wrapped with `normalize`, `resize` and `patch_grid`.

    Attributes:
        module: Module or callable computing the raw output from a batch of inputs.
        owner: Module owning `module` when it is a bound method, registered as a submodule so
            that moving the encoder to a device or switching its mode reaches the weights, and
            `None` otherwise.
        name: Identifier keying the leaderboard.
        description: One sentence summary of the encoder.
        pretrained_on: Keys of the datasets and corpora seen in pretraining, or `None` when
            undeclared.
        image_size: Side in pixels of the square input of the default `preprocess`.
    """

    def __init__(
        self,
        module: nn.Module | Callable[[torch.Tensor], Any],
        *,
        name: str,
        description: str,
        image_size: int = 224,
        preprocess: Callable[[NDArray[np.uint8]], torch.Tensor] | None = None,
        postprocess: Callable[[Any], torch.Tensor] | None = None,
        postprocess_dense: Callable[[Any], torch.Tensor] | None = None,
        pretrained_on: Sequence[str] | None = None,
    ) -> None:
        """Wrap a module and its processing callables.

        Args:
            module: Module or callable mapping a batch of inputs to the raw output. When it is a
                bound method of a module, for instance `model.forward_features`, that module
                becomes a submodule, so it follows the encoder across devices and modes and its
                weights enter the fingerprint.
            name: Identifier keying the leaderboard.
            description: One sentence summary of the encoder.
            image_size: Side in pixels of the square input of the default `preprocess`, 224 by
                default.
            preprocess: Map from an (H, W, 3) uint8 image to an input tensor, resizing to
                `image_size` when `None`.
            postprocess: Map from the raw output to (B, D) embeddings, the identity when `None`.
            postprocess_dense: Map from the raw output to (B, D, h, w) feature maps, `None` when
                the encoder provides no dense features.
            pretrained_on: Keys of the datasets and corpora seen in pretraining, empty when there
                were none and `None` when undeclared.
        """
        super().__init__()
        self.module = module
        owner = getattr(module, "__self__", None)
        self.owner = owner if isinstance(owner, nn.Module) else None
        self.name = name
        self.description = description
        self.pretrained_on = None if pretrained_on is None else tuple(pretrained_on)
        self.image_size = image_size
        self._preprocess = preprocess
        self._postprocess = postprocess
        self._postprocess_dense = postprocess_dense

    @property
    def has_dense(self) -> bool:
        """Whether a `postprocess_dense` callable was given."""
        return self._postprocess_dense is not None

    def preprocess(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """Map an RGB image to a model input tensor with the given callable or the default.

        Args:
            image: Image of shape (H, W, 3) with values in [0, 255].

        Returns:
            The model input tensor of one image.
        """
        if self._preprocess is None:
            return super().preprocess(image)
        return self._preprocess(image)

    def forward(self, images: torch.Tensor) -> Any:
        """Raw output of the wrapped module for a batch of preprocessed images.

        Args:
            images: Stacked outputs of `preprocess`, of shape (B, ...).

        Returns:
            The output of `module` for `images`.
        """
        return self.module(images)

    def postprocess(self, output: Any) -> torch.Tensor:
        """Global embedding computed with the given callable or the default.

        Args:
            output: Output of `forward` for a batch of B images.

        Returns:
            A tensor of shape (B, D).

        Raises:
            TypeError: If no callable was given and `output` is not a tensor.
        """
        if self._postprocess is None:
            return super().postprocess(output)
        return self._postprocess(output)

    def postprocess_dense(self, output: Any) -> torch.Tensor:
        """Spatial feature map computed with the given callable.

        Args:
            output: Output of `forward` for a batch of B images.

        Returns:
            A tensor of shape (B, D, h, w).

        Raises:
            NotImplementedError: If no `postprocess_dense` callable was given.
        """
        if self._postprocess_dense is None:
            return super().postprocess_dense(output)
        return self._postprocess_dense(output)

    def _sources(self) -> list[str]:
        """Code of the encoder classes followed by the code of every given stage."""
        stages = (self.module, self._preprocess, self._postprocess, self._postprocess_dense)
        return super()._sources() + [_source(stage) for stage in stages if stage is not None]


def patch_grid(tokens: torch.Tensor, height: int | None = None) -> torch.Tensor:
    """Reshape a sequence of patch tokens into a spatial feature map.

    Class and register tokens must be removed beforehand. Without `height` the number of rows is
    the rounded square root of the number of tokens, which gives a square grid for square inputs.

    Args:
        tokens: Patch tokens of shape (B, N, D) in row major order.
        height: Number of rows h of the grid, inferred when `None` or zero.

    Returns:
        A tensor of shape (B, D, h, w) with w equal to N divided by h.

    Raises:
        ValueError: If N is not a multiple of the number of rows.
    """
    batch, count, dim = tokens.shape
    rows = height or round(count**0.5)
    if count % rows:
        raise ValueError(f"{count} tokens do not form a grid with {rows} rows")
    return tokens.transpose(1, 2).reshape(batch, dim, rows, count // rows)


def run(
    encoder: Encoder,
    images: Sequence[NDArray[np.uint8]],
    device: torch.device,
    *,
    dense: bool = False,
) -> torch.Tensor:
    """Apply the full encoder pipeline to a batch of raw images.

    Gradients are not disabled here, so callers wrap the call in `torch.inference_mode` when
    needed.

    Args:
        encoder: Encoder whose stages are applied in turn.
        images: B images of shape (H, W, 3) with values in [0, 255], possibly of different sizes.
        device: Device on which the stacked inputs are placed before `forward`.
        dense: Whether to return dense feature maps instead of global embeddings.

    Returns:
        A tensor of shape (B, D), or (B, D, h, w) when `dense` is set, on the device chosen by the
        encoder.

    Raises:
        ValueError: If the features do not have the expected number of dimensions or one row per
            image.
    """
    return encode(encoder, [encoder.preprocess(image) for image in images], device, dense=dense)


def encode(
    encoder: Encoder,
    inputs: Sequence[torch.Tensor],
    device: torch.device,
    *,
    dense: bool = False,
) -> torch.Tensor:
    """Apply `forward` and the postprocessing to a batch of preprocessed images.

    Separating this stage from `preprocess` lets callers prepare later batches while the device
    computes the current one.

    Args:
        encoder: Encoder whose stages are applied in turn.
        inputs: B outputs of `preprocess`, stacked and moved to `device`.
        device: Device on which the stacked inputs are placed before `forward`.
        dense: Whether to return dense feature maps instead of global embeddings.

    Returns:
        A tensor of shape (B, D), or (B, D, h, w) when `dense` is set.

    Raises:
        ValueError: If the features do not have the expected number of dimensions or one row per
            image.
    """
    batch = torch.stack(list(inputs)).to(device)
    output = encoder(batch)
    features = encoder.postprocess_dense(output) if dense else encoder.postprocess(output)
    expected = 4 if dense else 2
    if features.ndim != expected or features.shape[0] != len(inputs):
        shape = "(B, D, h, w)" if dense else "(B, D)"
        raise ValueError(f"{encoder.name} produced {tuple(features.shape)}, expected {shape}")
    return features
