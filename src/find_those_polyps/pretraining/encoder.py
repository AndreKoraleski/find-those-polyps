"""Benchmark encoder of a backbone trained by the pretraining recipes.

`PretrainedViT` reads a backbone written by `find_those_polyps.pretraining.trainer.export` and
evaluates it as the DINOv2 reference is evaluated, with the class token as the global feature and
the patch tokens as the dense one. Its name and the corpora it declares come from the file, so
that an encoder never understates what its weights have seen. A team encoder is a factory around
it.

```python
from find_those_polyps.pretraining.encoder import PretrainedViT


def build() -> PretrainedViT:
    return PretrainedViT("runs/pretraining/vitb14-dino/vitb14-dino.safetensors")
```
"""

import json
from pathlib import Path

import timm
from safetensors import safe_open
from safetensors.torch import load_file
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer

from find_those_polyps.baselines.dinov2 import TokenEncoder, check_size
from find_those_polyps.baselines.resnet import DEFAULT_SIZE, sized
from find_those_polyps.pretraining.architecture import ARCHITECTURE, PATCH


def architecture(image_size: int) -> VisionTransformer:
    """Untrained backbone of the recipes for a fixed input size.

    Args:
        image_size: Side in pixels of the square input.

    Returns:
        The network without a classifier.

    Raises:
        TypeError: If timm does not build a vision transformer.
    """
    network = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=0, img_size=image_size)
    if not isinstance(network, VisionTransformer):
        raise TypeError(f"{ARCHITECTURE} is not a vision transformer")
    return network


class PretrainedViT(TokenEncoder):
    """ViT-B/14 trained by a pretraining recipe, read from an exported file.

    The name is the one recorded at export, followed by the input size when it is not 224 pixels.

    Args:
        weights: File written by `export`.
        image_size: Side in pixels of the square input, a multiple of fourteen. Defaults to 224.

    Attributes:
        pretrained_on: Corpora recorded at export, the one of the starting weights included.
        method: Name of the method that trained the backbone.

    Raises:
        FileNotFoundError: If the file does not exist, so that random weights are never evaluated
            under the name of a trained model.
        ValueError: If the file lacks the metadata of an export or does not match the backbone.
    """

    def __init__(self, weights: str | Path, image_size: int = DEFAULT_SIZE) -> None:
        check_size(image_size)
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"missing pretrained backbone {path}")
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
        if not {"name", "method", "pretrained_on"} <= metadata.keys():
            raise ValueError(f"{path} was not written by the pretraining export")
        network = architecture(image_size)
        state = load_file(path)
        side = image_size // PATCH
        state["pos_embed"] = resample_abs_pos_embed(
            state["pos_embed"], new_size=[side, side], num_prefix_tokens=network.num_prefix_tokens
        )
        result = network.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"{path} does not match {ARCHITECTURE}, missing {result.missing_keys[:5]}, "
                f"unexpected {result.unexpected_keys[:5]}"
            )
        method = metadata["method"]
        super().__init__(
            network,
            name=sized(metadata["name"], image_size),
            description=f"ViT-B/14 pretrained with {method} by the recipes of this benchmark.",
            image_size=image_size,
        )
        self.method = method
        self.pretrained_on = tuple(json.loads(metadata["pretrained_on"]))
