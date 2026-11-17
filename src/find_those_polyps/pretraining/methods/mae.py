"""Masked autoencoders, MAE of He et al. (2022).

The backbone encodes a random quarter of the patches of a frame, and a light decoder, which is
thrown away after training, reconstructs the pixels of the others from the encoded patches and a
shared mask token. The loss is the squared error on the hidden patches only.
"""

from collections.abc import Callable
from typing import Any

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from timm.models.vision_transformer import Block, VisionTransformer
from torch import nn

from find_those_polyps.pretraining.architecture import IMAGE_SIZE, embed, encode
from find_those_polyps.pretraining.augment import crop
from find_those_polyps.pretraining.methods.base import Method, Schedule


class MAESettings(BaseModel):
    """Settings of MAE.

    Attributes:
        image_size: Side in pixels of the view.
        mask_ratio: Share of the patches hidden from the backbone.
        decoder_width: Width of the decoder.
        decoder_depth: Number of blocks of the decoder.
        decoder_heads: Attention heads of the decoder.
        normalize_pixels: Whether each target patch is standardized by its own mean and variance.
        scale: Share of the frame area that the crop covers.
    """

    model_config = ConfigDict(frozen=True)

    image_size: int = IMAGE_SIZE
    mask_ratio: float = Field(default=0.75, gt=0, lt=1)
    decoder_width: int = Field(default=512, gt=0)
    decoder_depth: int = Field(default=8, gt=0)
    decoder_heads: int = Field(default=16, gt=0)
    normalize_pixels: bool = True
    scale: tuple[float, float] = (0.2, 1.0)


def sincos_positions(width: int, side: int) -> torch.Tensor:
    """Fixed sine and cosine position embeddings of a square grid.

    Args:
        width: Width of the embeddings, a multiple of four.
        side: Number of cells along each side of the grid.

    Returns:
        Embeddings of shape (`side` squared, `width`) in row major order, half of each for the
        row and half for the column.
    """
    frequencies = 1.0 / 10000 ** (torch.arange(width // 4, dtype=torch.float32) / (width // 4))
    cells = torch.arange(side, dtype=torch.float32)
    angles = cells[:, None] * frequencies[None, :]
    axis = torch.cat([angles.sin(), angles.cos()], dim=1)
    rows = axis[:, None, :].expand(side, side, -1)
    columns = axis[None, :, :].expand(side, side, -1)
    return torch.cat([rows, columns], dim=-1).reshape(side * side, width)


class MAE(Method):
    """MAE around a vision transformer.

    Args:
        backbone: Vision transformer that encodes the visible patches.
        settings: Settings of the method, those of the paper when `None`.

    Attributes:
        settings: Settings of the method.
        patch: Side in pixels of the patches of the backbone.
        project: Projection of the encoded tokens to the width of the decoder.
        mask_token: Learned token standing for every hidden patch.
        decoder: Blocks of the decoder.
        norm: Final normalization of the decoder.
        predict: Projection of the decoded tokens to the pixels of a patch.
        positions: Fixed position embeddings of the decoder, the class token first.
    """

    name = "mae"
    scratch = Schedule(
        epochs=400, base_lr=1.5e-4, weight_decay=0.05, warmup_epochs=40, betas=(0.9, 0.95)
    )
    continued = Schedule(
        epochs=50, base_lr=1.5e-5, weight_decay=0.05, warmup_epochs=5, betas=(0.9, 0.95)
    )

    positions: torch.Tensor

    def __init__(self, backbone: VisionTransformer, settings: MAESettings | None = None) -> None:
        super().__init__(backbone)
        self.settings = s = settings or MAESettings()
        self.patch = int(backbone.patch_embed.patch_size[0])
        if s.image_size % self.patch:
            raise ValueError(f"image_size must be a multiple of {self.patch}, got {s.image_size}")
        self.project = nn.Linear(backbone.embed_dim, s.decoder_width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, s.decoder_width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(
            *[
                Block(s.decoder_width, s.decoder_heads, mlp_ratio=4.0, qkv_bias=True)
                for _ in range(s.decoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(s.decoder_width)
        self.predict = nn.Linear(s.decoder_width, self.patch**2 * 3)
        grid = sincos_positions(s.decoder_width, s.image_size // self.patch)
        positions = torch.cat([torch.zeros(1, s.decoder_width), grid])
        self.register_buffer("positions", positions[None], persistent=False)

    def views(self) -> Callable[[Image.Image], Any]:
        """One random crop without photometric distortion.

        Returns:
            The augmentation, from a frame to one tensor.
        """
        pipeline: Callable[[Image.Image], Any] = crop(
            self.settings.image_size, self.settings.scale, colour=False
        )
        return pipeline

    def patches(self, images: torch.Tensor) -> torch.Tensor:
        """Pixels of every patch of a batch.

        Args:
            images: Images of shape (B, 3, S, S).

        Returns:
            Pixels of shape (B, N, patch squared times three), patches in row major order.
        """
        batch, channels, side, _ = images.shape
        cells = side // self.patch
        split = images.reshape(batch, channels, cells, self.patch, cells, self.patch)
        return split.permute(0, 2, 4, 3, 5, 1).reshape(batch, cells * cells, -1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Squared error of the reconstruction on the hidden patches.

        Args:
            batch: Views of shape (B, 3, S, S).

        Returns:
            The loss averaged over the hidden patches.
        """
        tokens = embed(self.backbone, batch)
        prefix = self.backbone.num_prefix_tokens
        patches = tokens[:, prefix:]
        count, total, width = patches.shape
        kept = max(1, round(total * (1 - self.settings.mask_ratio)))
        order = torch.rand(count, total, device=batch.device).argsort(dim=1)
        restore = order.argsort(dim=1)
        visible = patches.gather(1, order[:, :kept, None].expand(-1, -1, width))
        encoded = self.project(encode(self.backbone, torch.cat([tokens[:, :prefix], visible], 1)))
        hidden = self.mask_token.expand(count, total - kept, -1).to(encoded.dtype)
        shuffled = torch.cat([encoded[:, prefix:], hidden], dim=1)
        ordered = shuffled.gather(1, restore[:, :, None].expand(-1, -1, shuffled.shape[-1]))
        decoded = torch.cat([encoded[:, :1], ordered], dim=1) + self.positions
        predicted = self.predict(self.norm(self.decoder(decoded)))[:, 1:].float()
        target = self.patches(batch).float()
        if self.settings.normalize_pixels:
            mean = target.mean(dim=-1, keepdim=True)
            target = (target - mean) / (target.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        masked = (restore >= kept).float()
        error = ((predicted - target) ** 2).mean(dim=-1)
        loss: torch.Tensor = (error * masked).sum() / masked.sum()
        return loss
