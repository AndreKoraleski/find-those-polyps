"""Linear heads of several repeats of a dense probe, trained side by side as one module.

Each repeat trains its own batch normalization and convolution on its own batches. A head this
small leaves an accelerator mostly idle, so the heads of all repeats are stacked along the
channels, with one convolution group per repeat, and run in the kernels of a single head. The
repeats stay independent, since batch normalization treats every channel apart and a grouped
convolution connects no two groups. Each repeat follows the arithmetic it would follow alone, up
to the rounding of the larger kernels.
"""

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn


class StackedHeads(nn.Module):
    """Heads of batch normalization and convolution of several repeats, run side by side.

    The input holds the batch of every repeat, of shape `(B, S, D, h, w)` for `S` repeats with
    `D` channels, and the output the logits of every repeat, of shape `(B, S, C, h, w)`. The
    parameters of repeat `s` are the slices of the stacked ones at position `s`, so a single
    optimizer acting elementwise updates every repeat as its own optimizer would.

    Args:
        heads: Heads of equal shapes, each a batch normalization followed by a convolution,
            whose current parameters and statistics the stacked head copies.

    Attributes:
        repeats: Number of repeats `S`.
        dim: Number of input channels `D` of every repeat.
        out: Number of output channels `C` of every repeat.
        kernel: Kernel size of the convolution of every repeat.
        padding: Padding of the convolution of every repeat.
        norm: Batch normalization over the `S * D` stacked channels.
        conv: Convolution from `S * D` to `S * C` channels in `S` groups.
    """

    def __init__(self, heads: Sequence[nn.Sequential]) -> None:
        super().__init__()
        norm, conv = heads[0][0], heads[0][1]
        if not isinstance(norm, nn.BatchNorm2d) or not isinstance(conv, nn.Conv2d):
            raise TypeError("every head must be a batch normalization followed by a convolution")
        self.repeats = len(heads)
        self.dim = norm.num_features
        self.out = conv.out_channels
        self.kernel = cast(tuple[int, int], conv.kernel_size)
        self.padding = cast(tuple[int, int], conv.padding)
        self.norm = nn.BatchNorm2d(self.repeats * self.dim, eps=norm.eps, momentum=norm.momentum)
        self.conv = nn.Conv2d(
            self.repeats * self.dim,
            self.repeats * self.out,
            self.kernel,
            padding=self.padding,
            groups=self.repeats,
            bias=conv.bias is not None,
        )
        states = [head.state_dict() for head in heads]
        merged = {}
        for key, value in states[0].items():
            module, name = key.split(".")
            stacked = value if value.ndim == 0 else torch.cat([state[key] for state in states])
            merged[f"{'norm' if module == '0' else 'conv'}.{name}"] = stacked
        self.load_state_dict(merged)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Logits of every repeat on its own batch.

        A pointwise convolution is computed as a matrix product broadcast over the images and
        repeats, which reads the features in place and runs far faster than a grouped
        convolution on accelerators. Wider kernels go through the grouped convolution.

        Args:
            values: Features of shape `(B, S, D, h, w)`, the batch of repeat `s` at position `s`.

        Returns:
            Logits of shape `(B, S, C, h, w)`.
        """
        normalized = self.norm(values.flatten(1, 2))
        if self.kernel != (1, 1):
            logits: torch.Tensor = self.conv(normalized)
            return logits.view(len(logits), self.repeats, self.out, *logits.shape[2:])
        batch, _, _, height, width = values.shape
        weight = self.conv.weight.view(self.repeats, self.out, self.dim)
        logits = torch.matmul(weight, normalized.view(batch, self.repeats, self.dim, -1))
        if self.conv.bias is not None:
            logits = logits + self.conv.bias.view(1, self.repeats, self.out, 1)
        return logits.view(batch, self.repeats, self.out, height, width)

    def member(self, index: int) -> nn.Sequential:
        """Copy of the head of one repeat, as it would stand after training alone.

        The copy is built without initializing its parameters, so it draws nothing from the
        random generators.

        Args:
            index: Position `s` of the repeat.

        Returns:
            A batch normalization followed by a convolution on the device of the stacked head,
            holding copies of the parameters and statistics of the repeat.
        """
        device = self.conv.weight.device
        head = nn.Sequential(
            nn.BatchNorm2d(self.dim, eps=self.norm.eps, momentum=self.norm.momentum, device="meta"),
            nn.Conv2d(
                self.dim,
                self.out,
                self.kernel,
                padding=self.padding,
                bias=self.conv.bias is not None,
                device="meta",
            ),
        ).to_empty(device=device)
        state = {}
        for key, value in self.state_dict().items():
            module, name = key.split(".")
            size = self.dim if module == "norm" else self.out
            part = value if value.ndim == 0 else value[index * size : (index + 1) * size]
            state[f"{'0' if module == 'norm' else '1'}.{name}"] = part.detach().clone()
        head.load_state_dict(state)
        return head
