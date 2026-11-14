"""Reference encoders that bound and validate the benchmark."""

from find_those_polyps.baselines.deit import ImageNetViT
from find_those_polyps.baselines.dinov2 import DINOv2, TokenEncoder
from find_those_polyps.baselines.gastronet import GastroNetDINOv2
from find_those_polyps.baselines.handcrafted import ColourTexture
from find_those_polyps.baselines.resnet import ImageNetResNet50, RandomResNet50, ResNet50Encoder

__all__ = [
    "ColourTexture",
    "DINOv2",
    "GastroNetDINOv2",
    "ImageNetResNet50",
    "ImageNetViT",
    "RandomResNet50",
    "ResNet50Encoder",
    "TokenEncoder",
]
