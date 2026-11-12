"""Reference encoders that bound and validate the benchmark."""

from find_those_polyps.baselines.dinov2 import DINOv2
from find_those_polyps.baselines.resnet import ImageNetResNet50, RandomResNet50, ResNet50Encoder

__all__ = ["DINOv2", "ImageNetResNet50", "RandomResNet50", "ResNet50Encoder"]
