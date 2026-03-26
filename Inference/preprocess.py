"""
Image preprocessing for inference.
Handles loading from file path or PIL Image and converting to model-ready tensor.
"""

import os
from typing import Union

import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.config import IMAGE_SIZE, DEVICE
from Core.utils import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Converts a raw image into a normalized tensor ready for MobileNetV3.

    Usage:
        preprocessor = ImagePreprocessor()
        tensor = preprocessor.load_and_transform("path/to/image.jpg")
    """

    def __init__(self, image_size: int = IMAGE_SIZE) -> None:
        self._transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        logger.debug("ImagePreprocessor initialized with size=%d", image_size)

    def load_and_transform(self, image_path: str) -> torch.Tensor:
        """
        Load an image from disk and return a batched tensor on DEVICE.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            Tensor of shape [1, 3, H, W] on the configured device.

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If the file cannot be opened as an image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

        logger.debug("Loaded image: %s", image_path)
        return self.transform_pil(image)

    def transform_pil(self, image: Image.Image) -> torch.Tensor:
        """
        Transform an already-opened PIL Image to a batched tensor.

        Args:
            image: PIL Image (will be converted to RGB if needed).

        Returns:
            Tensor of shape [1, 3, H, W] on the configured device.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self._transform(image).unsqueeze(0).to(DEVICE)
        return tensor
