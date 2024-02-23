import kornia as K
from torch.nn import Module
from torch import Tensor
from typing import Tuple
from kornia.enhance import equalize_clahe
import numpy as np
import torch
from PIL import Image


class EqualizeClahe(Module):
    def __init__(self, 
                clip_limit: float = 40.0,
                grid_size: Tuple[int, int] = (8, 8),
                slow_and_differentiable: bool = False
                 ) -> None:
        super().__init__()
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.slow_and_differentiable = slow_and_differentiable

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(clip_limit={self.clip_limit}, "
            f"grid_size={self.grid_size}, "
            f"slow_and_differentiable={self.slow_and_differentiable})"
        )

    def forward(self, input: Tensor) -> Tensor:
        # ref: https://kornia.readthedocs.io/en/latest/_modules/kornia/enhance/equalization.html#equalize_clahe
        return equalize_clahe(input, self.clip_limit, self.grid_size, self.slow_and_differentiable)
    


class HistogramNormalize:
    """Performs histogram normalization on numpy array and returns 8-bit image.

    Code was taken from lightly, but adpated to work with PIL image as input:
    https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_custom_augmentations.html
    who adapted it from Facebook:
    https://github.com/facebookresearch/CovidPrognosis

    """

    def __init__(self, number_bins: int = 256):
        self.number_bins = number_bins

    def __call__(self, image: np.array) -> Image:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        # Get the image histogram.
        image_histogram, bins = np.histogram(
            image.flatten(), self.number_bins, density=True
        )
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # Use linear interpolation of cdf to find new pixel values.
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
        pil_image = Image.fromarray(np.uint8(image_equalized.reshape(image.shape)))
        return pil_image