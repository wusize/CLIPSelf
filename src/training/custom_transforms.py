import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop, InterpolationMode
from typing import Tuple
from torch import Tensor

class CustomRandomResize(nn.Module):

    def __init__(self, scale=(0.5, 2.0), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.min_scale, self.max_scale = min(scale), max(scale)
        self.interpolation = interpolation

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        new_size = [int(height * scale), int(width * scale)]
        img = F.resize(img, new_size, self.interpolation)

        return img


class CustomRandomCrop(RandomCrop):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """

        width, height = F.get_image_size(img)
        tar_h, tar_w = self.size

        tar_h = min(tar_h, height)
        tar_w = min(tar_w, width)
        i, j, h, w = self.get_params(img, (tar_h, tar_w))

        return F.crop(img, i, j, h, w)

class SimpleRandomCrop(nn.Module):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def forward(self, img):
        width, height = F.get_image_size(img)
        tar_size = min(width, height)
        i, j, h, w = self.get_params(img, (tar_size, tar_size))

        return F.crop(img, i, j, h, w)
