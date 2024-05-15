# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
from PIL import Image
from torchvision.datasets.vision import VisionDataset

__noise_types__ = [
    "brightness",
    "defocus_blur",
    "frost",
    "glass_blur",
    "saturate",
    "spatter",
    "elastic_transform",
    "gaussian_blur",
    "impulse_noise",
    "motion_blur",
    "shot_noise",
    "speckle_noise",
    "contrast",
    "fog",
    "gaussian_noise",
    "jpeg_compression",
    "pixelate",
    "snow",
    "zoom_blur",
]

__labels__ = [
    "labels",
]


class CIFAR10C(VisionDataset):
    """ CIFAR10-C dataset from Hendrycks et al. (2019)
        https://arxiv.org/abs/1903.12261

        Retrieved from https://zenodo.org/records/2535967
    """
    def __init__(
        self,
        root: Union[str, Path],
        noise_type: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super(CIFAR10C, self).__init__(root, transforms, transform, target_transform)

        if noise_type not in __noise_types__:
            raise ValueError(f"Unsupported noise type: {noise_type}. Allowed noise types: {__noise_types__}")

        img_path = os.path.join(root, noise_type + '.npy')
        if not os.path.exists(img_path):
            raise RuntimeError(f"Dataset file not found: {img_path}")

        label_path = os.path.join(root,'labels.npy')
        if not os.path.exists(label_path):
            raise RuntimeError(f"Labels file not found: {label_path}")

        self._init_data(img_path, label_path)


    def _init_data(self, img_path, label_path):
    #    img = torch.as_tensor(
    #        np.load(img_path, allow_pickle=True)
    #    ).float()

        img = np.load(img_path, allow_pickle=True)
        self.data = img.reshape(-1, 32, 32, 3) # HWC

        self.targets = torch.as_tensor(
            np.load(label_path, allow_pickle=True)
        ).long()

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
