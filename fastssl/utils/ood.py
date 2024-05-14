# -*- coding: utf-8 -*-

import os
import torch
import numpy as np

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


def CIFAR10C(root, noise_type, train=False, download=False, transform=None):
    """ CIFAR10-C dataset from Hendrycks et al. (2019)
        https://arxiv.org/abs/1903.12261

        Retrieved from https://zenodo.org/records/2535967
    """
    if train:
        raise ValueError("Only validation split is available")
    if download:
        raise NotImplementedError(
            "Please download dataset manually from https://zenodo.org/records/2535967"
        )
    if transform:
        raise NotImplementedError(
            "Transforms are currently not supported at the moment")
        )

    if noise_type not in __noise_types__:
        raise ValueError(f"Unsupported noise type: {noise_type}. Allowed noise types: {__noise_types__}")

    img_path = os.path.join(root, noise_type + '.npy.')
    if not os.path.exists(img_path):
        raise RuntimeError(f"Dataset file not found: {img_path}")

    label_path = os.path.join(root,'labels.npy.')
    if not os.path.exists(label_path):
        raise RuntimeError(f"Labels file not found: {label_path}")

     img = torch.as_tensor(
         np.load(img_path, allow_pickle=True)
     ).float()

     labels = torch.as_tensor(
         np.load(label_path, allow_pickle=True)
     ).long()

     img = img[:, None, ...].transpose(4, 1)

     dset = torch.utils.data.TensorDataset(img, labels.view(-1))

     return dset

