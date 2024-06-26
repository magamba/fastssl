"""
## NOTE: Supports FFCV and PyTorch dataloaders. 
* linear classifier: 
    * FP16 is sufficient, reasonable speed and little drop in accuracy (Acc@1 is within +/- 0.1)
* SSL : Seems like autocast is important for good performance.
"""
from typing import List 

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
import numpy as np
import torch
import torchvision.transforms as tvt
from ffcv.transforms import RandomResizedCrop, RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

from fastssl.data.cifar_transforms import STLTransform, SSLPT_STL, ReScale, STLClassifierTransform, STLTransformFFCV

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

def to_device(device):
    if device == 'cuda:0':
        return ToDevice(device, non_blocking=True)
    else:
        return ToDevice("cpu")

def gen_image_pipeline(device="cuda:0", transform_cls=None, rescale=False):
    image_pipeline : List[Operation] = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        to_device(device),
        ToTorchImage(),
        Convert(torch.float32),
    ]
    # no rescaling required anymore!
    # if rescale:
    #     image_pipeline.append(ReScale(1.0/255.0))
    image_pipeline.append(transform_cls())

    return image_pipeline

def gen_image_pipeline_ffcv_ssl(device="cuda:0", transform_cls=None, rescale=False):
    if transform_cls:
        image_pipeline: List[Operation] = [
            RandomResizedCropRGBImageDecoder(
                output_size=(
                    transform_cls.dataset_side_length,
                    transform_cls.dataset_side_length,
                ),
                scale=transform_cls.dataset_resize_scale,
                ratio=transform_cls.dataset_resize_ratio,
            )
        ]

        image_pipeline.extend(transform_cls.transform_list)

    else:
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    image_pipeline.extend(
        [
            ToTensor(),
            to_device(device),
            ToTorchImage(),
            Convert(torch.float32),
        ]
    )

    return image_pipeline

def gen_image_pipeline_ffcv_test(device="cuda:0", transform_cls=None, rescale=False):
    # image_pipeline : List[Operation] = [SimpleRGBImageDecoder()]
    image_pipeline : List[Operation] = [RandomResizedCropRGBImageDecoder(
                                        output_size=(transform_cls.dataset_side_length,transform_cls.dataset_side_length),
                                        scale=transform_cls.dataset_resize_scale,ratio=transform_cls.dataset_resize_ratio)]
    if transform_cls:
        image_pipeline.extend(transform_cls.transform_list)

    image_pipeline.extend([
        ToTensor(),
        to_device(device),
        ToTorchImage(),
        Convert(torch.float32),
    ])

    return image_pipeline

def gen_label_pipeline(device="cuda:0", transform_cls=None):
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        ToDevice("cuda:0"),
        Squeeze()]
    return label_pipeline

def gen_image_label_pipeline(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    transform_cls: STLClassifierTransform = None,
    rescale: bool = False,
    device: str = 'cuda:0',
    num_augmentations: int = 1,
    label_noise: int = 0
    ):
    """
    Args:
        train_dataset (str, optional): path to train dataset. Defaults to None.
        val_dataset (str, optional): path to test dataset. Defaults to None.
        batch_size (int, optional): batch-size. Defaults to None.
        num_workers (int, optional): number of CPU workers. Defaults to None.
        transform_cls (STLClassifierTransform, optional): Transforms to be applied for the original image. Defaults to None.
        rescale (bool, optional): Flag to rescale pixel vals to [0,1]. Defaults to False.
        device (_type_, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of total image augmentations. Defaults to 1.
        transform_cls_augs (STLTransformFFCV, optional): Transforms to be applied to generate other augmentations. Defaults to None.
        label_noise (int, optional): Ratio (percentage) of label noise (int from 0 to 100)
    Returns
        loaders       : dict('train': dataloader, 'test': dataloader)
    """

    datadir = {
        'train': train_dataset,
        'test': val_dataset
    }

    loaders = {}

    for split in ['train', 'test']:
        if datadir[split] is None: continue
        label_pipeline  = gen_label_pipeline(device=device)
        image_pipeline = gen_image_pipeline(
            device=device, transform_cls=transform_cls, rescale=rescale
        )
        if num_augmentations > 1:
            image_pipeline_augs = [
                gen_image_pipeline_ffcv_ssl(
                    device=device, transform_cls=transform_cls_augs, rescale=rescale
                )
            ] * (num_augmentations - 1)
        else:
            image_pipeline_augs = []
        ordering = OrderOption.RANDOM if split == 'train' else OrderOption.SEQUENTIAL
        # ordering = OrderOption.RANDOM #if split == 'train' else OrderOption.SEQUENTIAL

        pipelines = {'image' : image_pipeline, 'label' : label_pipeline}
        if label_noise > 0 and split == "train":
            pipelines.update({'ground_truth': label_pipeline, 'sample_idx': label_pipeline})
        
        custom_field_img_mapper = {}
        for i, aug_pipeline in enumerate(image_pipeline_augs):
            pipelines["image{}".format(i + 1)] = aug_pipeline
            custom_field_img_mapper["image{}".format(i + 1)] = "image"
        
        loaders[split] = Loader(
            datadir[split],
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            pipelines=pipelines,
            custom_field_mapper=custom_field_img_mapper,
           )
    return loaders

def gen_image_label_pipeline_ffcv_ssl_test(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    transform_cls: STLTransformFFCV = None,
    rescale: bool = False,
    device: str = "cuda:0",
    num_augmentations: int = 2,
):
    """Test function for generating multiple augmentations from each image.

    Args:
        train_dataset (str, optional): Train dataset filename. Defaults to None.
        val_dataset (str, optional): Test dataset filename. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to None.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        transform_cls (STLTransformFFCV, optional): Transform object. Defaults to None.
        rescale (bool, optional): Flag to rescale pixel vals to [0,1]. Defaults to False.
        device (_type_, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of patches. Defaults to 2.

    Returns:
        loaders: dict('train': dataloader, 'test': dataloader)
    """
    datadir = {"train": train_dataset, "test": val_dataset}
    assert num_augmentations > 1, "Please use at least 2 augmentations for SSL."

    loaders = {}
    for split in ["train", "test"]:
        if datadir[split] is None: continue
        image_pipeline_og = gen_image_pipeline(device=device, rescale=rescale)
        label_pipeline = gen_label_pipeline(device=device)
        image_pipeline_augs = [
            gen_image_pipeline_ffcv_ssl(
                device=device, transform_cls=transform_cls, rescale=rescale
            )
        ] * num_augmentations
        ordering = OrderOption.SEQUENTIAL
        pipelines = {"image": image_pipeline_og, "label": label_pipeline}
        custom_field_img_mapper = {}
        for i, aug_pipeline in enumerate(image_pipeline_augs):
            pipelines["image{}".format(i + 1)] = aug_pipeline
            custom_field_img_mapper["image{}".format(i + 1)] = "image"

        loaders[split] = Loader(
            datadir[split],
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            pipelines=pipelines,
            custom_field_mapper=custom_field_img_mapper,
        )

    return loaders

def gen_image_label_pipeline_ffcv_ssl(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    transform_cls: STLTransformFFCV = None,
    rescale: bool = False,
    device: str = "cuda:0",
    num_augmentations: int = 2,
):
    """
    Args:
        train_dataset : path to train dataset
        val_dataset   : path to test dataset
        batch_size    : batch-size
        num_workers   : number of workers
    Returns
        loaders       : dict('train': dataloader, 'test': dataloader)
    """

    datadir = {"train": train_dataset, "test": val_dataset}
    assert num_augmentations > 1, "Please use at least 2 augmentations for SSL."

    loaders = {}

    for split in ["train"]:
        if train_dataset is None: continue
        image_pipeline1 = gen_image_pipeline_ffcv_ssl(
            device=device, transform_cls=transform_cls, rescale=rescale
        )
        label_pipeline = gen_label_pipeline(device=device)
        image_pipeline_augs = [
            gen_image_pipeline_ffcv_ssl(
                device=device, transform_cls=transform_cls, rescale=rescale
            )
        ] * (
            num_augmentations - 1
        )  # creating other augmentations

        ordering = OrderOption.RANDOM  # if split == 'train' else OrderOption.SEQUENTIAL
        # ordering = OrderOption.SEQUENTIAL #if split == 'train' else OrderOption.SEQUENTIAL

        pipelines = {"image": image_pipeline1, "label": label_pipeline}
        custom_field_img_mapper = {}
        for i, aug_pipeline in enumerate(image_pipeline_augs):
            pipelines["image{}".format(i + 1)] = aug_pipeline
            custom_field_img_mapper["image{}".format(i + 1)] = "image"

        loaders[split] = Loader(
            datadir[split],
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            pipelines=pipelines,
            custom_field_mapper=custom_field_img_mapper,
        )

    for split in ["test"]:
        if val_dataset is None: continue
        label_pipeline = gen_label_pipeline(device=device)
        image_pipeline = gen_image_pipeline(
            device=device, transform_cls=STLClassifierTransform, rescale=rescale
        )

        ordering = (
            OrderOption.SEQUENTIAL
        )  # if split == 'train' else OrderOption.SEQUENTIAL

        loaders[split] = Loader(
            datadir[split],
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

    return loaders

def stl_ffcv(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    device: str = "cuda:0",
    num_augmentations: int = 2,
    test_ffcv: bool = False,
):
    """Function to return dataloader for STL-10 SSL

    Args:
        train_dataset (str, optional): Train dataset filename. Defaults to None.
        val_dataset (str, optional): Test dataset filename. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to None.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        device (str, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of patches. Defaults to 2.
        test_ffcv (bool, optional): Flag to use pipeline for testing FFCV multi-augmentations

    Returns:
        loaders : dict('train': dataloader, 'test': dataloader)
    """

    transform_cls = STLTransformFFCV()
    if test_ffcv:
        gen_img_label_fn = gen_image_label_pipeline_ffcv_ssl_test
    else:
        gen_img_label_fn = gen_image_label_pipeline_ffcv_ssl
    return gen_img_label_fn(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_cls=transform_cls,
        rescale=False,
        device=device,
        num_augmentations=num_augmentations,
    )

def stl_classifier_ffcv(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    device: str = "cuda:0",
    num_augmentations: int = 1,
    label_noise: int = 0
):
    """Function to return dataloader for STL-10 classification

    Args:
        train_dataset (str, optional): Train dataset filename. Defaults to None.
        val_dataset (str, optional): Test dataset filename. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to None.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        device (_type_, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of patches. Defaults to 1.
        label_noise (int, optional): Ratio (percentage) of label noise. From 0 to 100.

    Returns:
        loaders : dict('train': dataloader, 'test': dataloader)
    """
    
    transform_cls = STLClassifierTransform
    transform_cls_extra_augs = STLTransformFFCV()
    return gen_image_label_pipeline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_cls=transform_cls,
        device=device,
        num_augmentations=num_augmentations,
        transform_cls_augs=transform_cls_extra_augs,
        label_noise=label_noise,
    )

def stl10_pt(
    datadir,
    batch_size=None,
    num_workers=None,
    device="cuda:0",
    splits=['unlabeled']):
    """
    Create pytorch compatible dataloaders for STL-10.
    """
    loaders = {}
    for split in splits:
        dataset = torchvision.datasets.STL10(
            root=datadir, split=split, download=False,
            transform=SSLPT_STL())
        loaders['train' if 'unlabeled' in split else split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loaders

def stl10_classifier_pt(
    datadir,
    batch_size=None,
    num_workers=None,
    device="cuda:0",
    splits=['train', 'test']):
    """
    Create pytorch compatible dataloaders for CIFAR-10.
    """
    loaders = {}
    for split in splits:
        dataset = torchvision.datasets.STL10(
            root=datadir, split=split, download=True,
            transform=STLClassifierTransform())
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loaders
