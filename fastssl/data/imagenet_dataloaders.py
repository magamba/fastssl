# Ffcv dataloaders for imagenet to run barlow twinsr
import torch
import torchvision
from typing import List
from ffcv.fields.decoders import (
    IntDecoder,
    SimpleRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    RandomHorizontalFlip, 
    Squeeze, 
    ToDevice, 
    ToTensor, 
    Convert,
    NormalizeImage,
    ToTorchImage, 
    RandomResizedCrop
)
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from fastssl.data.custom_transforms import TransformImagenet, GaussianBlur, Solarization
import torchvision.transforms as transforms
from PIL import Image

from fastssl.data.cifar_transforms import (
    ImagenetClassifierTransform,
    ImagenetTransformFFCV
)

import numpy as np
from fastssl.utils.label_correction import with_indices, corrupt_labels

IMG_SIZE = 224
DEFAULT_CROP_RATIO = 224/256


def to_device(device):
    if device == "cuda:0":
        return ToDevice(device, non_blocking=True)
    else:
        return ToDevice("cpu")


def gen_image_pipeline(device="cuda:0", transform_cls=None):
    image_pipeline: List[Operation] = [
        CenterCropRGBImageDecoder((IMG_SIZE,IMG_SIZE), ratio=DEFAULT_CROP_RATIO),
        ToTensor(),
        to_device(device),
        ToTorchImage(),
        # Convert(torch.float32),
        # NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]

    if transform_cls:
        image_pipeline.extend(transform_cls().transform_list)

    return image_pipeline


def gen_image_pipeline_ffcv_ssl(device="cuda:0", transform_cls=None):
    if transform_cls:
        image_pipeline: List[Operation] = [
            # RandomResizedCrop((224, 224))
            RandomResizedCrop(
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
            # Convert(torch.float32),
            # NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
        ]
    )

    return image_pipeline


def gen_label_pipeline(device="cuda:0", transform_cls=None):
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        ToDevice("cuda:0"),
        Squeeze(),
    ]
    return label_pipeline


def gen_image_label_pipeline(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    transform_cls: ImagenetClassifierTransform = None,
    device: str = "cuda:0",
    num_augmentations: int = 1,
    transform_cls_augs: ImagenetTransformFFCV = None,
):
    """Generate image and label pipelines for supervised classification.

    Args:
        train_dataset (str, optional): path to train dataset. Defaults to None.
        val_dataset (str, optional): path to test dataset. Defaults to None.
        batch_size (int, optional): batch-size. Defaults to None.
        num_workers (int, optional): number of CPU workers. Defaults to None.
        transform_cls (ImagenetClassifierTransform, optional): Transforms to be applied for the original image. Defaults to None.
        device (str, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of total image augmentations. Defaults to 1.
        transform_cls_augs (ImagenetTransformFFCV, optional): Transforms to be applied to generate other augmentations. Defaults to None.

    Returns:
        loaders : dict('train': dataloader, 'test': dataloader)
    """

    datadir = {"train": train_dataset, "test": val_dataset}
    assert num_augmentations > 0, "Please use at least 1 augmentation for classifier."

    loaders = {}

    for split in ["train", "test"]:
        if datadir[split] is None: continue
        label_pipeline = gen_label_pipeline(device=device)
        image_pipeline = gen_image_pipeline(
            device=device, transform_cls=transform_cls
        )
        if num_augmentations > 1:
            image_pipeline_augs = [
                gen_image_pipeline_ffcv_ssl(
                    device=device, transform_cls=transform_cls_augs
                )
            ] * (num_augmentations - 1)
        else:
            image_pipeline_augs = []
        ordering = OrderOption.RANDOM if split == "train" else OrderOption.SEQUENTIAL
        # ordering = OrderOption.RANDOM #if split == 'train' else OrderOption.SEQUENTIAL

        pipelines = {"image": image_pipeline, "label": label_pipeline}
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
            # pipelines={'image' : image_pipeline, 'label' : label_pipeline}
            pipelines=pipelines,
            custom_field_mapper=custom_field_img_mapper,
        )
    return loaders


def gen_image_label_pipeline_ffcv_ssl(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    transform_cls: ImagenetTransformFFCV = None,
    device: str = "cuda:0",
    num_augmentations: int = 2,
    extra_augmentations: bool = False,
):
    """Function for generating multiple augmentations from each image.

    Args:
        train_dataset (str, optional): Train dataset filename. Defaults to None.
        val_dataset (str, optional): Test dataset filename. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to None.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        transform_cls (CifarTransformFFCV, optional): Transform object. Defaults to None.
        device (str, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of patches. Defaults to 2.

    Returns:
        loaders: dict('train': dataloader, 'test': dataloader)
    """

    datadir = {"train": train_dataset, "test": val_dataset}
    assert num_augmentations > 1, "Please use at least 2 augmentations for SSL."

    loaders = {}

    for split in ["train"]:
        if train_dataset is None: continue
        image_pipeline1 = gen_image_pipeline_ffcv_ssl(
            device=device, transform_cls=transform_cls
        )
        label_pipeline = gen_label_pipeline(device=device)
        image_pipeline_augs = [
            gen_image_pipeline_ffcv_ssl(
                device=device, transform_cls=transform_cls
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
            device=device, transform_cls=ImagenetClassifierTransform
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

    if extra_augmentations and train_dataset is not None:
        split = "train_extra"
        num_samples = 5
        image_pipeline1 = gen_image_pipeline_ffcv_ssl(
            device=device, transform_cls=transform_cls
        )
        label_pipeline = gen_label_pipeline(device=device)
        image_pipeline_augs = [
            gen_image_pipeline_ffcv_ssl(
                device=device, transform_cls=transform_cls
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
            datadir["train"],
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=True,
            order=ordering,
            drop_last=False,
            pipelines=pipelines,
            custom_field_mapper=custom_field_img_mapper,
        )
    return loaders


def imagenet_ffcv(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    device: str = "cuda:0",
    num_augmentations: int = 2,
    extra_augmentations: bool = False,
):
    """Function to return dataloader for Imagenet-1k SSL

    Args:
        train_dataset (str, optional): Train dataset filename. Defaults to None.
        val_dataset (str, optional): Test dataset filename. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to None.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        device (str, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of patches. Defaults to 2.

    Returns:
        loaders : dict('train': dataloader, 'test': dataloader)
    """

    # transform_cls = CifarTransform
    transform_cls = ImagenetTransformFFCV()
    gen_img_label_fn = gen_image_label_pipeline_ffcv_ssl
    return gen_img_label_fn(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_cls=transform_cls,
        device=device,
        num_augmentations=num_augmentations,
        extra_augmentations=extra_augmentations,
    )


def imagenet_classifier_ffcv(
    train_dataset: str = None,
    val_dataset: str = None,
    batch_size: int = None,
    num_workers: int = None,
    device: str = "cuda:0",
    num_augmentations: int = 1,
):
    """Function to return dataloader for Imagenet-1k classification

    Args:
        train_dataset (str, optional): Train dataset filename. Defaults to None.
        val_dataset (str, optional): Test dataset filename. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to None.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        device (str, optional): CPU/GPU. Defaults to 'cuda:0'.
        num_augmentations (int, optional): Number of patches. Defaults to 1.

    Returns:
        loaders : dict('train': dataloader, 'test': dataloader)
    """

    transform_cls = ImagenetClassifierTransform
    transform_cls_extra_augs = ImagenetTransformFFCV()
    return gen_image_label_pipeline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_cls=transform_cls,
        device=device,
        num_augmentations=num_augmentations,
        transform_cls_augs=transform_cls_extra_augs,
    )
##############################################################################
################ LEGACY CODE : LEAVING FOR LATER #############################
##############################################################################
def get_sseval_imagenet_ffcv_dataloaders(
    train_dataset=None, val_dataset=None, batch_size=None, num_workers=None
):
    paths = {
        'train': train_dataset,
        'val': val_dataset
    }

    loaders = {}

    for name in ['train', 'val']:
        image_pipeline: List[Operation] = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
            RandomHorizontalFlip(),
            ToTensor(),
            #ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            torchvision.transforms.ConvertImageDtype(torch.float16),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            #ToDevice('cuda:0')
        ]
        ordering = OrderOption.QUASI_RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name],
                               batch_size=batch_size,
                               num_workers=num_workers,
                               order=ordering,
                               drop_last=(name == 'train'),
                               pipelines={
                                   'image': image_pipeline,
                                   'label': label_pipeline
                                    }
                               )

    return loaders

def get_ssltrain_imagenet_ffcv_dataloaders(
        train_dataset=None, batch_size=None, num_workers=None
):
    paths = {
        'train': train_dataset,
    }

    loaders = {}

    for name in ['train']:
        image_pipeline: List[Operation] = [
            CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
            ToTensor(),
            #ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            TransformImagenet()
        ]

        ordering = OrderOption.QUASI_RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name],
                               batch_size=batch_size,
                               num_workers=num_workers,
                               order=ordering,
                               drop_last=(name == 'train'),
                               pipelines={
                                   'image': image_pipeline,
                               }
                               )
    return loaders

def get_ssltrain_imagenet_pytorch_dataloaders(
        data_dir=None, batch_size=None, num_workers=None, extra_augmentations=False,
):
    paths = {
        'train': data_dir + '/train',
    }

    loaders = {}

    for name in ['train']:
        dataset = torchvision.datasets.ImageFolder(paths[name], Transform())
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=False, shuffle=True, drop_last=True
        )
        loaders[name] = loader

    if extra_augmentations:
        for name in ['train_extra']:
            dataset = torchvision.datasets.ImageFolder(paths['train'], MultiViewTransform())
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers,
                pin_memory=False, shuffle=True, drop_last=True
            )
            loaders[name] = loader

    return loaders


 
def get_ssleval_imagenet_pytorch_dataloaders(
        data_dir=None, batch_size=None, num_workers=None, label_noise=0, ood_eval=False,
):
    paths = {
        'train': data_dir + '/train',
    }
    
    if ood_eval:
        train_dir = "/".join(data_dir.split("/")[:-1]) # removing noise_type
        train_dir = "-".join(train_dir.split("-")[:-1]) # removing "-c" suffix from dataset
        paths['train'] = train_dir + '/train'
        assert label_noise == 0, "Error: choose either ood_eval=True or label_noise > 0"
        paths.update(
            { f'test-{l}': data_dir + f'/{l}' for l in range(1, 6) }
        )
    else:
        paths['test'] = data_dir + '/val'
    
    loaders = {}
    for name in paths.keys():
        dset_cls = torchvision.datasets.ImageFolder
        if name == 'train' and label_noise > 0:
            dset_cls = with_indices(dset_cls)

        dataset = dset_cls(paths[name], EvalTransform())
        if name == 'train' and label_noise > 0:
            assert not ood_eval, "Error: choose either ood_eval=True or label_noise > 0"
            try:
                targets = dataset.targets
            except AttributeError:
                targets = dataset.labels
            targets_orig = targets.copy()
            new_targets = corrupt_labels(
                targets=targets,
                label_noise=label_noise
            )
            dataset.samples = [ (s[0], t) for (s, t) in zip(dataset.samples, new_targets) ]
            try:
                _ = dataset.targets
                dataset.targets = new_targets
            except AttributeError:
                dataset.labels = new_targets
	            
            dataset._targets_orig = targets_orig
        
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=False, shuffle=False, drop_last=True
        )
        loaders[name] = loader

    return loaders


def get_ssltrain_imagenet_pytorch_dataloaders_distributed(
        data_dir=None, batch_size=None, num_workers=None, world_size=None
):
    paths = {
        'train': data_dir + '/train',
    }

    loaders = {}

    for name in ['train']:
        dataset = torchvision.datasets.ImageFolder(paths[name], Transform())
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        assert batch_size % world_size == 0
        per_device_batch_size = batch_size // world_size
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=per_device_batch_size, num_workers=num_workers,
            pin_memory=True, sampler=sampler
        )
        loaders[name] = loader

    return loaders, sampler


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            #transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(96, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            #transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(96, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

class MultiViewTransform(Transform):
    def __init__(self, n_augs: int = 5):
        super(MultiViewTransform, self).__init__()
        self.n_augs = n_augs
        
    def __call__(self, x):
        y1 = self.transform(x)
        output = (y1, ) + tuple(self.transform_prime(x) for _ in range(self.n_augs -1))
        return output 

class EvalTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((96, 96), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        return y1

class TransformGPU:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x1, x2):
        y1 = self.transform(x1)
        y2 = self.transform_prime(x2)
        return y1, y2

