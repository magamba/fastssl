import os
import sys
import numpy as np
from tqdm import tqdm
import torchvision
import torch
from ffcv.writer import DatasetWriter
from typing import List
from ffcv.fields import IntField, RGBImageField
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from fastssl.utils.ood import CIFAR10C

write_dataset = True
noise_level = 0
subsample_classes = False # if enabled, generates a reduced version of the dataset, with only a few classes sampled
unseen_classes = False # if true, sample classes from a secondary list

noise_type = "zoom_blur"
ood_noise_types = [
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

if unseen_classes:
    assert subsample_classes, "Error: subsample_classes must be True when unseen_classes is True"

dataset = 'cifar10c'
#if dataset=='cifar10':
#	dataset_folder = '/network/datasets/cifar10.var/cifar10_torchvision/'
#	ffcv_folder = '/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/cifar10'
#elif dataset=='cifar100':
#	dataset_folder = '/network/datasets/cifar100.var/cifar100_torchvision/'
#	ffcv_folder = '/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/cifar100'
#elif dataset=='stl10':
#	dataset_folder = '/network/datasets/stl10.var/stl10_torchvision/'
#	ffcv_folder = '/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/stl10'

if unseen_classes:
    classes_to_keep = ['ship', 'horse']
else:
    classes_to_keep = ['automobile', 'dog'] #, 'ship']

samples_per_class = 0 # 1500

dataset_folder = os.environ.get("DATA_DIR")
if dataset_folder is None:
    raise RuntimeError("Please run scripts/setup_env first")
ffcv_folder = dataset_folder

folder_name = str(dataset) if noise_type == "" else "cifar10-c"
if unseen_classes:
    folder_name += "-unseen_2"
elif subsample_classes:
    folder_name += "-subsampled_2"

if noise_level>0:
    folder_name += "-Noise_{}".format(int(noise_level))

if subsample_classes or noise_level > 0 or noise_type != "":
    ffcv_folder = os.path.join(ffcv_folder, folder_name)

if dataset == "cifar10c":
    assert noise_level == 0, "Error only test set available for CIFAR10-C"
    ffcv_folder = os.path.join(ffcv_folder, noise_type)

def with_indices(datasetclass):
    """ Wraps a DataSet class, so that it returns (data, target, index, ground_truth).
    """
    def __getitem__(self, index):
        data, target = datasetclass.__getitem__(self, index)
        try:
            ground_truth = self._targets_orig[index]
        except AttributeError:
            ground_truth = target
        
        return data, target, ground_truth, index
        
    return type(datasetclass.__name__, (datasetclass,), {
        '__getitem__': __getitem__,
    })

def subsample_dataset(dataset, classes_to_keep, samples_per_class, train=False):
    """ Subsample classes from dataset and return a modified dataset (in-place)
    """
    class_idx = { c: dataset.class_to_idx[c] for c in classes_to_keep }
    targets = np.asarray(dataset.targets)

    # select samples to keep according to classes_to_keep and samples_per_class
    mask_per_class = [ targets == class_idx[c] for c in class_idx ]
    samples_mask = np.zeros_like(mask_per_class[0])
    for i, mask in enumerate(mask_per_class):
        if train and samples_per_class > 0:
            cut_idx = np.where(mask)[0][samples_per_class]
            mask[cut_idx:] = False
        mask_per_class[i] = mask
        samples_mask = np.logical_or(samples_mask, mask)

    sample_idx = np.where(samples_mask)[0]

    targets = targets[sample_idx]
    dataset.data = dataset.data[sample_idx]
    dataset.classes = classes_to_keep

    for cid, c in enumerate(class_idx):
        targets[targets == class_idx[c]] = cid
        class_idx[c] = cid

    dataset.targets = targets
    dataset.class_to_idx = class_idx

    return dataset

def add_label_noise(targets,noise_percentage=0.1,seed=1234):
    from numpy.random import default_rng
    rng = default_rng(seed)

    if not isinstance(targets,torch.Tensor):
        targets_tensor = torch.LongTensor(targets)
    else:
        targets_tensor = torch.clone(targets)

    targets_shape = targets_tensor.shape
    corrupt_idx = int(np.ceil(targets_tensor.shape[0] * noise_percentage))
    num_classes = targets_tensor.unique().shape[0]
    if corrupt_idx == 0:
        return targets

    noise = torch.zeros_like(targets_tensor)
    noise[0:corrupt_idx] = torch.from_numpy(
        rng.integers(low=1, high=num_classes, size=(corrupt_idx,))
    )
    shuffle = torch.from_numpy(rng.permutation(noise.shape[0]))
    noise = noise[shuffle]

    new_labels = (
        (targets_tensor + noise) % num_classes
        ).type(torch.LongTensor)

    num_targets_match = (new_labels==targets_tensor).sum()
    print("Amount of label noise added: {:.3f}".format(
        100.0*(1-num_targets_match/targets_shape[0])))

    if not isinstance(targets,torch.Tensor):
        # assuming targets was a list
        new_labels = list(new_labels)
    return new_labels

if dataset=='cifar10':
    CIFAR10 = torchvision.datasets.CIFAR10

    if noise_level > 0:
        CIFAR10 = with_indices(
            CIFAR10
        )

    trainset = CIFAR10(
        root=dataset_folder, train=True, download=False, transform=None
    )
        
    testset = torchvision.datasets.CIFAR10(
        root=dataset_folder, train=False, download=False, transform=None
    )

if dataset=='cifar100':
    CIFAR100 = torchvision.datasets.CIFAR100
    if noise_level > 0:
        CIFAR100 = with_indices(
            CIFAR100
        )

    trainset = CIFAR100(
        root=dataset_folder, train=True, download=False, transform=None
    )
    testset = torchvision.datasets.CIFAR100(
        root=dataset_folder, train=False, download=False, transform=None
    )

if dataset=='cifar10c':
    trainset = None
    testset = CIFAR10C(
        root=os.path.join(dataset_folder, folder_name), noise_type=noise_type, transform=None
    )

elif dataset=='stl10':
    STL10 = torchvision.datasets.STL10
    if noise_level > 0:
        STL10 = with_indices(
            STL10
        )

    unlabeledset = torchvision.datasets.STL10(
        root=dataset_folder, split="unlabeled", download=False, transform=None
    )
    trainset = STL10(
        root=dataset_folder, split='train', download=False, transform=None)
    testset = torchvision.datasets.STL10(
        root=dataset_folder, split='test', download=False, transform=None)

dataset_str = f"{dataset}_" if noise_level == 0 and not subsample_classes and dataset != "cifar10c" else ""
train_beton_fpath = os.path.join(ffcv_folder, dataset_str + "train.beton")
test_beton_fpath = os.path.join(ffcv_folder, dataset_str + "test.beton")

## WRITE TO BETON FILES
if write_dataset:
    datasets = {'train': trainset, 'test':testset}
    for name,ds in datasets.items():
        if dataset == "cifar10c" and name == "train": continue
        #breakpoint()
        if dataset == "cifar10" and subsample_classes:
            ds = subsample_dataset(ds, classes_to_keep, samples_per_class, train= name == "train")
        
        if name == 'train' and noise_level>0:
            try:
                targets = ds.targets
            except AttributeError:
                targets = ds.labels
            targets_orig = targets.copy()
            new_targets = add_label_noise(targets=targets,
                noise_percentage=noise_level/100.0)
            try:
                _ = ds.targets
                ds.targets = new_targets
            except AttributeError:
                ds.labels = new_targets
	            
            ds._targets_orig = targets_orig

        path = train_beton_fpath if name=='train' else test_beton_fpath
        fields = {
            'image': RGBImageField(),
            'label': IntField(),
        }
        if noise_level > 0 and name == "train":
            fields.update({
                'ground_truth': IntField(),
                'sample_idx': IntField(),
            })
        writer = DatasetWriter(path, fields)
        writer.from_indexed_dataset(ds)
    if dataset=='stl10':
        datasets = {"unlabeled": unlabeledset}
        unlabeled_beton_fpath = os.path.join(ffcv_folder, "unlabeled.beton")
        for name, ds in datasets.items():
            breakpoint()
            path = unlabeled_beton_fpath
            writer = DatasetWriter(path, {"image": RGBImageField(), "label": IntField()})
            writer.from_indexed_dataset(ds)

## VERIFY the WRITTEN DATASET
BATCH_SIZE = 500
loaders = {}
for name in ["train", "test"]:
    if name == "train" and dataset == "cifar10c": continue
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
    ]  # ToDevice('cuda:0'),
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    image_pipeline.extend(
        [
            ToTensor(),
            # ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            # torchvision.transforms.ConvertImageDtype(torch.float32),
            Convert(torch.float32),
        ]
    )

    pipelines = {'image': image_pipeline, 'label': label_pipeline}
    if noise_level > 0:
        pipelines.update({'ground_truth': label_pipeline, 'sample_idx': label_pipeline})

    loaders[name] = Loader(
        os.path.join(ffcv_folder, "{}.beton".format(name)),
        batch_size=BATCH_SIZE,
        num_workers=1,
        order=OrderOption.SEQUENTIAL,
        # drop_last=(name=='train'),
        drop_last=False,
        pipelines=pipelines,
    )

transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

skip_train = False
if dataset=='cifar10':
    dataset_cls = torchvision.datasets.CIFAR10
    if noise_level > 0:
        dataset_cls = with_indices(dataset_cls)
    trainset = dataset_cls(
        root=dataset_folder, train=True, download=False, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(
        root=dataset_folder, train=False, download=False, transform=transform_test)
    if subsample_classes:
        trainset = subsample_dataset(trainset, classes_to_keep, samples_per_class, train=True)
        testset = subsample_dataset(testset, classes_to_keep, samples_per_class)
elif dataset=='cifar100':
    dataset_cls = torchvision.datasets.CIFAR100
    if noise_level > 0:
        dataset_cls = with_indices(dataset_cls)
    trainset = dataset_cls(
        root=dataset_folder, train=True, download=False, transform=transform_test)
    testset = torchvision.datasets.CIFAR100(
        root=dataset_folder, train=False, download=False, transform=transform_test)
elif dataset=='cifar10c':
    skip_train = True
    testset = CIFAR10C(
        root=os.path.join(dataset_folder, folder_name), noise_type=noise_type, transform=transform_test)
elif dataset=='stl10':
    dataset_cls = torchvision.datasets.STL10
    if noise_level > 0:
        dataset_cls = with_indices(dataset_cls)
    trainset = dataset_cls(
        root=dataset_folder, split='train', download=False, transform=transform_test)
    testset = torchvision.datasets.STL10(
        root=dataset_folder, split='test', download=False, transform=transform_test)

if not skip_train:
    trainloader = torch.utils.data.DataLoader(
	    trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

if not skip_train:
    if noise_level > 0:
        X_ffcv, y_ffcv, gt_ffcv, sid_ffcv = next(iter(loaders['train']))
        X_tv, y_tv, gt_tv, sid_tv = next(iter(trainloader))
    else:
        X_ffcv, y_ffcv = next(iter(loaders['train']))
        X_tv, y_tv = next(iter(trainloader))
    print('FFCV stats:',X_ffcv.shape,X_ffcv.mean(),X_ffcv.min(),X_ffcv.max())
    print('torchV stats:',X_tv.shape,X_tv.mean(),X_tv.min(),X_tv.max())
    print(torch.allclose(X_ffcv,X_tv*255.))
    #breakpoint()

	# calculate mean and std of dataset
    print("ffcv dataset stats...")
    mean = 0.0
    std = 0.0
    nb_samples = 0.
    for inp in loaders['train']:
        img = inp[0]
        batch_samples = img.size(0)
        data = img.view(batch_samples,img.size(1),-1)
        mean+= data.mean(2).sum(0)
        std+= data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    print("Train Dataset mean",mean)
    print("Train Dataset std",std)
mean = 0.0
std = 0.0
nb_samples = 0.
for img,_ in loaders['test']:
    batch_samples = img.size(0)
    data = img.view(batch_samples,img.size(1),-1)
    mean+= data.mean(2).sum(0)
    std+= data.std(2).sum(0)
    nb_samples += batch_samples
mean /= nb_samples
std /= nb_samples
print("Test Dataset mean",mean)
print("Test Dataset std",std)

print("tv dataset stats...")
if not skip_train:
    mean = 0.0
    std = 0.0
    nb_samples = 0.
    for inp in trainloader:
        img = inp[0]
        batch_samples = img.size(0)
        data = img.view(batch_samples,img.size(1),-1)
        mean+= data.mean(2).sum(0)
        std+= data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    print("Train Dataset mean", mean)
    print("Train Dataset std", std)
    print("Train Dataset mean*255", mean*255)
    print("Train Dataset std*255", std*255)

mean = 0.0
std = 0.0
nb_samples = 0.
for img,_ in testloader:
    batch_samples = img.size(0)
    data = img.view(batch_samples,img.size(1),-1)
    mean+= data.mean(2).sum(0)
    std+= data.std(2).sum(0)
    nb_samples += batch_samples
mean /= nb_samples
std /= nb_samples
print("Test Dataset mean",mean)
print("Test Dataset std",std)
print("Test Dataset mean*255", mean*255)
print("Test Dataset std*255", std*255)

if noise_level > 0:
    num_corrupted_ffcv = 0.
    num_corrupted = 0.
    num_samples = 0
    for (_, t_ffcv, gt_ffcv, sid_ffcv), (_, t_tv, gt_tv, sid_tv) in zip(loaders["train"], trainloader):
        assert torch.all(torch.eq(sid_ffcv, sid_tv))
        assert torch.all(torch.eq(gt_ffcv, gt_tv))
        assert torch.all(torch.eq(gt_tv, t_tv))
        num_corrupted_ffcv += torch.sum(torch.ne(t_ffcv, gt_ffcv)).float().item()
        num_corrupted += torch.sum(torch.ne(t_ffcv, t_tv)).float().item()
        num_samples += t_ffcv.shape[0]
    num_corrupted_ffcv = int(num_corrupted_ffcv / num_samples *100)
    num_corrupted = int(num_corrupted / num_samples * 100)
    print(f"FFCV - ratio of corrupted samples: {num_corrupted_ffcv}%")
    print(f"FFCV vs TV - ratio of corrupted samples: {num_corrupted}%")
    assert (num_corrupted == num_corrupted_ffcv)
    assert (num_corrupted == int(noise_level))

### ===============================================================================
# Usage:
#   1. set dataset manually
#   2. Automate noisy label creation via the following bash script
#
# oldnoise=5; \
# for n in 5 10 15 20 40 60 80 100; do \
#     echo "Noise: $n"; \
#     sed_string="0,/noise_level = $oldnoise""/{s/noise_level = $oldnoise""/noise_level = $n/}"; \
#     echo "Sed string: $sed_string"; \
#     sed -i "$sed_string" write_ffcv_datasets.py; \
#     oldnoise=$n; \
#     python write_ffcv_datasets.py; \
# done; \
# sed_string="0,/noise_level = $oldnoise""//{s/noise_level = $oldnoise""/noise_level = 5/}"; \
# echo "Sed string: $sed_string"; \
# sed -i "$sed_string" write_ffcv_datasets.py; \
# unset oldnoise sed_string n;
#
### ===============================================================================
# STL10 stats
# ffcv dataset stats...
# Train Dataset mean tensor([113.9112, 112.1515, 103.6948])
# Train Dataset std tensor([57.1603, 56.4828, 57.0975])
# Test Dataset mean tensor([114.5820, 112.7223, 104.1996])
# Test Dataset std tensor([57.3148, 56.5328, 57.2032])
# tv dataset stats...
# Train Dataset mean tensor([0.4467, 0.4398, 0.4066])
# Train Dataset std tensor([0.2242, 0.2215, 0.2239])
# Test Dataset mean tensor([0.4472, 0.4396, 0.4050])
# Test Dataset std tensor([0.2249, 0.2217, 0.2237])
### ===============================================================================
