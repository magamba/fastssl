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


write_dataset = True
noise_level = 15

try:
	dataset_folder = os.environ.get("DATA_DIR")
	ffcv_folder = dataset_folder
except KeyError:
	print("Please run scripts/setup_env first")
	sys.exit(1)

dataset = 'cifar10'
#if dataset=='cifar10':
#	dataset_folder = '/network/datasets/cifar10.var/cifar10_torchvision/'
#	ffcv_folder = '/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/cifar10'
#elif dataset=='cifar100':
#	dataset_folder = '/network/datasets/cifar100.var/cifar100_torchvision/'
#	ffcv_folder = '/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/cifar100'
#elif dataset=='stl10':
#	dataset_folder = '/network/datasets/stl10.var/stl10_torchvision/'
#	ffcv_folder = '/network/projects/_groups/linclab_users/ffcv/ffcv_datasets/stl10'

if noise_level>0:
	ffcv_folder = os.path.join(ffcv_folder, "{}-Noise_{}".format(dataset, int(noise_level)))

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

## WRITE TO BETON FILES
if write_dataset:
	if dataset=='cifar10':
		trainset = torchvision.datasets.CIFAR10(
		    root=dataset_folder, train=True, download=False, transform=None)
		testset = torchvision.datasets.CIFAR10(
		    root=dataset_folder, train=False, download=False, transform=None)

	if dataset=='cifar100':
		trainset = torchvision.datasets.CIFAR100(
		    root=dataset_folder, train=True, download=False, transform=None)
		testset = torchvision.datasets.CIFAR100(
		    root=dataset_folder, train=False, download=False, transform=None)

	elif dataset=='stl10':
		trainset = torchvision.datasets.STL10(
		    root=dataset_folder, split='train', download=False, transform=None)
		testset = torchvision.datasets.STL10(
		    root=dataset_folder, split='test', download=False, transform=None)

	train_beton_fpath = os.path.join(ffcv_folder,'train.beton')
	test_beton_fpath = os.path.join(ffcv_folder,'test.beton')
	datasets = {'train': trainset, 'test':testset}

	for name,ds in datasets.items():
		#breakpoint()
		if name == 'train' and noise_level>0:
			new_targets = add_label_noise(targets=ds.targets,
				 noise_percentage=noise_level/100.0)
			ds.targets = new_targets

		path = train_beton_fpath if name=='train' else test_beton_fpath
		writer = DatasetWriter(path, {
			'image': RGBImageField(),
			'label': IntField()
			})
		writer.from_indexed_dataset(ds)

## VERIFY the WRITTEN DATASET
BATCH_SIZE = 5000
loaders = {}
for name in ['train','test']:
	label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), Squeeze()]	# ToDevice('cuda:0'),
	image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

	image_pipeline.extend([ToTensor(),
        # ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        # torchvision.transforms.ConvertImageDtype(torch.float32),
        Convert(torch.float32),
        ])

	loaders[name] = Loader(os.path.join(ffcv_folder,'{}.beton'.format(name)),
							batch_size=BATCH_SIZE, num_workers=1,
							order=OrderOption.SEQUENTIAL,
							# drop_last=(name=='train'),
							pipelines={'image': image_pipeline, 'label': label_pipeline})

transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

if dataset=='cifar10':
	trainset = torchvision.datasets.CIFAR10(
	    root=dataset_folder, train=True, download=False, transform=transform_test)
	testset = torchvision.datasets.CIFAR10(
	    root=dataset_folder, train=False, download=False, transform=transform_test)
elif dataset=='cifar100':
	trainset = torchvision.datasets.CIFAR100(
	    root=dataset_folder, train=True, download=False, transform=transform_test)
	testset = torchvision.datasets.CIFAR100(
	    root=dataset_folder, train=False, download=False, transform=transform_test)
elif dataset=='stl10':
	trainset = torchvision.datasets.STL10(
	    root=dataset_folder, split='train', download=False, transform=transform_test)
	testset = torchvision.datasets.STL10(
	    root=dataset_folder, split='test', download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(
	trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
testloader = torch.utils.data.DataLoader(
	testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

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
for img,_ in loaders['train']:
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
mean = 0.0
std = 0.0
nb_samples = 0.
for img,_ in trainloader:
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
