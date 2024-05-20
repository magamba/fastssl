from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
import numpy as np

def eval_step_clean_restored(model, dataloader, epoch=None, epochs=None, split=""):
    model.eval()
    total_1, total_5 = 0, 0
    total_clean_1, total_corr_1, total_restored_1 = 0., 0., 0.
    total_clean_5, total_corr_5, total_restored_5 = 0., 0., 0.
    total_samples, total_clean, total_corr = 0, 0, 0
    test_bar = tqdm(dataloader, desc=f"Label correction: {split}")
    for inp in test_bar:
        # for data, target in test_bar:
        inp = list(inp)
        # WARNING: every epoch could have different augmentations of images
        target = inp.pop(1)
        ground_truth = inp.pop(1)
        sample_idx = inp.pop(1)
        for x in inp:
            x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        ground_truth = ground_truth.cuda(non_blocking=True)
        total_samples += inp[0].shape[0]
        
        clean_idx = torch.eq(target, ground_truth)
        corr_idx = torch.ne(target, ground_truth)

        total_clean += clean_idx.sum().float().item()
        total_corr += corr_idx.sum().float().item()
        
        # total_samples += data.shape[0]
        # data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with autocast():
            logits = model(inp)
            
            preds = torch.argsort(logits, dim=1, descending=True)
            total_1 += torch.sum(
                (preds[:, 0:1] == target[:, None]).any(dim=-1).float()
            ).item()
            total_5 += torch.sum(
                (preds[:, 0:5] == target[:, None]).any(dim=-1).float()
            ).item()
            total_clean_1 += torch.sum(
                (preds[clean_idx, 0:1] == target[clean_idx, None]).any(dim=-1).float()
            ).item()
            total_clean_5 += torch.sum(
                (preds[clean_idx, 0:5] == target[clean_idx, None]).any(dim=-1).float()
            ).item()
            total_corr_1 += torch.sum(
                (preds[corr_idx, 0:1] == target[corr_idx, None]).any(dim=-1).float()
            ).item()
            total_corr_5 += torch.sum(
                (preds[corr_idx, 0:5] == target[corr_idx, None]).any(dim=-1).float()
            ).item()
            total_restored_1 += torch.sum(
                (preds[corr_idx, 0:1] == ground_truth[corr_idx, None]).any(dim=-1).float()
            ).item()
            total_restored_5 += torch.sum(
                (preds[corr_idx, 0:5] == ground_truth[corr_idx, None]).any(dim=-1).float()
            ).item()

        acc_1 = total_1 / total_samples * 100
        acc_5 = total_5 / total_samples * 100
        
        if total_clean > 0:
            acc_clean_1 = total_clean_1 / total_clean * 100
            acc_clean_5 = total_clean_5 / total_clean * 100
        else:
            acc_clean_1 = 0.
            acc_clean_5 = 0.
        
        if total_corr > 0:
            acc_corr_1 = total_corr_1 / total_corr * 100
            acc_corr_5 = total_corr_5 / total_corr * 100
            acc_restored_1 = total_restored_1 / total_corr * 100
            acc_restored_5 = total_restored_5 / total_corr * 100
        else:
            acc_corr_1 = 0.
            acc_corr_5 = 0.
            acc_restored_1 = 0.
            acc_restored_5 = 0.

        test_bar.set_description(
            "{} Epoch: [{}/{}] ACC@1: {:.2f}% clean ACC@1: {:.2f}% corr ACC@1: {:.2f}% restored ACC@1: {:.2f}%".format(
                split, epoch, epochs, acc_1, acc_clean_1, acc_corr_1, acc_restored_1
            )
        )
    return acc_1, acc_5, acc_clean_1, acc_clean_5, acc_corr_1, acc_corr_5, acc_restored_1, acc_restored_5


def with_indices(datasetclass):
    """ Wraps a DataSet class, so that it returns (data, target, index, ground_truth).
    """
    def __getitem__(self, index):
        data, target = datasetclass.__getitem__(self, index)
        try:
            ground_truth = self._targets_orig[index]
        except AttributeError:
            print("Warning: no ground truth found")
            ground_truth = target
        
        return data, target, ground_truth, index
        
    return type(datasetclass.__name__, (datasetclass,), {
        '__getitem__': __getitem__,
    })
    

def corrupt_labels(targets, label_noise, seed=1234):
    """ Corrupt @label_noise percent of the targets
        using labels sampled uniformly at random from other classes.
    """
    from numpy.random import default_rng
    rng = default_rng(seed)

    if label_noise <= 0:
        return targets

    if label_noise > 1:
        label_noise = float(label_noise) / 100.

    if not isinstance(targets,torch.Tensor):
        targets_tensor = torch.LongTensor(targets)
    else:
        targets_tensor = torch.clone(targets)

    targets_shape = targets_tensor.shape
    corrupt_idx = int(np.ceil(targets_tensor.shape[0] * label_noise))
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

