import tqdm
import torch
from torch.cuda.amp import autocast

def eval_step_clean_restored(model, dataloader, epoch=None, epochs=None, split=""):
    model.eval()
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
        corr_idx = ~clean_idx
        
        total_clean += clean_idx.sum().item()
        total_corr = corr_idx.sum().item()
        
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
        acc_clean_1 = total_clean_1 / total_clean * 100
        acc_clean_5 = total_clean_5 / total_clean * 100
        acc_corr_1 = total_corr_1 / total_corr * 100
        acc_corr_5 = total_corr_5 / total_corr * 100
        acc_restored_1 = total_restored_1 / total_corr * 100
        acc_restored_5 = total_restored_5 / total_corr * 100
        test_bar.set_description(
            "{} Epoch: [{}/{}] ACC@1: {:.2f}% ACC@5: {:.2f}% clean ACC@1: {:.2f}% clean ACC@5: {:.2f}%  corr ACC@1: {:.2f}% corr ACC@5: {:.2f}% restored ACC@1: {:.2f}% restored ACC@5: {:.2f}%".format(
                split, epoch, epochs, acc_1, acc_5, acc_clean_1, acc_clean_5, acc_corr_1, acc_corr_5, acc_restored_1, acc_restored_5
            )
        )
    return acc_clean_1, acc_clean_5, acc_corr_1, acc_corr_5, acc_restored_1, acc_restored_5
