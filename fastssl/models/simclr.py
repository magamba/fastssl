import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastssl.models.ssl import SSL
from fastssl.models.backbone import BackBone

def SimCLRLoss(model, inp, _temperature=0.05):
    """ Perform model forward pass and compute the SimCLR loss,
        a.k.a. the normalized temperature-scaled cross entropy.
        
        Args:
            model: a torch.nn.Module
            inp: a torch.Tensor (or list thereof)
        Returns:
            loss: scalar tensor
    """
    # generate samples from tuple
    inp = list(inp)
    _ = inp.pop(1)
    num_augs = len(inp)
    for x in inp:
        x = x.cuda(non_blocking=True)

    bsz = inp[0].shape[0]

    # forward pass
    z_list = [model(x) for x in inp]
    z_norm_list = [(z - z.mean(0)) / z.std(0) for z in z_list]

    loss = 0.0
    # compute sum across all patches of each image for each embedding dim
    z_norm_sum = torch.sum(torch.stack(z_norm_list), dim=0)
    for i in range(num_augs):
        # take embedding of one patch
        z1_norm = z_norm_list[i]
        # take mean embedding of all other patches
        z2_norm = (z_norm_sum - z_norm_list[i]) / (num_augs - 1)

        z1_norm_large = z1_norm
        z2_norm_large = z2_norm
        
        labels = F.one_hot(torch.arange(bsz), bsz*2).float().cuda(non_blocking=True)
        masks = F.one_hot(torch.arange(bsz), bsz).float().cuda(non_blocking=True)
        
        logits_aa = torch.matmul(z1_norm, z1_norm_large.T) / _temperature
        logits_aa = logits_aa - masks

        logits_bb = torch.matmul(z2_norm, z2_norm_large.T) / _temperature
        logits_bb = logits_bb - masks
        logits_ab = torch.matmul(z1_norm, z2_norm_large.T) / _temperature
        logits_ba = torch.matmul(z2_norm, z1_norm_large.T) / _temperature
        
        softmax_a = F.softmax(
            torch.cat((logits_ab, logits_aa), 1), dim=1
        )
        softmax_b = F.softmax(
            torch.cat((logits_ba, logits_bb), 1), dim=1
        )
        loss_a = F.cross_entropy(
            softmax_a, labels
        )
        loss_b = F.cross_entropy(
            softmax_b, labels
        )
        loss += loss_a + loss_b

    loss = loss / num_augs
    return loss


#def SimCLRLoss(model, inp, _temperature=0.05):
#    """
#    Peform model forward pass and compute the BYOL loss.
#    Args:
#        model: a torch.nn.Module
#        inp: a torch.Tensor
#    Returns:
#        loss: scalar tensor
#    """

#    # generate samples from tuple
#    inp = list(inp)
#    _ = inp.pop(1)
#    num_augs = len(inp)
#    for x in inp:
#        x = x.cuda(non_blocking=True)
#    # (x1, x2), _ = inp
#    # x1, x2 = x1.cuda(non_blocking=True), x2.cuda(non_blocking=True)

#    bsz = inp[0].shape[0]
#    # bsz = x1.shape[0]

#    # forward pass
#    z_list = [model(x) for x in inp]
#    # z1 = model(x1)   #NXD
#    # z2 = model(x2)   #NXD

#    z_nan = torch.as_tensor([torch.any(torch.isnan(z)) for z in z_list], dtype=torch.bool)
#    z_inf = torch.as_tensor([torch.any(torch.isinf(z)) for z in z_list], dtype=torch.bool)
#    print(f"z_list NaNs: {torch.any(z_nan)} z_list inf: {torch.any(z_inf)}")

#    z_norm_list = [(z - z.mean(0)) / z.std(0) for z in z_list]
#    # z1_norm = F.normalize(z1, dim=-1)
#    # z2_norm = F.normalize(z2, dim=-1)

#    loss = 0.0
#    # compute sum across all patches of each image for each embedding dim
#    z_norm_sum = torch.sum(torch.stack(z_norm_list), dim=0)
#    for i in range(num_augs):
#        # take embedding of one patch
#        z1_norm = z_norm_list[i]
#        # take mean embedding of all other patches
#        z2_norm = (z_norm_sum - z_norm_list[i]) / (num_augs - 1)
#        all_z_norm = torch.cat([z1_norm, z2_norm], dim=0)

#        similarity_scores = (all_z_norm @ all_z_norm.T) / _temperature
#        eps = 1e-9

#        ones = torch.ones(bsz)
#        mask = (torch.diag(ones, bsz) + torch.diag(ones, -bsz)).cuda(non_blocking=True)

#        # subtract max value for stability
#        logits = (
#            similarity_scores - similarity_scores.max(dim=-1, keepdim=True)[0].detach()
#        )
#        # remove the diagonal entries of all 1./_temperature because they are cosine
#        # similarity of one image to itself.
#        exp_logits = torch.exp(logits) * (1 - torch.eye(2 * bsz)).cuda(
#            non_blocking=True
#        )

#        log_likelihood = -logits + torch.log(exp_logits.sum(dim=-1, keepdim=True) + eps)
#        print(f"Logits NaNs: {torch.any(torch.isnan(logits))} Exp logits NaNs: {torch.any(torch.isnan(exp_logits))} Log Logits NaNs: {torch.any(torch.isnan(torch.log(exp_logits.sum(dim=-1, keepdim=True) + eps)))} Log-likelihood NaNs: {torch.any(torch.isnan(log_likelihood))}")
#        print(f"Logits infs: {torch.any(torch.isinf(logits))} Exp logits infs: {torch.any(torch.isinf(exp_logits))} Log Logits infs: {torch.any(torch.isinf(torch.log(exp_logits.sum(dim=-1, keepdim=True) + eps)))} Log-likelihood infs: {torch.any(torch.isinf(log_likelihood))}")

#        loss += (log_likelihood * mask).sum() / mask.sum()
#    loss = loss / num_augs
#    return loss


class SimCLR(SSL):
    """
    Modified ResNet50 architecture to work with CIFAR10
    """

    def __init__(
        self, bkey="resnet50proj", projector_dim=128, dataset="cifar10", hidden_dim=512
    ):
        super(SimCLR, self).__init__()
        self.dataset = dataset
        self.bkey = bkey

        self.backbone = BackBone(
            name=self.bkey,
            dataset=self.dataset,
            projector_dim=projector_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x):
        projections = self._unnormalized_project(x)
        return F.normalize(projections, dim=-1)

    def _unnormalized_project(self, x):
        feats = self._unnormalized_feats(x)
        projections = self.backbone.proj(feats)
        return projections

    def _unnormalized_feats(self, x):
        x = self.backbone(x)
        feats = torch.flatten(x, start_dim=1)
        return feats
