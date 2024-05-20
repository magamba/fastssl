import numpy as np
from tqdm import tqdm
import torch

def covariance_decomposition(net, layer, data_loader, use_cuda=False, max_samples=0):
    """ Decompose feature covariance into intra-manifold and inter-manifold terms
    
        Assumes that data_loader returns training samples together with data augmentations
    """
    activations_arr = generate_activations_prelayer_torch(net, layer, data_loader, use_cuda, max_samples)
    
    # activations_arr NxAxD
    object_activations = activations_arr.mean(dim=1, keepdim=True)  # mean over augs
    mean_activations = object_activations.mean(dim=0, keepdim=True)
    # compute the inter-object manifold covariance
    sigma_obj = (object_activations - mean_activations).squeeze().T @ (
        object_activations - mean_activations
    ).squeeze()

    # compute the intra-object manifold covariance and take mean across objects
    sigma_augs = torch.bmm(
        (activations_arr - object_activations).permute((0, 2, 1)),
        (activations_arr - object_activations),
    ).mean(dim=0)
    
    sigma_augs_eigen = torch.linalg.svdvals(sigma_augs).cpu().numpy()
    sigma_obj_eigen = torch.linalg.svdvals(sigma_obj).cpu().numpy()
    
    return sigma_augs_eigen, sigma_obj_eigen


def generate_activations_prelayer_torch(net,layer,data_loader,use_cuda=False,max_samples=0):
    batch_ = next(iter(data_loader))
    num_augs = len(batch_) -1
    ndims = net.backbone.proj[0].weight.shape[1]
    
    activations = []
    def hook_fn(m,i,o):
        activations.append(i[0].reshape(-1, num_augs, ndims).cpu())
    handle = layer.register_forward_hook(hook_fn)

    if use_cuda:
        net = net.cuda()
    net.eval()
    
    num_samples = 0
    for i, inp in enumerate(tqdm(data_loader, desc="Covariance decomposition")):
        inp = list(inp)
        _ = inp.pop(1) # discarding labels
        if isinstance(inp, (tuple, list)) and isinstance(inp[0], (tuple, list)):
            inp = inp[0]
        num_samples += inp[0].shape[0]
        images = torch.vstack(inp) # (batch_size x <feat_dim> , batch_size x <feat_dim>, ...) -> (num_augs * batch_size x <feat_dim>)
        
        if use_cuda:
            images = images.cuda()
        with torch.no_grad():
            output = net(images)
        if max_samples > 0 and num_samples >= max_samples:
            break
    handle.remove()
    activations_torch = torch.vstack(activations) # batches x num_augs x <feat_dims> --> num_examples x num_augs x <feat_dims>
    del activations
    return activations_torch


