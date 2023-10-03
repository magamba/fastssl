import torch
from typing import Tuple
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def split_batch_gen(data_loader, batch_size, label_noise=0):
    """ Generator that splits batches from data_loader into smaller batches
        of @batch_size and yields them. Also returns a tqdm progress bar.
        
        Note: For batches (img, label, img1, img2, ...) the generator discards any augmentations and labels, 
              and only returns the first item (img).
    """
    dl_batch_size = data_loader.batch_size
    assert batch_size <= dl_batch_size, f"Error: Jacobian batch size ({batch_size}) larger than data loader's batch size ({dl_batch_size})."

    # generator discards labels and augmentations
    for batch_id, batch in enumerate(data_loader):
        if label_noise > 0:
            imgs, targets, ground_truths, sample_ids = batch[:4]
            for img, target, ground_truth, sample_id in zip(
                torch.split(imgs, batch_size), 
                torch.split(targets, batch_size), 
                torch.split(ground_truths, batch_size),
                torch.split(sample_ids, batch_size)):
            
                yield img, target, ground_truth, sample_id
            
        else:
            imgs, targets = batch[:2]
            for img, target in zip(
                torch.split(imgs, batch_size), 
                torch.split(targets, batch_size)):
                
                yield img, target
            

def get_jacobian_fn(net, layer, data_loader):
    """Wrapper to initialize Jacobian computation algorithm
    """
    activations = {}
    def hook_fn(m,i,o):
        activations["features"] = i[0]
    
    handle = None if layer is None else layer.register_forward_hook(hook_fn)

    device = next(net.parameters()).device
    batch = next(iter(data_loader))
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    batch = batch.to(device)
    output = net(batch)
    if layer is None:
        ndims = np.prod(output.shape[1:])
    else:
        ndims = np.prod(activations["features"].shape[1:])
    
    def tile_input(x):
        tile_shape = (ndims,) + (1,) * len(x.shape[1:])
        return x.repeat(tile_shape)
            
    def jacobian_fn(x):
        # discard augmentations
        inp = x[0] if isinstance(x, (list, tuple)) else x
        nsamples = inp.shape[0]
        inp = tile_input(inp)
        inp.requires_grad_(True)
        
        output = net(inp)
        features = output if layer is None else activations["features"]
        j = jacobian_features(inp, features, nsamples, ndims)
        inp.grad = None
        
        activations["features"] = None
        return j
    
    return jacobian_fn, handle


def input_jacobian(net, layer, data_loader, batch_size=128, use_cuda=False, label_noise=0):
    """ Compute average input Jacobian norm of features of @layer of @net using @data_loader.
    
        If label_noise is nonzero, also separately compute the average input 
        Jacobian norm on samples with corrupted and uncorrupted labels respectively.
        
        Note: Jacobian computation can be memory expensive, so this function needs its own batch
              size when iterating over @data_loader, in order for (potentially) large batches
              to be broken down into smaller chunks.
    """  
    jacobian_fn, handle = get_jacobian_fn(net, layer, data_loader)
    
    jacobian_norm, jacobian_norm_clean, jacobian_norm_corr = 0., 0., 0.
    num_samples, num_clean, num_corr = 0, 0, 0
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    
    num_batches = len(data_loader) * int(round(float(data_loader.batch_size) / float(batch_size)))
    progress_bar = tqdm(
        split_batch_gen(
            data_loader, batch_size, label_noise
        ),
        desc="Feature Input Jacobian",
        total = num_batches +1,
    )
    
    for i, batch in enumerate(progress_bar):
        
        x = batch[0].to(device) # read input img from batch
        num_samples += x.shape[0]
        
        jacobian = jacobian_fn(x)
        _, operator_norm, _ = spectral_norm(
            jacobian,
            num_steps=20
        )
        
        jacobian_norm += operator_norm.sum().float().item()
        
        if label_noise > 0:
            target = batch[1].to(device, non_blocking=True)
            ground_truth = batch[2].to(device, non_blocking=True)
            
            clean_idx = torch.eq(target, ground_truth)
            corr_idx = torch.ne(target, ground_truth)
            num_clean += clean_idx.sum().float().item()
            num_corr += corr_idx.sum().float().item()
            
            jacobian_norm_clean += operator_norm[clean_idx].sum().float().item()
            jacobian_norm_corr += operator_norm[corr_idx].sum().float().item()

        avg_norm = jacobian_norm / num_samples
        if num_corr > 0:
            avg_norm_corr = jacobian_norm_corr / num_corr
        else:
            avg_norm_corr = 0.
        if num_clean > 0:
            avg_norm_clean = jacobian_norm_clean / num_clean
        else:
            avg_norm_clean = 0.
            
        progress_bar.set_description(
            "Batch: [{}/{}] avg Jacobian norm: {:.2f} avg Jacobian norm clean: {:.2f} avg Jacobian norm corr: {:.2f}".format(
                i, num_batches, avg_norm, avg_norm_clean, avg_norm_corr
            )
        )
    if handle is not None:
        handle.remove()
    return avg_norm, avg_norm_clean, avg_norm_corr


@torch.jit.script
def batched_matrix_vector_prod(u: torch.Tensor, J: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """ Compute product u @ J.t() @ v
    """
    return torch.bmm(
        torch.transpose(
            torch.bmm(J, v), 
            1,
            2
        ), u
    ).squeeze(-1).squeeze(-1) # workaround to avoid squeezing batch dimension


@torch.jit.script
def spectral_norm_power_iteration(J: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Compute one iteration of the power method to estimate
        the largest singular value of J
    """
    u = torch.bmm(J, v)
    u /= torch.norm(u, p=2, dim=1).unsqueeze(-1)
    v = torch.matmul(torch.transpose(J, 1, 2), u)
    v /= torch.norm(v, p=2, dim=1).unsqueeze(-1)
    return (u, v)


@torch.jit.script
def spectral_norm(J: torch.Tensor, num_steps: int, atol: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Compute the spectral norm of @J using @num_steps iterations
        of the power method.
        
        @return u (torch.Tensor): left-singular vector
        @return sigma (torch.Tensor): largest singular value
        @return v (torch.Tensor): right-singular vector
    """
    device = J.device
    dtype = J.dtype
    J = J.view(J.shape[0], -1, J.shape[2])
    nbatches, nindims, noutdims = J.shape[0], J.shape[1], J.shape[2]
    
    batch_indices = torch.arange(nbatches, dtype=torch.long, device=device)
    atol = torch.full((1,), fill_value=atol, device=device, dtype=dtype)
    
    v = torch.randn(nbatches, noutdims, 1, device=device, dtype=dtype)
    v /= torch.norm(v, p=2, dim=1).unsqueeze(-1)
    sigma_prev = torch.zeros(nbatches, dtype=dtype, device=device)
    u_prev = torch.zeros((nbatches, nindims), dtype=dtype, device=device)
    v_prev = torch.zeros((nbatches, noutdims), dtype=dtype, device=device)
    
    for i in range(num_steps):
        u, v = spectral_norm_power_iteration(J, v)
        sigma = batched_matrix_vector_prod(u, J, v)
        diff_indices = torch.ge(
            torch.abs(sigma.squeeze() - sigma_prev[batch_indices]), atol
        )

        if not torch.any(diff_indices):
            break
        
        sigma_prev[batch_indices[diff_indices]] = sigma[diff_indices]
        u_prev[batch_indices[diff_indices]] = u[diff_indices].squeeze(-1)
        v_prev[batch_indices[diff_indices]] = v[diff_indices].squeeze(-1)
        u = u[diff_indices]
        v = v[diff_indices]
        J = J[diff_indices]
        batch_indices = batch_indices[diff_indices]
        
    return u_prev.squeeze(), sigma_prev, v_prev.squeeze()


@torch.jit.script
def jacobian_features(x: torch.Tensor, features: torch.Tensor, nsamples: int, ndims: int) -> torch.Tensor:
    """ Compute the Jacobian of @logits w.r.t. @input.
        
        Note: @x_in should track gradient computation before @features
              is computed, otherwise this method will fail. @x should
              store a gradient_fn corresponding to the function used to
              produce @logits.
        
        Params:
            @x : 4D Tensor Batch of inputs with .grad attribute populated 
                 according to @logits
            @logits: 2D Tensor Batch of network features at @x
            
        Return:
            Jacobian: Batch-indexed 2D torch.Tensor of shape (N,*, K).
            where N is the batch dimension, D is the (flattened) input
            space dimension, and K is the number of feature dimensions
            of the network.
    """
    x.retain_grad()
    indexing_mask = torch.eye(ndims, device=x.device).repeat((nsamples,1))
    
    features.backward(gradient=indexing_mask, retain_graph=True)
    jacobian = x.grad.data.view(nsamples, ndims, -1).transpose(1,2)
    
    return jacobian


