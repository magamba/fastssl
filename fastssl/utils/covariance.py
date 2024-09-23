import numpy as np
from tqdm import tqdm
import torch
import scipy
from numpy.linalg import LinAlgError
def covariance_decomposition(net, layer, data_loader, use_cuda=False, max_samples=0):
    """ Decompose feature covariance into intra-manifold and inter-manifold terms
    
        Assumes that data_loader returns training samples together with data augmentations
    """
    activations_arr = generate_activations_prelayer_torch(net, layer, data_loader, use_cuda, max_samples)
    
    # activations_arr NxAxD
    nobjects = activations_arr.shape[0]
    naugs = activations_arr.shape[1]
    object_activations = activations_arr.mean(dim=1, keepdim=True)  # mean over augs
    mean_activations = object_activations.mean(dim=0, keepdim=True)
    # compute the inter-object manifold covariance
    sigma_obj = (object_activations - mean_activations).squeeze().T @ (
        object_activations - mean_activations
    ).squeeze() / nobjects

    # compute the intra-object manifold covariance and take mean across objects
    sigma_augs = torch.bmm(
        (activations_arr - object_activations).permute((0, 2, 1)),
        (activations_arr - object_activations),
    ).mean(dim=0) / naugs
      
    sigma_augs_eigen = torch.linalg.svdvals(sigma_augs).cpu().numpy()
    sigma_obj_eigen = torch.linalg.svdvals(sigma_obj).cpu().numpy()
    
    try:
        try:
            discriminants_obj = scipy.linalg.eigvalsh(a=sigma_obj, b=sigma_augs)
        except LinAlgError:
            print("Sigma_intra inversion failed")
            eps = np.linalg.norm(sigma_augs) * torch.finfo(sigma_augs.dtype).eps
            sigma_augs_reg = sigma_augs + eps * np.eye(sigma_augs.shape[0])
            discriminants_obj = scipy.linalg.eigvalsh(a=sigma_obj, b=sigma_augs_reg)
            #discriminants_obj = np.zeros_like(sigma_augs_eigen)
    except LinAlgError:
        print("Regularized Sigma_intra inversion failed!")
        discriminants_obj = np.zeros_like(sigma_augs_eigen)

    try:
        try:
            discriminants_augs = scipy.linalg.eigvalsh(a=sigma_augs, b=sigma_obj)
        except LinAlgError:
            print("Sigma_inter inversion failed")
            eps = np.linalg.norm(sigma_obj) * torch.finfo(sigma_obj.dtype).eps
            sigma_obj_reg = sigma_obj + eps * np.eye(sigma_obj.shape[0])
            discriminants_augs = scipy.linalg.eigvalsh(a=sigma_augs, b=sigma_obj_reg)
            #discriminants_augs = np.zeros_like(discriminants_obj)
    except LinAlgError:
        print("Regularized Sigma_inter inversion failed!")
        discriminants_augs = np.zeros_like(sigma_obj_eigen)

#    # get full svd decomposition of sigma_obj
#    Uobj, sigma_obj_eigen, Vobj = torch.linalg.svd(sigma_obj, full_matrices=True)
#    projection = torch.matmul(sigma_augs.unsqueeze(0), Vobj.unsqueeze(-1)).squeeze()
#    projection = torch.linalg.vecdot(projection, Vobj)
#
#    # discard null space of sigma_obj and take square to compute projection norm
#    projecton = torch.sqrt(projection[:len(sigma_obj_eigen)])

    return sigma_augs_eigen, sigma_obj_eigen, discriminants_augs, discriminants_obj


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


