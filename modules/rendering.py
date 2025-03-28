"""
This script renders the input rays that are used to feed the NeRF model
It discretizes each ray in the input batch into a set of 3d points at different depths of the scene
Then the nerf model takes these 3d points (and the ray direction, optionally, as in the original nerf)
and predicts a volume density at each location (sigma) and the color with which it appears
"""

import math

import numpy as np
import torch


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Args:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Returns:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape  # get number of rays and samples
    weights = weights + eps  # avoid zero weights
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # compute probability density (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # cumulative distribution function (N_rays, N_samples)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # padded to 0~1 inclusive (N_rays, N_samples_+1)

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)  # deterministic sampling
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)  # random sampling
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)  # find indices in CDF
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    # gather values for interpolation
    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1
    """
    denom equals 0 means a bin has weight 0, in which case it will not be sampled
    anyway, therefore any value for it is fine (set to 1 here)
    """

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])  # interpolation
    return samples


def sample_3sigma(low_3sigma, high_3sigma, N, det, near, far, device=None):
    # Generate N linearly spaced values between 0 and 1
    t_vals = torch.linspace(0., 1., steps=N, device=device)
    # Calculate the step size for interpolation
    step_size = (high_3sigma - low_3sigma) / (N - 1)
    # Interpolate between low_3sigma and high_3sigma, clamping to [near, far]
    bin_edges = (low_3sigma.unsqueeze(-1) * (1. - t_vals) + high_3sigma.unsqueeze(-1) * t_vals).clamp(near, far)
    # Compute the scaling factor for each bin
    factor = (bin_edges[..., 1:] - bin_edges[..., :-1]) / step_size.unsqueeze(-1)
    # Generate N-1 linearly spaced values between -3 and 3
    x_in_3sigma = torch.linspace(-3., 3., steps=(N - 1), device=device)
    # Compute bin weights using Gaussian distribution
    bin_weights = factor * (1. / math.sqrt(2 * np.pi) * torch.exp(-0.5 * x_in_3sigma.pow(2))).unsqueeze(0).expand(
        *bin_edges.shape[:-1], N - 1)
    # Sample using the computed bin edges and weights
    return sample_pdf(bin_edges, bin_weights, N, det=det)


def compute_samples_around_depth(res, N_samples, z_vals, perturb, near, far, device=None):
    pred_depth = res['depth']  # Extract predicted depth values from the results dictionary
    pred_weight = res['weights']  # Extract weights corresponding to the depth values

    # Calculate the sampling standard deviation around the predicted depth.
    sampling_std = (((z_vals - pred_depth.unsqueeze(-1)).pow(2) * pred_weight).sum(-1)).sqrt()

    depth_min = pred_depth - 3. * sampling_std  # Define the minimum depth for the 3-sigma range
    depth_max = pred_depth + 3. * sampling_std  # Define the maximum depth for the 3-sigma range

    # Sample new depth values within the 3-sigma range
    z_vals_2 = sample_3sigma(depth_min, depth_max, N_samples, perturb == 0., near, far, device=device)

    return z_vals_2  # Return the sampled depth values


def GenerateGuidedSamples(res, z_vals, N_samples, perturb, near, far, mode='test', valid_depth=None, target_depths=None,
                          target_std=None, device=None, margin=0, stdscale=1):
    z_vals_2 = torch.empty_like(z_vals)  # Initialize z_vals_2 with the same shape as z_vals
    z_vals_2 = compute_samples_around_depth(res, N_samples, z_vals, perturb, near[0, 0], far[0, 0],
                                            device=device)  # Sample around predicted depth

    if mode == 'train':
        assert valid_depth is not None, 'valid_depth missing in training batch!'
        target_depth = torch.flatten(
            target_depths[:, 0][np.where(valid_depth.cpu() > 0)])  # Get valid target depths and flatten.
        target_weight = target_depths[:, 1][np.where(valid_depth.cpu() > 0)]  # Get valid target weights
        target_std = torch.flatten(
            target_std[np.where(valid_depth.cpu() > 0)])  # Get valid target standard deviations and flatten

        # Calculate min and max depth for sampling
        depth_min = target_depth - 3. * target_std
        depth_max = target_depth + 3. * target_std

        z_vals_2_bkp = z_vals_2.clone()  # Backup current z_vals_2

        # Sample within 3 sigma of ground truth depth
        gt_samples = sample_3sigma(depth_min, depth_max, N_samples, perturb == 0., near[0, 0], far[0, 0], device=device)
        z_vals_2[np.where(valid_depth.cpu() > 0)] = gt_samples  # Replace z_vals_2 with ground truth samples where valid

    return z_vals_2  # Return new depth samples


def render_rays(models, args, rays, ts, semantics=None, mode='test', valid_depth=None, target_depths=None,
                target_std=None):
    N_samples = args.n_samples
    N_importance = args.n_importance
    variant = args.model
    use_disp = False
    perturb = 1.0

    # ray origins, directions, near/far bounds
    rays_o, rays_d, near, far = rays[:, 0:3], rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]

    # sample depths for coarse model
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp:
        z_vals = near * (1 - z_steps) + far * z_steps  # linear sampling in depth space
    else:
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # linear sampling in disparity space

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    # discretize rays for coarse model
    xyz_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    # run coarse model
    typ = "coarse"
    if variant == "sp-nerf":
        from models.spnerf import inference
        sun_d = rays[:, 8:11]
        rays_t = None
        if args.beta:
            rays_t = models['t'](ts) if ts is not None else None
        result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t,
                           semantics=semantics)
        if args.guidedsample:
            # Guided sampling within the 3σ range of the target depth for sps-nerf
            z_vals_2 = GenerateGuidedSamples(result, z_vals, N_samples, perturb, near, far, mode=mode,
                                             valid_depth=valid_depth, target_depths=target_depths,
                                             target_std=target_std, device=rays.device, margin=args.margin,
                                             stdscale=args.stdscale).detach()
            z_vals_2, _ = torch.sort(z_vals_2, -1)  # Sort the additional depth samples
            z_vals_unsort = torch.cat([z_vals, z_vals_2], -1)  # Combine original and guided depth samples
            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_2], -1), -1)  # Sort combined depth samples
            xyz_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # Update 3D points
            result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t,
                               semantics=semantics, z_vals_unsort=z_vals_unsort)
        if args.sc_lambda > 0:  # Solar correction
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
            result_tmp = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t,
                                   semantics=semantics)
            result['weights_sc'] = result_tmp["weights"]
            result['transparency_sc'] = result_tmp["transparency"]
            result['sun_sc'] = result_tmp["sun"]
    else:
        raise ValueError(f'model {args.model} is not valid')

    result_ = {}
    for k in result.keys():
        result_[f"{k}_{typ}"] = result[k]  # store coarse model results with key suffix

    # run fine model
    if N_importance > 0:
        # sample depths for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, result_['weights_coarse'][:, 1:-1], N_importance, det=(perturb == 0)).detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)  # combine and sort coarse and fine samples

        # discretize rays for fine model
        xyz_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(
            2)  # (N_rays, N_samples+N_importance, 3)

        typ = "fine"
        if variant == "sp-nerf":
            sun_d = rays[:, 8:11]
            rays_t = None
            if args.beta:
                rays_t = models['t'](ts) if ts else None
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t,
                               semantics=semantics)
            if args.sc_lambda > 0:
                # solar correction
                xyz_fine = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)
                result_ = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t,
                                    semantics=semantics)
                result['weights_sc'] = result_["weights"]
                result['transparency_sc'] = result_["transparency"]
                result['sun_sc'] = result_["sun"]
        else:
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=rays_d, semantics=semantics)

        for k in result.keys():
            result_[f"{k}_{typ}"] = result[k]

    return result_
