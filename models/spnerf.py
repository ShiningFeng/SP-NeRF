import numpy as np
import torch


class Mapping(torch.nn.Module):
    def __init__(self, mapping_size, in_size, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = mapping_size
        self.in_channels = in_size
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = self.in_channels * (len(self.funcs) * self.N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (self.N_freqs - 1), self.N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class Siren(torch.nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def inference(model, args, rays_xyz, z_vals, rays_d=None, sun_d=None, rays_t=None, semantics=None, z_vals_unsort=None):
    """
    Runs the nerf model using a batch of input rays
    Args:
        model: NeRF model (coarse or fine)
        args: all input arguments
        rays_xyz: (N_rays, N_samples_, 3) sampled positions in the object space
                  N_samples_ is the number of sampled points in each ray;
                            = N_samples for coarse model
                            = N_samples+N_importance for fine model
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        rays_d: (N_rays, 3) direction vectors of the rays
        sun_d: (N_rays, 3) sun direction vectors
        rays_t (torch.Tensor, optional): Additional tensor related to rays, if applicable. Defaults to None.
        rays_s (torch.Tensor, optional): Additional tensor related to rays, if applicable. Defaults to None.
        z_vals_unsort: (N_rays, N_samples_) depths of the sampled positions before sorting, which is only used for visualizing the guided samples for the SpS-NeRF article
    Returns:
        result: dictionary with the output magnitudes of interest
    """

    N_rays = rays_xyz.shape[0]
    N_samples = rays_xyz.shape[1]
    xyz_ = rays_xyz.view(-1, 3)  # (N_rays*N_samples, 3)

    # handle additional inputs if provided
    rays_d_ = None if rays_d is None else torch.repeat_interleave(rays_d, repeats=N_samples, dim=0)
    sun_d_ = None if sun_d is None else torch.repeat_interleave(sun_d, repeats=N_samples, dim=0)
    rays_t_ = None if rays_t is None else torch.repeat_interleave(rays_t, repeats=N_samples, dim=0)
    semantics_ = None if semantics is None else torch.repeat_interleave(semantics, repeats=N_samples, dim=0)

    # split input batch into chunks to avoid possible problems with memory usage
    chunk = args.chunk
    batch_size = xyz_.shape[0]
    out_chunks = []

    for i in range(0, batch_size, chunk):
        out_chunk = model(
            xyz_[i:i + chunk],
            input_dir=None if rays_d_ is None else rays_d_[i:i + chunk],
            input_sun_dir=None if sun_d_ is None else sun_d_[i:i + chunk],
            input_t=None if rays_t_ is None else rays_t_[i:i + chunk],
            input_s=None if semantics_ is None else semantics_[i:i + chunk]
        )
        out_chunks.append(out_chunk)
    out = torch.cat(out_chunks, 0)

    out = out.view(N_rays, N_samples, model.number_of_outputs)
    rgbs = out[..., :3]  # (N_rays, N_samples, 3)
    sigmas = out[..., 3]  # (N_rays, N_samples)
    sun_v = out[..., 4:5]  # (N_rays, N_samples, 1)
    sky_rgb = out[..., 5:8]  # (N_rays, N_samples, 3)

    # define deltas, i.e. the length between the points in which the ray is discretized
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples-1)
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples)

    # compute alpha as in the formula (3) of the nerf paper
    noise_std = args.noise_std
    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples)

    # compute transparency and weights
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
    transparency = torch.cumprod(alphas_shifted, -1)[:, :-1]  # T in the paper
    weights = alphas * transparency  # (N_rays, N_samples) equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    # compute final outputs
    depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)
    irradiance = sun_v + (1 - sun_v) * sky_rgb  # equation 2 of the s-nerf paper
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs * irradiance, -2)  # (N_rays, 3)
    rgb_final = torch.clamp(rgb_final, min=0., max=1.)

    result = {'rgb': rgb_final,
              'depth': depth_final,
              'weights': weights,
              'transparency': transparency,
              'albedo': rgbs,
              'sun': sun_v,
              'sky': sky_rgb,
              'z_vals': z_vals}

    if z_vals_unsort is not None:
        result['z_vals_unsort'] = z_vals_unsort

    idx = 8
    if model.beta:
        uncertainty = out[..., idx:idx + 1]  # (N_rays, N_samples, 1)
        idx += 1
        result['beta'] = uncertainty

    if model.sem:
        sem_logits = out[..., idx:]  # (N_rays, N_samples, num_classes)
        sem_logits = torch.mean(sem_logits, dim=1)  # (N_rays, num_classes)
        result['sem_logits'] = sem_logits

    return result


class SPNeRF(torch.nn.Module):
    def __init__(self, num_sem_classes=3, s_embedding_factor=1, layers=8, feat=256, mapping=False,
                 mapping_sizes=[10, 4], skips=[4], siren=True, t_embedding_dims=16, beta=False, sem=False):
        super(SPNeRF, self).__init__()
        self.layers = layers
        self.skips = skips
        self.t_embedding_dims = t_embedding_dims
        self.mapping = mapping
        self.input_sizes = [3, 0]  # [input_xyz size, input_dir size]
        self.rgb_padding = 0.001
        self.beta = beta
        self.sem = sem
        self.num_sem_classes = num_sem_classes
        self.s_embedding_factor = s_embedding_factor
        self.semantic_size = self.num_sem_classes * self.s_embedding_factor  # embedding dimension for semantics

        # activation function
        nl = Siren() if siren else torch.nn.ReLU()

        # use positional encoding if specified
        in_size = self.input_sizes.copy()
        if mapping:
            self.mapping = [Mapping(map_sz, in_sz) for map_sz, in_sz in zip(mapping_sizes, self.input_sizes)]
            in_size[0] = 2 * mapping_sizes[0] * in_size[0]  # xyz dimension after positional encoding
        else:
            self.mapping = [torch.nn.Identity(), torch.nn.Identity()]

        # embedding layer for semantic labels
        if self.sem:
            self.semantic_embedding = torch.nn.Embedding(
                self.num_sem_classes + 1,  # 增加一个类别用于无效标签
                self.semantic_size,
                padding_idx=self.num_sem_classes)  # 设置无效标签的索引
        else:
            self.semantic_size = 0

        # adjust the first layer's input size to include semantic embedding
        self.input_size = in_size[0] + self.semantic_size

        # define the main network of fully connected layers (FC_NET)
        fc_layers = [torch.nn.Linear(self.input_size, feat), Siren(w0=30.0) if siren else nl]
        for i in range(1, layers):
            if i in skips:
                fc_layers.append(torch.nn.Linear(feat + self.input_size, feat))
            else:
                fc_layers.append(torch.nn.Linear(feat, feat))
            fc_layers.append(nl)
        self.fc_net = torch.nn.Sequential(*fc_layers)  # shared 8-layer structure

        # FC_NET output 1: volume density
        self.sigma_from_xyz = torch.nn.Sequential(torch.nn.Linear(feat, 1), torch.nn.Softplus())

        # FC_NET output 2: vector of features from the spatial coordinates
        self.feats_from_xyz = torch.nn.Linear(feat, feat)

        # FC_NET output 3 (optional): semantic logits
        if self.sem:
            self.logit_from_label = torch.nn.Sequential(
                torch.nn.Linear(feat, feat // 2),
                nl,
                torch.nn.Linear(feat // 2, num_sem_classes)
            )

        # branch output1: albedo color (the FC_NET output 2 is concatenated to the encoded viewing direction input)
        self.rgb_from_xyzdir = torch.nn.Sequential(
            torch.nn.Linear(feat + in_size[1], feat // 2),
            nl,
            torch.nn.Linear(feat // 2, 3),
            torch.nn.Sigmoid()
        )

        # branch output2: shading scalar
        sun_dir_in_size = 3
        sun_v_layers = [torch.nn.Linear(feat + sun_dir_in_size, feat // 2), Siren() if siren else nl]
        for i in range(1, 3):
            sun_v_layers.append(torch.nn.Linear(feat // 2, feat // 2))
            sun_v_layers.append(nl)
        sun_v_layers.append(torch.nn.Linear(feat // 2, 1))
        sun_v_layers.append(torch.nn.Sigmoid())
        self.sun_v_net = torch.nn.Sequential(*sun_v_layers)

        # branch output3: ambient color
        self.sky_color = torch.nn.Sequential(
            torch.nn.Linear(sun_dir_in_size, feat // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(feat // 2, 3),
            torch.nn.Sigmoid(),
        )

        if siren:
            self.fc_net.apply(sine_init)
            self.fc_net[0].apply(first_layer_sine_init)
            self.sun_v_net.apply(sine_init)
            self.sun_v_net[0].apply(first_layer_sine_init)

        # branch output4: uncertainty coefficient
        if self.beta:
            self.beta_from_xyz = torch.nn.Sequential(
                torch.nn.Linear(self.t_embedding_dims + feat, feat // 2),
                nl,
                torch.nn.Linear(feat // 2, 1),
                torch.nn.Softplus()
            )

        # Calculate the total number of outputs
        self.number_of_outputs = 8  # RGB (3) + sigma (1) + sun visibility (1) + sky RGB (3)
        if self.beta:
            self.number_of_outputs += 1  # Uncertainty beta (1)
        if self.sem:
            self.number_of_outputs += self.num_sem_classes  # Semantic logits

    def forward(self, input_xyz, input_dir=None, input_sun_dir=None, input_t=None, input_s=None, sigma_only=False):
        """
            Predicts the RGB values, volume density (sigma), and optionally semantic logits from a batch of input rays.
            The input rays are represented as a set of 3D spatial coordinates (xyz) with optional additional inputs.

            Args:
                input_xyz: (B, 3) Spatial coordinates
                input_dir: (B, 3) Direction vectors of the rays (optional)
                input_sun_dir: (B, 3) Sun direction vectors (optional)
                input_t: (B, T) Additional input related to rays (optional)
                input_s: (B, S) Semantic input per point (optional)
                sigma_only: If True, only sigma is computed

            Returns:
                If semantic information is included:
                    - If sigma_only is True:
                        sigma: (B, 1) volume density
                    - Otherwise:
                        output: (B, 4 + num_semantic_classes):
                            - The first 3 columns: RGB color
                            - The 4th column: volume density (sigma)
                            - The remaining columns: semantic logits, i.e., (B, num_semantic_classes)

                If semantic information is not included:
                    - If sigma_only is True:
                        sigma: (B, 1) volume density
                    - Otherwise:
                        output: (B, 4):
                            - The first 3 columns: RGB color
                            - The 4th column: volume density (sigma)
            """
        # Step 1: Map input_xyz
        input_xyz_mapped = self.mapping[0](input_xyz)

        # Step 2: Embed input_s if provided and concatenate with input_xyz_mapped
        if self.sem and input_s is not None:
            input_s = input_s.squeeze().long()
            invalid_label = self.num_sem_classes  # 将无效标签（-100）映射到 num_sem_classes（无效标签的索引）
            input_s = torch.where(
                input_s == -100,
                torch.tensor(invalid_label, device=input_s.device),
                input_s
            )
            # input_s = input_s.to(self.semantic_embedding.weight.device)
            input_s_embedded = self.semantic_embedding(input_s)
            input_combined = torch.cat((input_xyz_mapped, input_s_embedded),
                                       dim=1)  # (B, mapped_xyz_size + semantic_size)
        else:
            input_combined = input_xyz_mapped  # (B, mapped_xyz_size)

        # Step 3: Feature extraction through MLP layers
        xyz_ = input_combined
        for i in range(self.layers):
            if i in self.skips:
                xyz_ = torch.cat([xyz_, input_combined], -1)
            xyz_ = self.fc_net[2 * i](xyz_)
            xyz_ = self.fc_net[2 * i + 1](xyz_)
        shared_features = xyz_

        # FC_NET output 1: volume density
        sigma = self.sigma_from_xyz(shared_features)
        if sigma_only:
            return sigma

        # FC_NET output 2: vector of features from the spatial coordinates
        xyz_features = self.feats_from_xyz(shared_features)

        # branch output1: albedo color
        if self.input_sizes[1] > 0:
            input_xyzdir = torch.cat([xyz_features, self.mapping[1](input_dir)], -1)
        else:
            input_xyzdir = xyz_features

        rgb = self.rgb_from_xyzdir(input_xyzdir)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        out = torch.cat([rgb, sigma], 1)  # (B, 4)

        # branch output2: shading scalar
        input_sun_v_net = torch.cat([xyz_features, input_sun_dir], -1)
        sun_v = self.sun_v_net(input_sun_v_net)

        # branch output3: ambient color
        sky_color = self.sky_color(input_sun_dir)
        out = torch.cat([out, sun_v, sky_color], 1)  # (B, 8)

        # branch output4: uncertainty coefficient
        if self.beta:
            input_for_beta = torch.cat([xyz_features, input_t], -1)
            beta = self.beta_from_xyz(input_for_beta)
            out = torch.cat([out, beta], 1)  # (B, 8+1)

        # FC_NET output 3(optional): semantic logits
        if self.sem:
            sem_logits = self.logit_from_label(shared_features)
            out = torch.cat([out, sem_logits], 1)  # (B, 8 + num_semantic_classes)

        return out
