"""
This script defines the evaluation metrics and the loss functions
"""

import torch

from kornia.losses import ssim as ssim_


def uncertainty_aware_loss(loss_dict, inputs, gt_rgb, typ, beta_min=0.05):
    beta = torch.sum(inputs[f'weights_{typ}'].unsqueeze(-1) * inputs['beta_coarse'], -2) + beta_min
    loss_dict[f'{typ}_color'] = ((inputs[f'rgb_{typ}'] - gt_rgb) ** 2 / (2 * beta ** 2)).mean()
    loss_dict[f'{typ}_logbeta'] = (3 + torch.log(beta).mean()) / 2  # +3 to make c_b positive since beta_min = 0.05
    return loss_dict


def solar_correction(loss_dict, inputs, typ, lambda_sc=0.05):
    # computes the solar correction terms defined in Shadow NeRF and adds them to the dictionary of losses
    sun_sc = inputs[f'sun_sc_{typ}'].squeeze()
    term2 = torch.sum(torch.square(inputs[f'transparency_sc_{typ}'].detach() - sun_sc), -1)
    term3 = 1 - torch.sum(inputs[f'weights_sc_{typ}'].detach() * sun_sc, -1)
    loss_dict[f'{typ}_sc_term2'] = lambda_sc / 3. * torch.mean(term2)
    loss_dict[f'{typ}_sc_term3'] = lambda_sc / 3. * torch.mean(term3)
    return loss_dict


class SNerfLoss(torch.nn.Module):
    def __init__(self, lambda_sc=0.05):
        super().__init__()
        self.lambda_sc = lambda_sc
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_dict = {}
        typ = 'coarse'
        loss_dict[f'{typ}_color'] = self.loss(inputs[f'rgb_{typ}'], targets)
        if self.lambda_sc > 0:
            loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        if 'rgb_fine' in inputs:
            typ = 'fine'
            loss_dict[f'{typ}_color'] = self.loss(inputs[f'rgb_{typ}'], targets)
            if self.lambda_sc > 0:
                loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict


class SatNerfLoss(torch.nn.Module):
    def __init__(self, lambda_sc=0.0):
        super().__init__()
        self.lambda_sc = lambda_sc

    def forward(self, inputs, targets):
        loss_dict = {}
        typ = 'coarse'
        loss_dict = uncertainty_aware_loss(loss_dict, inputs, targets, typ)
        if self.lambda_sc > 0:
            loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        if 'rgb_fine' in inputs:
            typ = 'fine'
            loss_dict = uncertainty_aware_loss(loss_dict, inputs, targets, typ)
            if self.lambda_sc > 0:
                loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)
        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict


class DepthLoss(torch.nn.Module):
    def __init__(self, lambda_ds=1.0, GNLL=False, usealldepth=True, margin=0, stdscale=1):
        super().__init__()
        self.lambda_ds = lambda_ds / 3.
        self.GNLL = GNLL
        self.usealldepth = usealldepth
        self.margin = margin
        self.stdscale = stdscale
        self.loss = torch.nn.GaussianNLLLoss() if self.GNLL else torch.nn.MSELoss(reduce=False)

    def is_not_in_expected_distribution(self, pred_depth, pred_std, target_depth, target_std):
        depth_diff = (pred_depth - target_depth).abs()
        return torch.logical_or(depth_diff > target_std, pred_std > target_std)

    def ComputeSubsetDepthLoss(self, inputs, typ, target_depth, target_weight, target_valid_depth, target_std):
        if target_valid_depth is None:
            print(
                f'target_valid_depth is None! Use all the target_depth by default! target_depth.shape[0]: {target_depth.shape[0]}')
            target_valid_depth = torch.ones(target_depth.shape[0])

        # Extract the value based on valid_mask
        valid_mask = target_valid_depth > 0
        z_vals = inputs[f'z_vals_{typ}'][valid_mask]
        pred_depth = inputs[f'depth_{typ}'][valid_mask]
        pred_weight = inputs[f'weights_{typ}'][valid_mask]
        # z_vals = inputs[f'z_vals_{typ}'][np.where(target_valid_depth.cpu() > 0)]
        # pred_depth = inputs[f'depth_{typ}'][np.where(target_valid_depth.cpu() > 0)]
        # pred_weight = inputs[f'weights_{typ}'][np.where(target_valid_depth.cpu() > 0)]

        if pred_depth.shape[0] == 0:
            print(
                f'ZERO target_valid_depth in this depth loss computation! target_weight.device: {target_weight.device}')
            return torch.zeros((1,), device=target_weight.device, requires_grad=True)

        pred_std = (((z_vals - pred_depth.unsqueeze(-1)).pow(2) * pred_weight).sum(-1)).sqrt()

        # Filter the target data based on valid_mask
        target_weight = target_weight[valid_mask]
        target_depth = target_depth[valid_mask]
        target_std = target_std[valid_mask]
        # target_weight = target_weight[np.where(target_valid_depth.cpu() > 0)]
        # target_depth = target_depth[np.where(target_valid_depth.cpu() > 0)]
        # target_std = target_std[np.where(target_valid_depth.cpu() > 0)]

        # Determine which depths to apply the loss to
        apply_depth_loss = torch.ones(target_depth.shape[0])
        if not self.usealldepth:
            apply_depth_loss = self.is_not_in_expected_distribution(pred_depth, pred_std, target_depth, target_std)

        # Filter the predictions based on the apply_depth_loss mask
        pred_depth = pred_depth[apply_depth_loss]
        if pred_depth.shape[0] == 0:
            print('ZERO apply_depth_loss in this depth loss computation!')
            return torch.zeros((1,), device=target_weight.device, requires_grad=True)
        pred_std = pred_std[apply_depth_loss]
        target_depth = target_depth[apply_depth_loss]

        numerator = float(pred_depth.shape[0])  # Count of predicted depths used in loss computation
        denominator = float(target_valid_depth.shape[0])  # Total count of valid target depths
        scaling_factor = numerator / denominator

        if self.GNLL:
            return scaling_factor * self.loss(pred_depth, target_depth, pred_std)
        else:
            return scaling_factor * target_weight[apply_depth_loss] * self.loss(pred_depth, target_depth)

    def forward(self, inputs, targets, weights=1., target_valid_depth=None, target_std=None):
        loss_dict = {}
        typ = 'coarse'
        if not self.usealldepth:
            loss_dict[f'{typ}_ds'] = self.ComputeSubsetDepthLoss(inputs, typ, targets, weights, target_valid_depth,
                                                                 target_std)
        else:
            loss_dict[f'{typ}_ds'] = self.loss(inputs['depth_coarse'], targets)

        if 'depth_fine' in inputs:
            typ = 'fine'
            if not self.usealldepth:
                loss_dict[f'{typ}_ds'] = self.ComputeSubsetDepthLoss(inputs, typ, targets, weights, target_valid_depth,
                                                                     target_std)
            else:
                loss_dict[f'{typ}_ds'] = self.loss(inputs['depth_fine'], targets)

        if not self.usealldepth:
            for k in loss_dict.keys():
                loss_dict[k] = self.lambda_ds * torch.mean(loss_dict[k])
        else:
            for k in loss_dict.keys():
                loss_dict[k] = self.lambda_ds * torch.mean(weights * loss_dict[k])

        loss = sum(l for l in loss_dict.values())
        return loss, loss_dict


class SemanticLoss(torch.nn.Module):
    def __init__(self, lambda_ss=1.0):
        super(SemanticLoss, self).__init__()
        self.lambda_ss = lambda_ss
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, inputs, targets):
        loss_dict = {}
        typ = 'coarse'
        sem_logits_coarse = inputs[f'sem_logits_{typ}']  # (N_rays * N_samples, num_classes)
        loss_dict[f'{typ}_ss'] = self.cross_entropy_loss(sem_logits_coarse, targets)  # N_rays x num_classes

        if f'sem_logits_fine' in inputs:
            typ = 'fine'
            sem_logits_fine = inputs[f'sem_logits_{typ}']
            loss_dict[f'{typ}_ss'] = self.cross_entropy_loss(sem_logits_fine, targets)

        for k in loss_dict.keys():
            loss_dict[k] = self.lambda_ss * loss_dict[k]

        loss = sum(loss_dict.values())
        return loss, loss_dict


def load_loss(args):
    if args.model == "sp-nerf":
        if args.beta:
            loss_function = SatNerfLoss(lambda_sc=args.sc_lambda)
        else:
            loss_function = SNerfLoss(lambda_sc=args.sc_lambda)
    else:
        raise ValueError(f'model {args.model} is not valid')
    return loss_function


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def ssim(image_pred, image_gt):
    """
    image_pred and image_gt: (1, 3, H, W)
    important: kornia==0.5.3
    """
    return torch.mean(ssim_(image_pred, image_gt, 3))


def miou(pred_semantics, gt_semantics, num_classes):
    """
    compute Mean Intersection over Union (mIoU).
    """
    iou_per_class = []

    for cls in range(num_classes):
        pred_cls = (pred_semantics == cls)
        gt_cls = (gt_semantics == cls)
        intersection = (pred_cls & gt_cls).float().sum()
        union = (pred_cls | gt_cls).float().sum()

        if union == 0:
            iou_per_class.append(torch.tensor(0.0))
        else:
            iou_per_class.append(intersection / union)

    miou_value = torch.mean(torch.stack(iou_per_class))
    return miou_value


def overall_accuracy(pred_semantics, gt_semantics):
    """
    compute Overall Accuracy (OA).
    """
    correct = (pred_semantics == gt_semantics).float().sum()
    total = torch.numel(gt_semantics)
    oa_value = correct / total
    return oa_value
