import argparse
import datetime
import os
import tempfile
import warnings

import lpips
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from kornia.losses import ssim as ssim_
from osgeo import gdal

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="osgeo.gdal")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Initialize LPIPS model
lpips_loss = lpips.LPIPS(net='alex')  # Use the same initialization as in the batch evaluation


def predefined_val_ts(img_id):
    return 0


def save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number, num_sem_classes):
    rays = sample["rays"].squeeze()
    rgbs = sample["rgbs"].squeeze()
    src_id = sample["src_id"][0]
    src_path = os.path.join(dataset.img_dir, src_id + ".tif")

    typ = "fine" if "rgb_fine" in results else "coarse"
    if "h" in sample and "w" in sample:
        W, H = sample["w"][0], sample["h"][0]
    else:
        W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))  # assume squared images

    img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    depth = results[f"depth_{typ}"]

    # save depth prediction
    _, _, alts = dataset.get_latlonalt_from_nerf_prediction(rays.cpu(), depth.cpu())
    out_path = f"{out_dir}/depth/{src_id}_epoch{epoch_number}.tif"
    utils.save_output_image(alts.reshape(1, H, W), out_path, src_path)

    # save dsm
    out_path = f"{out_dir}/dsm/{src_id}_epoch{epoch_number}.tif"
    dsm = dataset.get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)

    # save rgb image
    out_path = f"{out_dir}/rgb/{src_id}_epoch{epoch_number}.tif"
    utils.save_output_image(img, out_path, src_path)

    # save gt rgb image
    out_path = f"{out_dir}/gt_rgb/{src_id}_epoch{epoch_number}.tif"
    utils.save_output_image(img_gt, out_path, src_path)

    # Save semantic map as GeoTIFF with original class values
    if f"sem_logits_{typ}" in results:
        sem_logits = results[f"sem_logits_{typ}"]  # (N_rays, num_classes)
        sem_pred = sem_logits.argmax(dim=-1).view(H, W).cpu().numpy()  # (H, W)

        # Map internal indices back to original class IDs for the GeoTIFF
        remapped_sem = utils.remap_semantics_to_original(sem_pred, num_sem_classes)
        out_path_tif = f"{out_dir}/semantic/{src_id}_epoch{epoch_number}.tif"
        utils.save_output_image(remapped_sem[np.newaxis, ...], out_path_tif, src_path)

        # Create and save a colored PNG output_visual of the semantic map
        out_path_png = f"{out_dir}/semantic/{src_id}_epoch{epoch_number}.png"
        utils.save_sem_image(sem_pred, out_path_png, num_sem_classes)

    # save shadow modelling images
    if f"sun_{typ}" in results:
        s_v = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'sun_{typ}'], -2)
        out_path = f"{out_dir}/sun/{src_id}_epoch{epoch_number}.tif"
        utils.save_output_image(s_v.view(1, H, W).cpu(), out_path, src_path)

        rgb_albedo = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'albedo_{typ}'], -2)
        out_path = f"{out_dir}/albedo/{src_id}_epoch{epoch_number}.tif"
        utils.save_output_image(rgb_albedo.cpu().view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)

        if f"ambient_a_{typ}" in results:
            a_rgb = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'ambient_a_{typ}'], -2)
            out_path = f"{out_dir}/ambient_a/{src_id}_epoch{epoch_number}.tif"
            utils.save_output_image(a_rgb.view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)

            b_rgb = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'ambient_b_{typ}'], -2)
            out_path = f"{out_dir}/ambient_b/{src_id}_epoch{epoch_number}.tif"
            utils.save_output_image(b_rgb.view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)

        if f"beta_{typ}" in results:
            beta = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'beta_{typ}'], -2)
            out_path = f"{out_dir}/beta/{src_id}_epoch{epoch_number}.tif"
            utils.save_output_image(beta.view(1, H, W).cpu(), out_path, src_path)

        if f"sky_{typ}" in results:
            sky_rgb = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'sky_{typ}'], -2)
            out_path = f"{out_dir}/sky/{src_id}_epoch{epoch_number}.tif"
            utils.save_output_image(sky_rgb.cpu().view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)


def mse(image_pred, image_gt, reduction='mean'):
    """Calculate Mean Squared Error (MSE) between two images."""
    value = (image_pred - image_gt) ** 2
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, max_pixel_value=1.0):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse_val = mse(image_pred, image_gt, reduction='mean')
    if mse_val == 0:
        return float('inf')
    return 10 * torch.log10(max_pixel_value ** 2 / mse_val)


def ssim(image_pred, image_gt):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    image_pred and image_gt: tensors with shape (1, 3, H, W)
    """
    return torch.mean(ssim_(image_pred, image_gt, 3))


def compute_lpips(pred_rgb_tensor, gt_rgb_tensor):
    """Compute LPIPS perceptual loss."""
    # LPIPS expects input tensors normalized to [-1, 1]
    pred_rgb_tensor = (pred_rgb_tensor * 2.0 - 1.0)
    gt_rgb_tensor = (gt_rgb_tensor * 2.0 - 1.0)

    lpips_value = lpips_loss(pred_rgb_tensor, gt_rgb_tensor)  # LPIPS returns a tensor
    return lpips_value.item()  # Convert to scalar value


def compute_mae(pred_dsm_path, src_id, aoi_id, gt_dir, out_dir, epoch_number):
    """Compute Mean Absolute Error (MAE) between predicted DSM and ground truth DSM."""
    mae, _ = compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, aoi_id, gt_dir, out_dir, epoch_number)
    return mae


def compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, aoi_id, gt_dir, out_dir, epoch_number):
    """
    Compute pointwise differences between predicted DSM and ground truth DSM,
    return MAE and residual map path.
    """
    gt_dsm_path = os.path.join(gt_dir, f"{aoi_id}_DSM.tif")
    gt_roi_path = os.path.join(gt_dir, f"{aoi_id}_DSM.txt")

    assert os.path.exists(gt_dsm_path), f"{gt_dsm_path} not found"
    assert os.path.exists(pred_dsm_path), f"{pred_dsm_path} not found"

    try:
        gt_roi_metadata = np.loadtxt(gt_roi_path)
    except Exception as e:
        raise ValueError(f"Error loading ROI metadata from {gt_roi_path}: {e}")

    rdsm_diff_path = os.path.join(out_dir, f"{src_id}_rdsm_diff_epoch{epoch_number}.tif")
    rdsm_path = os.path.join(out_dir, f"{src_id}_rdsm_epoch{epoch_number}.tif")

    # Compute the difference map
    diff = dsm_pointwise_diff(
        in_dsm_path=pred_dsm_path,
        gt_dsm_path=gt_dsm_path,
        dsm_metadata=gt_roi_metadata,
        out_rdsm_path=rdsm_path,
        out_err_path=rdsm_diff_path
    )

    mae = np.mean(np.abs(diff))

    print(f"Difference map range: {diff.min()} to {diff.max()}")
    print(f"Computed MAE: {mae}")

    return mae, rdsm_diff_path


def dsm_pointwise_diff(in_dsm_path, gt_dsm_path, dsm_metadata, out_rdsm_path=None, out_err_path=None):
    """
    Computes pointwise differences between a generated DSM and a reference DSM.
    """
    if len(dsm_metadata) != 4:
        raise ValueError("dsm_metadata must be [xoff, yoff, xsize, resolution].")

    # Read DSM metadata
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]  # Offsets in x and y directions
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])  # Bounding box size in pixels
    resolution = dsm_metadata[3]  # Resolution in meters per pixel

    # Define the bounding box for cropping using GDAL translate
    ulx, uly = xoff, yoff + ysize * resolution  # Upper-left corner
    lrx, lry = xoff + xsize * resolution, yoff  # Lower-right corner

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create unique temporary file paths
        unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        pred_dsm_crop_path = os.path.join(temp_dir, f"crop_pred_dsm_{unique_identifier}.tif")
        pred_rdsm_temp_path = os.path.join(temp_dir, f"crop_rdsm_{unique_identifier}.tif")

        # Crop the predicted DSM using GDAL
        ds = gdal.Open(in_dsm_path)
        if ds is None:
            raise FileNotFoundError(f"Failed to open {in_dsm_path} with GDAL.")
        gdal.Translate(pred_dsm_crop_path, ds, projWin=[ulx, uly, lrx, lry])
        ds = None  # Close the dataset to release resources

        # Read ground truth DSM
        with rasterio.open(gt_dsm_path) as f:
            gt_dsm = f.read(1)

        # Read the cropped predicted DSM
        with rasterio.open(pred_dsm_crop_path) as f:
            pred_dsm_cropped = f.read(1)
            profile = f.profile

        # Ensure dimensions match after cropping
        if pred_dsm_cropped.shape != gt_dsm.shape:
            raise ValueError("Cropped predicted DSM and GT DSM have mismatched dimensions.")

        # Attempt to perform alignment using DSMR module; fallback to Z-axis alignment if unavailable
        try:
            from modules import dsmr
            transform = dsmr.compute_shift(gt_dsm_path, pred_dsm_crop_path, scaling=False)
            dsmr.apply_shift(pred_dsm_crop_path, pred_rdsm_temp_path, *transform)
            with rasterio.open(pred_rdsm_temp_path, "r") as f:
                pred_rdsm = f.read(1)
        except ImportError:
            print("Warning: DSMR module not found! Falling back to Z-axis mean adjustment.")
            # Simple Z-axis alignment: shift predicted DSM to match the mean of the ground truth
            pred_rdsm = pred_dsm_cropped + np.nanmean(gt_dsm - pred_dsm_cropped)

        # Replace NaN values with the minimum depth
        min_depth = min(np.nanmin(pred_rdsm), np.nanmin(gt_dsm))
        pred_rdsm = np.nan_to_num(pred_rdsm, nan=min_depth)
        gt_dsm = np.nan_to_num(gt_dsm, nan=min_depth)
        err = pred_rdsm - gt_dsm

        # Save registered DSM and error map if required
        if out_rdsm_path:
            with rasterio.open(out_rdsm_path, 'w', **profile) as dst:
                dst.write(pred_rdsm, 1)

        if out_err_path:
            with rasterio.open(out_err_path, 'w', **profile) as dst:
                dst.write(err, 1)

    return err


def plot_residual_map(residual_map_path, src_id, output_dir, scale="percentile", clip_percent=98):
    """Plot the residual map with and without enhancement, save both as PNG files."""
    with rasterio.open(residual_map_path) as src:
        residual = src.read(1)

    max_abs_diff = np.max(np.abs(residual))

    # Plot and save the original residual map
    plt.figure(figsize=(10, 8))
    plt.imshow(residual, cmap='RdBu', vmin=-max_abs_diff, vmax=max_abs_diff)
    plt.colorbar(label='')
    plt.axis('off')
    original_png_filename = os.path.join(output_dir, f"{src_id}_residual_map_original.png")
    plt.savefig(original_png_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Original residual map saved as {original_png_filename}")

    # Enhance the residual map
    if scale == "percentile":
        # Clip extreme values using the specified percentile
        vmin = np.percentile(residual, 100 - clip_percent)
        vmax = np.percentile(residual, clip_percent)
    else:  # Default to max absolute value (linear scale)
        vmin = -max_abs_diff
        vmax = max_abs_diff

    # Plot and save the enhanced residual map
    plt.figure(figsize=(10, 8))
    plt.imshow(residual, cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.colorbar(label='')
    plt.axis('off')
    enhanced_png_filename = os.path.join(output_dir, f"{src_id}_residual_map_enhanced.png")
    plt.savefig(enhanced_png_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Enhanced residual map saved as {enhanced_png_filename}")


def load_image(image_path):
    """Load an image and convert it to a NumPy array."""
    with rasterio.open(image_path) as dataset:
        img = dataset.read()  # (C, H, W)
        if img.shape[0] == 1:  # Single channel
            print(f"Warning: Single-channel image {image_path}. Converting to 3 channels.")
            img = np.repeat(img, 3, axis=0)  # Repeat to make it 3 channels
        img = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)
    return img


def normalize_image(img):
    """Normalize image values to the range [0, 1]."""
    img = img.astype(np.float32)
    max_val = np.max(img)
    if max_val > 1.0:
        if max_val <= 1.05:
            # Slightly above 1.0 due to floating-point errors, clip to [0, 1]
            img = np.clip(img, 0.0, 1.0)
        else:
            # Significantly above 1.0, assume in [0, 255], divide by 255
            img = img / 255.0
    # Values in [0, 1], no change needed
    return img


def eval_aoi(args):
    """Evaluate Area of Interest (AOI) by computing various metrics."""
    # Define directory paths
    pred_dsm_dir = os.path.join(args.logs_dir, 'val', 'dsm')
    pred_rgb_dir = os.path.join(args.logs_dir, 'val', 'rgb')
    gt_dsm_dir = os.path.join(args.dataset_dir, 'Truth')
    gt_rgb_base_dir = os.path.join(args.dataset_dir, 'RGB')
    epoch_number = args.epoch_number

    # Output directory to save intermediate results and residual maps
    output_dir = os.path.join(args.output_dir, 'dsm_diff')
    os.makedirs(output_dir, exist_ok=True)

    # Get list of predicted DSM files for the specified epoch
    pred_dsm_files = sorted([
        f for f in os.listdir(pred_dsm_dir)
        if f.endswith(f'_epoch{epoch_number}.tif')
    ])

    # Initialize metrics lists
    psnr_list = []
    ssim_list = []
    mae_list = []
    lpips_list = []  # New list for LPIPS

    # Iterate over all predicted DSM files
    for pred_dsm_file in pred_dsm_files:
        # Extract src_id and aoi_id from filename
        src_id = pred_dsm_file.replace(f'_epoch{epoch_number}.tif', '')
        aoi_id = '_'.join(src_id.split('_')[:2])

        print(f"\nProcessing {src_id}...")

        # Define file paths
        pred_dsm_path = os.path.join(pred_dsm_dir, pred_dsm_file)
        pred_rgb_path = os.path.join(pred_rgb_dir, f"{src_id}_epoch{epoch_number}.tif")
        gt_dsm_path = os.path.join(gt_dsm_dir, f"{aoi_id}_DSM.tif")
        gt_rgb_path = os.path.join(gt_rgb_base_dir, aoi_id, f"{src_id}.tif")

        # Check if all required files exist
        if not os.path.exists(pred_dsm_path):
            print(f"Predicted DSM not found: {pred_dsm_path}")
            continue
        if not os.path.exists(gt_dsm_path):
            print(f"GT DSM not found: {gt_dsm_path}")
            continue
        if not os.path.exists(pred_rgb_path):
            print(f"Predicted RGB not found: {pred_rgb_path}")
            continue
        if not os.path.exists(gt_rgb_path):
            print(f"GT RGB not found: {gt_rgb_path}")
            continue

        # Compute MAE and get residual map path
        mae, residual_map_path = compute_mae_and_save_dsm_diff(
            pred_dsm_path=pred_dsm_path,
            src_id=src_id,
            aoi_id=aoi_id,
            gt_dir=gt_dsm_dir,
            out_dir=output_dir,
            epoch_number=epoch_number
        )
        mae_list.append(mae)

        # Plot and save both original and enhanced residual maps
        plot_residual_map(residual_map_path, src_id, output_dir, scale="percentile", clip_percent=98)

        # Load RGB images
        pred_rgb = load_image(pred_rgb_path)
        gt_rgb = load_image(gt_rgb_path)

        # Normalize images to [0, 1] range
        pred_rgb_normalized = normalize_image(pred_rgb)
        gt_rgb_normalized = normalize_image(gt_rgb)

        # Convert to Tensor format
        pred_rgb_tensor = torch.from_numpy(pred_rgb_normalized).permute(2, 0, 1).unsqueeze(0).float()
        gt_rgb_tensor = torch.from_numpy(gt_rgb_normalized).permute(2, 0, 1).unsqueeze(0).float()

        # Compute PSNR and SSIM with max_pixel_value=1.0
        psnr_val = psnr(pred_rgb_tensor, gt_rgb_tensor, max_pixel_value=1.0).item()
        ssim_val = ssim(pred_rgb_tensor, gt_rgb_tensor).item()
        lpips_val = compute_lpips(pred_rgb_tensor, gt_rgb_tensor)

        # Append metrics to respective lists
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)

        print(f"{src_id}: PSNR {psnr_val:.3f} / SSIM {ssim_val:.3f} / LPIPS {lpips_val:.3f} / MAE {mae:.3f}")

    # Compute mean metrics
    mean_psnr = np.mean(psnr_list) if psnr_list else 0
    mean_ssim = np.mean(ssim_list) if ssim_list else 0
    mean_mae = np.mean(mae_list) if mae_list else np.nan
    mean_lpips = np.mean(lpips_list) if lpips_list else np.nan

    print(f"\nMean PSNR: {mean_psnr:.3f}")
    print(f"Mean SSIM: {mean_ssim:.3f}")
    print(f"Mean MAE: {mean_mae:.3f}")
    print(f"Mean LPIPS: {mean_lpips:.3f}\n")
    print('Eval finished!')


def Test_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True, help='Path to the project directory')
    parser.add_argument("--exp_name", type=str, required=True, help='Experiment name when training SP-NeRF')
    parser.add_argument("--dataset_dir", type=str, required=True, help='Path to the dataset directory')
    parser.add_argument("--epoch_number", type=int, default=28, help='Epoch number to evaluate')
    args = parser.parse_args()

    # Define additional paths based on input arguments
    args.logs_dir = os.path.join(args.project_dir, 'output', args.exp_name, 'logs')
    args.output_dir = os.path.join(args.project_dir, 'output', args.exp_name, 'eval')

    return args


if __name__ == '__main__':
    args = Test_parser()
    eval_aoi(args)
