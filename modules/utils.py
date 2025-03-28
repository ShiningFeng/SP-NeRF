"""
This script integrates the original train_utils and sat_utils,
combining functions for handling satellite images, georeferenced data, and essential utilities for the training process.
"""

import datetime
import glob
import json
import os
import shutil

import cv2
import numpy as np
import rasterio
import rpcm
import torch
import torchvision.transforms as T
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, StepLR

from .opt import SEMANTIC_CONFIG

"""
===============================================
Section 1: sat_utils
===============================================
"""


def get_file_id(filename):
    """
    return what is left after removing directory and extension from a path
    """
    return os.path.splitext(os.path.basename(filename))[0]


def read_dict_from_json(input_path):
    with open(input_path) as f:
        d = json.load(f)
    return d


def write_dict_to_json(d, output_path):
    with open(output_path, "w") as f:
        json.dump(d, f, indent=2)
    return d


def rpc_scaling_params(v):
    """
    find the scale and offset of a vector
    """
    vec = np.array(v).ravel()
    scale = (vec.max() - vec.min()) / 2
    offset = vec.min() + scale
    return scale, offset


def rescale_rpc(rpc, alpha):
    """
    Scale a rpc model following an image resize
    Args:
        rpc: rpc model to scale
        alpha: resize factor
               e.g. 2 if the image is upsampled by a factor of 2
                    1/2 if the image is downsampled by a factor of 2
    Returns:
        rpc_scaled: the scaled version of P by a factor alpha
    """
    import copy

    rpc_scaled = copy.copy(rpc)
    rpc_scaled.row_scale *= float(alpha)
    rpc_scaled.col_scale *= float(alpha)
    rpc_scaled.row_offset *= float(alpha)
    rpc_scaled.col_offset *= float(alpha)
    return rpc_scaled


def geodetic_to_ecef(lat, lon, alt):
    """
    convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    """
    # WGS-84 ellipsiod parameters
    a = 6378137.0  # Semi-major axis in meters
    b = 6356752.314245  # Semi-minor axis in meters
    e2 = 1 - (b ** 2 / a ** 2)  # First eccentricity squared

    # Convert latitude, longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)  # Calculate N

    # Calculate ECEF coordinates
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((b ** 2 / a ** 2) * N + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_latlon_custom(x, y, z):
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a ** 2
    esq = e ** 2
    b = np.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt((x ** 2) + (y ** 2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + (ep ** 2) * b * (np.sin(th) ** 3)), (p - esq * a * (np.cos(th) ** 3)))
    N = a / (np.sqrt(1 - esq * (np.sin(lat) ** 2)))
    alt = p / np.cos(lat) - N
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    return lat, lon, alt


def utm_from_latlon(lats, lons):
    """
    convert lat-lon to utm
    """
    import pyproj
    import utm
    from pyproj import Transformer

    n = utm.latlon_to_zone_number(lats[0], lons[0])  # Get UTM zone number
    l = utm.latitude_to_zone_letter(lats[0])  # Get UTM zone letter
    proj_src = pyproj.Proj("+proj=latlong")  # Source projection (lat/lon)
    proj_dst = pyproj.Proj(f"+proj=utm +zone={n}{l}")  # Destination projection (UTM)
    transformer = Transformer.from_proj(proj_src, proj_dst)  # Coordinate transformer
    easts, norths = transformer.transform(lons, lats)  # Transform to UTM coordinates
    return easts, norths  # Return UTM easting and northing


def dsm_pointwise_diff(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None, out_rdsm_path=None,
                       out_err_path=None):
    """
    Computes pointwise differences between a generated DSM and a reference DSM.
    """
    from osgeo import gdal

    # Create a unique identifier for temporary files
    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    pred_dsm_path = f"tmp_crop_dsm_to_delete_{unique_identifier}.tif"
    pred_rdsm_path = f"tmp_crop_rdsm_to_delete_{unique_identifier}.tif"

    # Read DSM metadata
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]  # Offsets in x and y directions
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])  # Size of the bounding box
    resolution = dsm_metadata[3]  # Resolution in meters per pixel

    # Define the bounding box for cropping using GDAL translate
    ulx, uly = xoff, yoff + ysize * resolution  # Upper-left corner
    lrx, lry = xoff + xsize * resolution, yoff  # Lower-right corner

    # Crop the predicted DSM to the bounding box
    ds = gdal.Open(in_dsm_path)  # Open the input DSM
    ds = gdal.Translate(pred_dsm_path, ds, projWin=[ulx, uly, lrx, lry])  # Crop the DSM
    ds = None  # Close the dataset to release resources

    # If a water mask is provided, apply it to the cropped DSM
    if gt_mask_path is not None:
        with rasterio.open(gt_mask_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
        with rasterio.open(pred_dsm_path, "r") as f:
            profile = f.profile
            pred_dsm = f.read()[0, :, :]
        pred_dsm[water_mask.astype(bool)] = np.nan  # Set the DSM values in water areas to NaN
        with rasterio.open(pred_dsm_path, 'w', **profile) as dst:
            dst.write(pred_dsm, 1)

        # read predicted and gt dsms
        with rasterio.open(gt_dsm_path, "r") as f:
            gt_dsm = f.read()[0, :, :]
        with rasterio.open(pred_dsm_path, "r") as f:
            profile = f.profile
            pred_dsm = f.read()[0, :, :]

    # Try to import DSMR library for registration, if available
    fix_xy = False
    try:
        from modules import dsmr
    except ImportError:
        print("Warning: dsmr not found! DSM registration will only use the Z dimension")
        fix_xy = True  # Fall back to simple Z adjustment if DSMR is not available

    if fix_xy:
        # Adjust DSM by mean Z difference if DSMR is not available
        pred_rdsm = pred_dsm + np.nanmean(gt_dsm - pred_dsm)
        with rasterio.open(pred_rdsm_path, 'w', **profile) as dst:
            dst.write(pred_rdsm, 1)
    else:
        # Use DSMR to compute and apply the transformation for better registration
        from modules import dsmr
        transform = dsmr.compute_shift(gt_dsm_path, pred_dsm_path, scaling=False)  # Compute shift
        dsmr.apply_shift(pred_dsm_path, pred_rdsm_path, *transform)  # Apply the shift
        with rasterio.open(pred_rdsm_path, "r") as f:
            pred_rdsm = f.read(1)  # Read the registered DSM
    err = pred_rdsm - gt_dsm

    os.remove(pred_dsm_path)  # Remove the cropped DSM file
    if out_rdsm_path is not None:
        if os.path.exists(out_rdsm_path):
            os.remove(out_rdsm_path)  # Remove existing file if it exists
        os.makedirs(os.path.dirname(out_rdsm_path), exist_ok=True)  # Ensure directory exists
        shutil.copyfile(pred_rdsm_path, out_rdsm_path)  # Copy the file to the specified location

    os.remove(pred_rdsm_path)  # Remove the temporary registered DSM file
    if out_err_path is not None:
        if os.path.exists(out_err_path):
            os.remove(out_err_path)  # Remove existing file if it exists
        os.makedirs(os.path.dirname(out_err_path), exist_ok=True)  # Ensure directory exists
        with rasterio.open(out_err_path, 'w', **profile) as dst:
            dst.write(err, 1)  # Write the error map to the specified location

    return err  # Return the computed error map


def compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, aoi_id, gt_dir, out_dir, epoch_number, save=True):
    gt_dsm_path = os.path.join(gt_dir, f"{aoi_id}_DSM.tif")
    gt_roi_path = os.path.join(gt_dir, f"{aoi_id}_DSM.txt")

    gt_seg_path = None
    assert os.path.exists(gt_roi_path), f"{gt_roi_path} not found"
    assert os.path.exists(gt_dsm_path), f"{gt_dsm_path} not found"

    gt_roi_metadata = np.loadtxt(gt_roi_path)
    rdsm_diff_path = os.path.join(out_dir, f"{src_id}_rdsm_diff_epoch{epoch_number}.tif")
    rdsm_path = os.path.join(out_dir, f"{src_id}_rdsm_epoch{epoch_number}.tif")
    diff = dsm_pointwise_diff(pred_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path,
                              out_rdsm_path=rdsm_path, out_err_path=rdsm_diff_path)
    if not save:
        os.remove(rdsm_diff_path)
        os.remove(rdsm_path)
    return np.nanmean(abs(diff.ravel()))


def sort_by_increasing_view_incidence_angle(json_dir):
    incidence_angles = []
    json_paths = glob.glob(os.path.join(json_dir, "*.json"))
    for json_p in json_paths:
        with open(json_p) as f:
            d = json.load(f)
        rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")
        c_lon, c_lat = d["geojson"]["center"][0], d["geojson"]["center"][1]
        alpha, _ = rpc.incidence_angles(c_lon, c_lat, z=0)  # alpha = view incidence angle in degrees
        incidence_angles.append(alpha)
    return [x for _, x in sorted(zip(incidence_angles, json_paths))]


def sort_by_increasing_solar_incidence_angle(json_dir):
    solar_incidence_angles = []
    json_paths = glob.glob(os.path.join(json_dir, "*.json"))
    for json_p in json_paths:
        with open(json_p) as f:
            d = json.load(f)
        sun_el = np.radians(float(d["sun_elevation"]))
        sun_az = np.radians(float(d["sun_azimuth"]))
        sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
        surface_normal = np.array([0., 0., 1.0])
        u1 = sun_d / np.linalg.norm(sun_d)
        u2 = surface_normal / np.linalg.norm(surface_normal)
        alpha = np.degrees(np.arccos(np.dot(u1, u2)))  # alpha = solar incidence angle in degrees
        solar_incidence_angles.append(alpha)
    return [x for _, x in sorted(zip(solar_incidence_angles, json_paths))]


"""
===============================================
Section 2: Train_utils
===============================================
"""


def get_epoch_number_from_train_step(train_step, dataset_len, batch_size):
    return int(train_step // (dataset_len // batch_size))


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_parameters(models):
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:
        # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters


def get_scheduler(optimizer, lr_scheduler, num_epochs):
    eps = 1e-8
    if lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eps)
    elif lr_scheduler == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.01)
    elif lr_scheduler == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
        # scheduler = MultiStepLR(optimizer, milestones=[50,100,200], gamma=0.5)
    elif lr_scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    else:
        raise ValueError('lr scheduler not recognized!')
    return scheduler


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    Converts depth map to a colorized image for output_visual.

    Parameters:
    - depth: Tensor of depth values (H, W).
    - cmap: (Optional) Color map for output_visual. Default is 'Jet'.
    """
    x = depth.cpu().numpy()  # Convert tensor to numpy array on CPU
    x = np.nan_to_num(x)  # Replace NaNs with 0
    mi, ma = np.min(x), np.max(x)  # Find min and max depth values
    x = (x - mi) / (ma - mi + 1e-8)  # Normalize to [0, 1]
    x = (255 * x).astype(np.uint8)  # Scale to [0, 255]
    x = np.clip(x, 0, 255)  # Clip to valid range
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))  # Convert to color image
    x_ = T.ToTensor()(x_)  # Convert to PyTorch tensor
    return x_  # Return colorized depth tensor


def save_output_image(input, output_path, source_path):
    """
    Saves an input image as a rasterio-compatible image.

    Parameters:
    - input: Input image data (D, H, W), where D is the number of channels, can be a pytorch tensor or a numpy array
    - output_path: Path to save the output image.
    - source_path: Path to the source image (used for profile settings).
    """
    # convert input to numpy array float32
    if torch.is_tensor(input):
        im_np = input.type(torch.FloatTensor).cpu().numpy()
    else:
        im_np = input.astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(source_path, 'r') as src:
        profile = src.profile
        profile["dtype"] = rasterio.float32
        profile["height"] = im_np.shape[1]
        profile["width"] = im_np.shape[2]
        profile["count"] = im_np.shape[0]
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(im_np)


def convert_semantic_to_color(sem_pred, num_sem_classes):
    """
    Converts semantic labels to RGB color images for output_visual.

    Args:
        sem_pred (np.ndarray): Semantic prediction map with labels (H, W).

    Returns:
        np.ndarray: Color image corresponding to semantic labels (H, W, 3).
    """
    color_mapping = SEMANTIC_CONFIG[num_sem_classes]["color_mapping"]

    # Initialize the color image
    height, width = sem_pred.shape
    color_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # Default to white

    # Assign colors to pixels based on their labels
    for label, color in color_mapping.items():
        mask = (sem_pred == label)
        color_image[mask] = color

    return color_image


def remap_semantics_to_original(sem_pred, num_sem_classes):
    """
    Remaps the internal semantic indices back to the original classification IDs.

    Args:
        sem_pred (np.ndarray): Semantic map with internal indices (H, W).

    Returns:
        np.ndarray: Semantic map with original classification IDs (H, W).
    """
    class_mapping = SEMANTIC_CONFIG[num_sem_classes]["class_mapping"]

    # Map internal indices to the original classification IDs
    remapped_sem = np.full_like(sem_pred, 65, dtype=np.int32)  # Default to 65 (Unlabeled)
    for internal_idx, original_id in class_mapping.items():
        remapped_sem[sem_pred == internal_idx] = original_id

    return remapped_sem


def save_sem_image(sem_pred, output_path, num_sem_classes):
    """
    Saves a semantic segmentation image as a colored map with a legend.

    Args:
        sem_pred (np.ndarray): Grayscale semantic map with labels (H, W).
        output_path (str): Path to save the output output_visual image.
    """
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt

    color_mapping = SEMANTIC_CONFIG[num_sem_classes]["color_mapping"]
    semantic_names = SEMANTIC_CONFIG[num_sem_classes]["semantic_names"]

    # Convert the semantic prediction to uint8 type
    sem_pred = sem_pred.astype(np.uint8)
    height, width = sem_pred.shape
    visualization = np.ones((height, width, 3), dtype=np.uint8) * 255  # Default to white for areas not in mapping

    # Assign colors to pixels based on their label
    for label, rgb_color in color_mapping.items():
        mask = (sem_pred == label)
        visualization[mask] = rgb_color

    # Create a plot for output_visual and save it with a legend
    plt.figure(figsize=(12, 12))
    plt.imshow(visualization, interpolation='nearest')
    plt.axis('off')

    # Prepare legend entries
    legend_labels = [semantic_names[label] for label in sorted(semantic_names.keys())]
    legend_colors = [np.array(color_mapping[label]) / 255 for label in sorted(color_mapping.keys())]
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                   markerfacecolor=color, markersize=10, linestyle='None')
        for class_name, color in zip(legend_labels, legend_colors)
    ]
    plt.legend(handles=handles, loc='upper right', title="Classes")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Save an additional visualization without a legend
    fig_no_legend_path = os.path.splitext(output_path)[0] + "_no_legend" + os.path.splitext(output_path)[1]
    plt.figure(figsize=(12, 12))
    plt.imshow(visualization, interpolation='nearest')
    plt.axis('off')
    plt.savefig(fig_no_legend_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
