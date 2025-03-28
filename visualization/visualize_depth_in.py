import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize, PowerNorm


def read_points_2d(file_path):
    points_2d = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(int, line.split())
            points_2d.append((x, y))
    return np.array(points_2d)


def read_points_3d(file_path):
    points_3d = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            points_3d.append((x, y, z))
    return np.array(points_3d)


def read_tiff(file_path):
    with rasterio.open(file_path) as dataset:
        image = dataset.read([1, 2, 3])  # Read RGB bands
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C) format
    return image


# Generate a depth image with the same size as the TIFF image
def generate_padded_depth_image(image_shape, points_2d, points_3d):
    depth_image = np.full(image_shape[:2], np.nan)
    for (x, y), (_, _, z) in zip(points_2d, points_3d):
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            depth_image[y, x] = z
    return depth_image


# Visualize the raw depth data only (no special processing)
def visualize_results(depth_image, output_path='scene_comparison_utm.png'):
    """
    Use imshow to display the raw depth data without additional processing.
    Default imshow scaling applies, but no custom filtering or normalization.
    """
    plt.figure(figsize=(7, 7))
    im = plt.imshow(depth_image, cmap='viridis', interpolation='nearest')
    plt.title('Raw Depth Visualization (No Additional Processing)')
    plt.axis('off')
    plt.colorbar(im, label='Depth (Z value)')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Overlay the depth map on top of the TIFF image (with optional filtering and normalization)
def overlay_depth_on_tiff(tiff_image, depth_image, output_path='scene_overlay.png',
                          filter_values=False, use_power_norm=False, adjust_max=False):
    """
    Args:
        filter_values: Keep only depth values > -10 if True
        use_power_norm: Use PowerNorm for non-linear scaling if True
        adjust_max: Reduce the max depth by 10% if True to enhance color contrast
    """
    # Optionally filter out depth values below a threshold
    if filter_values:
        depth_image_filtered = np.where(depth_image > -10, depth_image, np.nan)
    else:
        depth_image_filtered = depth_image

    # Determine maximum depth for normalization
    vmax_value = np.nanmax(depth_image_filtered)
    if adjust_max and not np.isnan(vmax_value):
        vmax_value = 0.9 * vmax_value

    # Determine if a non-linear scaling is used
    if use_power_norm:
        norm = PowerNorm(gamma=0.5, vmin=np.nanmin(depth_image_filtered), vmax=vmax_value)
    else:
        norm = Normalize(vmin=np.nanmin(depth_image_filtered), vmax=vmax_value)

    # Generate RGBA array with a chosen colormap
    depth_colored = plt.cm.Blues(norm(depth_image_filtered))
    depth_colored[:, :, 3] = 0.8  # Apply uniform transparency
    depth_colored[np.isnan(depth_image_filtered)] = [0, 0, 0, 0]  # Set NaN areas to be fully transparent

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(tiff_image)
    ax.imshow(depth_colored, alpha=1.0)
    ax.set_title('Depth Overlay on TIFF Image')
    ax.axis('off')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Side-by-side visualization of TIFF image and the depth map
def visualize_tiff_depth_side_by_side(tiff_image, depth_image, output_path='scene_side_by_side.png'):
    """
    Show the original TIFF image on the left and the depth image on the right with a colorbar.
    """
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color='gray')  # Handle NaN values with a gray color
    norm = Normalize(vmin=np.nanmin(depth_image), vmax=np.nanmax(depth_image))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(tiff_image)
    axes[0].set_title('TIFF Image (Color)')
    axes[0].axis('off')

    im = axes[1].imshow(depth_image, cmap=cmap, norm=norm)
    axes[1].set_title('Depth Image')
    axes[1].axis('off')

    fig.colorbar(im, ax=axes[1], label='Depth (Z value)')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the output directory one level up in 'output_visual'
    output_dir = os.path.join(current_dir, '..', 'output_visual')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define file paths for 2D points, 3D points, and the TIFF image
    file_2d = os.path.join(current_dir, '..', 'dataset', 'DFC2019', 'Depth', 'JAX_214_009_RGB_2DPts.txt')
    file_utm = os.path.join(current_dir, '..', 'dataset', 'DFC2019', 'Depth', 'JAX_214_009_RGB_3DPts.txt')
    tiff_path = os.path.join(current_dir, '..', 'dataset', 'DFC2019', 'RGB', 'JAX_214', 'JAX_214_009_RGB.tif')

    # Read data from text files and TIFF
    points_2d = read_points_2d(file_2d)
    utm_points_3d = read_points_3d(file_utm)
    tiff_image = read_tiff(tiff_path)

    # Generate a depth image matching the TIFF dimensions
    depth_image_utm = generate_padded_depth_image(tiff_image.shape, points_2d, utm_points_3d)

    # Visualize the raw depth image (no overlay or special processing)
    visualize_results(depth_image_utm, output_path=os.path.join(output_dir, 'scene_comparison_utm.png'))

    # Side-by-side comparison of TIFF image and depth map
    visualize_tiff_depth_side_by_side(tiff_image, depth_image_utm,
                                      output_path=os.path.join(output_dir, 'scene_side_by_side.png'))

    # Overlay the depth map onto the TIFF image with different options
    overlay_depth_on_tiff(tiff_image, depth_image_utm,
                          output_path=os.path.join(output_dir, 'scene_overlay_no_filter_no_norm.png'),
                          filter_values=False, use_power_norm=False, adjust_max=False)

    overlay_depth_on_tiff(tiff_image, depth_image_utm,
                          output_path=os.path.join(output_dir, 'scene_overlay_filter_no_norm.png'),
                          filter_values=True, use_power_norm=False, adjust_max=False)

    overlay_depth_on_tiff(tiff_image, depth_image_utm,
                          output_path=os.path.join(output_dir, 'scene_overlay_no_filter_power_norm.png'),
                          filter_values=False, use_power_norm=True, adjust_max=False)

    overlay_depth_on_tiff(tiff_image, depth_image_utm,
                          output_path=os.path.join(output_dir, 'scene_overlay_filter_power_norm.png'),
                          filter_values=True, use_power_norm=True, adjust_max=False)

    overlay_depth_on_tiff(tiff_image, depth_image_utm,
                          output_path=os.path.join(output_dir, 'scene_overlay_filter_power_norm_adjusted_max.png'),
                          filter_values=True, use_power_norm=True, adjust_max=True)
