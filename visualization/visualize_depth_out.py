import warnings

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def visualize_and_save_dsm(dsm_path, output_path):
    """
    Visualize the DSM and save the result as an image.

    Parameters:
        dsm_path (str): Path to the DSM TIF file.
        output_path (str): Path to save the visualization output.
    """
    # Ignore georeference warnings
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # Open the DSM file
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)  # Read the first band

    # Replace NaN values with the minimum DSM value
    dsm_min = np.nanmin(dsm)
    dsm[np.isnan(dsm)] = dsm_min

    # Create a colormap
    cmap = plt.cm.viridis

    # Plot the DSM data
    plt.figure(figsize=(10, 8))
    plt.imshow(dsm, cmap=cmap, vmin=dsm_min, vmax=np.nanmax(dsm))
    cbar = plt.colorbar()
    cbar.set_label("")  # You can customize the label if needed
    plt.axis('off')

    # Save the visualization as a PNG image
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Visualization saved to: {output_path}")


# Specify input DSM TIF file path and output image path
dsm_tif_path = "path/to/your/dsm.tif"
output_image_path = "path/to/your/output.png"

# Call the function to visualize and save the DSM
visualize_and_save_dsm(dsm_tif_path, output_image_path)
