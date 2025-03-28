import os

from osgeo import gdal


def convert_tiff(in_tiff, out_tiff, compress_type="NONE"):
    """
    Convert a TIFF file using GDAL to a specified compression format (default is no compression).

    :param in_tiff:      Input TIFF file path
    :param out_tiff:     Output TIFF file path
    :param compress_type: Compression type. Default is "NONE" (no compression).
                          Can be set to other formats such as "PACKBITS", "LZW", etc.
    """
    ds = gdal.Open(in_tiff, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Unable to open file: {in_tiff}")

    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("Could not get GTiff driver")

    creation_options = [
        f"COMPRESS={compress_type}"
    ]

    out_ds = driver.CreateCopy(out_tiff, ds, strict=1, options=creation_options)

    ds = None
    out_ds = None

    print(f"[OK] Successfully converted {in_tiff} to {out_tiff} (COMPRESS={compress_type})")


def batch_convert_tiffs(input_tiffs, out_dir, compress_type="NONE"):
    """
    Batch convert multiple TIFF files and save them in a specified directory (file names remain unchanged).

    :param input_tiffs:   List of input TIFF file paths
    :param out_dir:       Output directory (created if it does not exist)
    :param compress_type: Compression type, default is "NONE"
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for in_tiff in input_tiffs:
        base_name = os.path.basename(in_tiff)
        out_tiff = os.path.join(out_dir, base_name)
        convert_tiff(in_tiff, out_tiff, compress_type)


if __name__ == "__main__":
    # Specify the input TIFF file paths to be converted
    input_files = [
        "path/to/your/dataset/DFC2019_269/RGB/JAX_269/JAX_269_006_RGB.tif",
    ]

    # Specify the output directory
    output_dir = "path/to/your/output/directory"

    # Specify the compression mode ("NONE", "PACKBITS", "LZW", etc.)
    compress_mode = "NONE"

    # Perform batch conversion
    batch_convert_tiffs(input_files, output_dir, compress_mode)
