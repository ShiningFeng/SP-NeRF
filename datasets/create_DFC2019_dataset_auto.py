import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import rasterio
import rpcm
import srtm4


def get_file_id(filename):
    """
    return what is left after removing directory and extension from a path
    """
    return os.path.splitext(os.path.basename(filename))[0]


def rio_open(*args, **kwargs):
    """Open a raster file with warnings suppressed."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return rasterio.open(*args, **kwargs)


def get_image_lonlat_aoi(rpc, h, w):
    """Generate GeoJSON polygon for the Area of Interest (AOI) based on RPC and image dimensions."""
    z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)  # Get elevation from SRTM4 model
    cols, rows, alts = [0, w, w, 0], [0, 0, h, h], [z] * 4  # Define image corners
    lons, lats = rpc.localization(cols, rows, alts)  # Convert pixel coords to lon/lat
    lonlat_coords = np.vstack((lons, lats)).T  # Stack into 2D array
    geojson_polygon = {"coordinates": [lonlat_coords.tolist()], "type": "Polygon"}  # Create GeoJSON
    x_c = lons.min() + (lons.max() - lons.min()) / 2  # Center longitude
    y_c = lats.min() + (lats.max() - lats.min()) / 2  # Center latitude
    geojson_polygon["center"] = [x_c, y_c]  # Add center to GeoJSON
    return geojson_polygon  # Return GeoJSON object


def run_ba(img_dir, output_dir):
    """Run Bundle Adjustment for RPC model refinement."""
    from bundle_adjust.cam_utils import SatelliteImage
    from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline
    from bundle_adjust import loader

    # Load input images and RPCs
    myimages = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    myrpcs = [rpcm.rpc_from_geotiff(p) for p in myimages]
    input_images = [SatelliteImage(fn, rpc) for fn, rpc in zip(myimages, myrpcs)]
    ba_input_data = {
        'in_dir': img_dir,
        'out_dir': os.path.join(output_dir, "ba_files"),
        'images': input_images
    }

    # Redirect stdout and stderr to log file
    os.makedirs(ba_input_data['out_dir'], exist_ok=True)
    path_to_log_file = os.path.join(ba_input_data['out_dir'], "bundle_adjust.log")
    print("Running bundle adjustment for RPC model refinement ...")
    print(f"Path to log file: {path_to_log_file}")
    log_file = open(path_to_log_file, "w+")
    sys.stdout = log_file
    sys.stderr = log_file

    # Configure and run bundle adjustment pipeline
    tracks_config = {
        'FT_reset': False,
        'FT_save': True,
        'FT_sift_detection': 's2p',
        'FT_sift_matching': 'epipolar_based'
    }
    ba_extra = {"cam_model": "rpc"}
    ba_pipeline = BundleAdjustmentPipeline(
        ba_input_data,
        tracks_config=tracks_config,
        extra_ba_config=ba_extra
    )
    ba_pipeline.run()

    # Restore stdout and stderr
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    log_file.close()
    print("... done!")
    print(f"Path to output files: {ba_input_data['out_dir']}")

    # Save bundle adjustment parameters
    ba_params_dir = os.path.join(ba_pipeline.out_dir, "ba_params")
    os.makedirs(ba_params_dir, exist_ok=True)
    np.save(os.path.join(ba_params_dir, "pts_ind.npy"), ba_pipeline.ba_params.pts_ind)
    np.save(os.path.join(ba_params_dir, "cam_ind.npy"), ba_pipeline.ba_params.cam_ind)
    np.save(os.path.join(ba_params_dir, "pts3d.npy"), ba_pipeline.ba_params.pts3d_ba - ba_pipeline.global_transform)
    np.save(os.path.join(ba_params_dir, "pts2d.npy"), ba_pipeline.ba_params.pts2d)
    fnames_in_use = [ba_pipeline.images[idx].geotiff_path for idx in ba_pipeline.ba_params.cam_prev_indices]
    loader.save_list_of_paths(os.path.join(ba_params_dir, "geotiff_paths.txt"), fnames_in_use)


def create_dataset_from_DFC2019_data(aoi_id, img_dir, dataset_dir, output_dir, use_ba=False, sun_angles_list=None):
    """
    Create JSON metadata files for each image and collect sun angles.

    Outputs include image metadata such as dimensions, RPC models, sun angles, acquisition dates, and geojson polygons.
    """
    os.makedirs(output_dir, exist_ok=True)
    path_to_dsm = os.path.join(dataset_dir, f"Truth/{aoi_id}_DSM.tif")

    # Define MSI path based on AOI ID
    if aoi_id[:3] == "JAX":
        path_to_msi = "http://138.231.80.166:2334/core3d/Jacksonville/WV3/MSI"
    elif aoi_id[:3] == "OMA":
        path_to_msi = "http://138.231.80.166:2334/core3d/Omaha/WV3/MSI"
    else:
        path_to_msi = ""

    if use_ba:
        from bundle_adjust import loader
        geotiff_paths = loader.load_list_of_paths(os.path.join(output_dir, "ba_files/ba_params/geotiff_paths.txt"))
        ba_geotiff_basenames = [os.path.basename(x) for x in geotiff_paths]
        ba_kps_pts3d_ind = np.load(os.path.join(output_dir, "ba_files/ba_params/pts_ind.npy"))
        ba_kps_cam_ind = np.load(os.path.join(output_dir, "ba_files/ba_params/cam_ind.npy"))
        ba_kps_pts2d = np.load(os.path.join(output_dir, "ba_files/ba_params/pts2d.npy"))
    else:
        geotiff_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))

    json_dir = os.path.join(output_dir, 'JSON')
    os.makedirs(json_dir, exist_ok=True)

    for rgb_p in geotiff_paths:
        d = {"img": os.path.basename(rgb_p)}
        src = rio_open(rgb_p)
        d["height"] = int(src.meta["height"])
        d["width"] = int(src.meta["width"])
        original_rpc = rpcm.RPCModel(src.tags(ns='RPC'), dict_format="geotiff")

        img_id = src.tags().get("NITF_IID2", "").replace(" ", "_")
        msi_p = f"{path_to_msi}/{img_id}.NTF" if path_to_msi else ""
        if msi_p and os.path.exists(msi_p):
            src = rio_open(msi_p)
            d["sun_elevation"] = float(src.tags().get("NITF_USE00A_SUN_EL", 0.0))
            d["sun_azimuth"] = float(src.tags().get("NITF_USE00A_SUN_AZ", 0.0))
            d["acquisition_date"] = src.tags().get('NITF_STDIDC_ACQUISITION_DATE', "")
        else:
            d["sun_elevation"] = 0.0
            d["sun_azimuth"] = 0.0
            d["acquisition_date"] = ""

        d["geojson"] = get_image_lonlat_aoi(original_rpc, d["height"], d["width"])

        src = rio_open(path_to_dsm)
        dsm = src.read(1)  # Read the first band
        d["min_alt"] = int(np.round(dsm.min() - 1))
        d["max_alt"] = int(np.round(dsm.max() + 1))

        if use_ba:
            # Use corrected RPC
            rpc_path = os.path.join(output_dir, f"ba_files/rpcs_adj/{get_file_id(rgb_p)}.rpc_adj")
            if os.path.exists(rpc_path):
                d["rpc"] = rpcm.rpc_from_rpc_file(rpc_path).__dict__
            else:
                d["rpc"] = original_rpc.__dict__

            # Additional fields for depth supervision
            ba_kps_pts3d_path = os.path.join(output_dir, "ba_files/ba_params/pts3d.npy")
            if os.path.exists(ba_kps_pts3d_path):
                shutil.copyfile(ba_kps_pts3d_path, os.path.join(json_dir, "pts3d.npy"))
            cam_idx = ba_geotiff_basenames.index(d["img"]) if d["img"] in ba_geotiff_basenames else -1
            if cam_idx != -1:
                d["keypoints"] = {
                    "2d_coordinates": ba_kps_pts2d[ba_kps_cam_ind == cam_idx, :].tolist(),
                    "pts3d_indices": ba_kps_pts3d_ind[ba_kps_cam_ind == cam_idx].tolist()
                }
        else:
            # Use original RPC
            d["rpc"] = original_rpc.__dict__

        # Save JSON metadata
        json_path = os.path.join(json_dir, f"{get_file_id(rgb_p)}.json")
        with open(json_path, "w") as f:
            json.dump(d, f, indent=2)

        # Collect sun angles if list is provided
        if sun_angles_list is not None:
            sun_angles_list.append((d["img"], d["sun_elevation"], d["sun_azimuth"]))

    return json_dir


def create_train_test_splits(input_sample_ids, test_percent=0.15, min_test_samples=2):
    """Randomly split sample IDs into training and testing sets."""

    def shuffle_array(array):
        import random
        v = array.copy()
        random.shuffle(v)
        return v

    n_samples = len(input_sample_ids)
    input_sample_ids = np.array(input_sample_ids)
    all_indices = shuffle_array(np.arange(n_samples))
    n_test = max(min_test_samples, int(test_percent * n_samples))
    n_train = n_samples - n_test

    train_indices = all_indices[:n_train]
    test_indices = all_indices[-n_test:]

    train_samples = input_sample_ids[train_indices].tolist()
    test_samples = input_sample_ids[test_indices].tolist()

    return train_samples, test_samples


def read_DFC2019_lonlat_aoi(aoi_id, dataset_dir):
    """Read DSM file and convert UTM bounding box to longitude/latitude GeoJSON."""
    from bundle_adjust import geo_utils
    zonestring = ""
    if aoi_id[:3] == "JAX":
        zonestring = "17"  # UTM Zone 17R
    elif aoi_id.lower().startswith("beijing"):
        zonestring = "50"
    elif aoi_id.lower().startswith("shanghai"):
        zonestring = "51"
    elif aoi_id.lower().startswith("changchun"):
        zonestring = "51"
    elif aoi_id.lower().startswith("lanzhou"):
        zonestring = "48"
    elif aoi_id.lower().startswith("wuhan"):
        zonestring = "50"
    elif aoi_id.lower().startswith("lhasa"):
        zonestring = "46"
    else:
        raise ValueError(f"AOI not valid. Received {aoi_id}")

    roi = np.loadtxt(os.path.join(dataset_dir, "Truth", f"{aoi_id}_DSM.txt"))

    xoff, yoff, xsize, ysize, resolution = roi[0], roi[1], int(roi[2]), int(roi[2]), roi[3]
    ulx, uly = xoff, yoff + ysize * resolution  # Upper-left coordinates
    lrx, lry = xoff + xsize * resolution, yoff  # Lower-right coordinates
    xmin, xmax, ymin, ymax = ulx, lrx, uly, lry  # Bounding box
    easts = [xmin, xmin, xmax, xmax, xmin]
    norths = [ymin, ymax, ymax, ymin, ymin]
    lons, lats = geo_utils.lonlat_from_utm(easts, norths, zonestring)
    lonlat_bbx = geo_utils.geojson_polygon(np.vstack((lons, lats)).T)
    return lonlat_bbx


def crop_geotiff_lonlat_aoi(geotiff_path, output_path, lonlat_aoi):
    """Crop a GeoTIFF image to the specified AOI and update RPC metadata."""
    with rasterio.open(geotiff_path, 'r') as src:
        profile = src.profile
        tags = src.tags()

    crop, x, y = rpcm.utils.crop_aoi(geotiff_path, lonlat_aoi)
    rpc = rpcm.rpc_from_geotiff(geotiff_path)
    rpc.row_offset -= y
    rpc.col_offset -= x

    # Update profile based on cropped image dimensions
    not_pan = len(crop.shape) > 2
    if not_pan:
        profile["height"] = crop.shape[1]
        profile["width"] = crop.shape[2]
    else:
        profile["height"] = crop.shape[0]
        profile["width"] = crop.shape[1]
        profile["count"] = 1

    # Write cropped image to output path
    with rasterio.open(output_path, 'w', **profile) as dst:
        if not_pan:
            dst.write(crop)
        else:
            dst.write(crop, 1)
        dst.update_tags(**tags)
        dst.update_tags(ns='RPC', **rpc.to_geotiff_dict())


def create_satellite_dataset(aoi_id, dataset_dir, output_dir, mode='train', ba=True, crop_aoi=True, splits=False,
                             sun_angles_list=None):
    """Prepare satellite dataset by cropping, running bundle adjustment, and generating metadata."""
    if crop_aoi:
        # Read AOI boundaries
        aoi_lonlat = read_DFC2019_lonlat_aoi(aoi_id, dataset_dir)
        img_dir = os.path.join(dataset_dir, "RGB", aoi_id)

        # Read image lists based on mode
        with open(os.path.join(dataset_dir, 'train.txt'), "r") as f:
            train_imgs = f.read().split("\n")[:-1]

        sub_dir = f"{aoi_id}_{len(train_imgs)}_imgs"
        myimages = [f"{img[:-5]}.tif" for img in train_imgs]

        if mode == 'test':
            sub_dir += '_tmp'
            with open(os.path.join(dataset_dir, 'test.txt'), "r") as f:
                test_imgs = f.read().split("\n")[:-1]
            myimages += [f"{img[:-5]}.tif" for img in test_imgs]

        print(f'Image list of {mode} mode:\n{myimages}')
        output_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Copy DSM files to output directory
        dsm_dir = os.path.join(dataset_dir, 'Truth')
        DSMTxtFile = os.path.join(dsm_dir, f"{aoi_id}_DSM.txt")
        DSMFile = os.path.join(dsm_dir, f"{aoi_id}_DSM.tif")
        dsm_copy_dir = os.path.join(output_dir, "Truth")
        os.makedirs(dsm_copy_dir, exist_ok=True)
        shutil.copyfile(DSMTxtFile, os.path.join(dsm_copy_dir, os.path.basename(DSMTxtFile)))
        shutil.copyfile(DSMFile, os.path.join(dsm_copy_dir, os.path.basename(DSMFile)))

        # Create RGB_Crops directory
        rgb_crop_dir = os.path.join(output_dir, "RGB_Crops")
        os.makedirs(rgb_crop_dir, exist_ok=True)
        rgb_crop_dir = os.path.join(rgb_crop_dir, aoi_id)
        os.makedirs(rgb_crop_dir, exist_ok=True)

        # Crop each image and save to RGB_Crops
        for img in myimages:
            rgb_crop_path = os.path.join(rgb_crop_dir, img)
            geotiff_path = os.path.join(img_dir, img)
            crop_geotiff_lonlat_aoi(geotiff_path, rgb_crop_path, aoi_lonlat)
        img_dir = rgb_crop_dir  # Update image directory to cropped images
    else:
        img_dir = os.path.join(dataset_dir, "RGB", aoi_id)

    if ba:
        run_ba(img_dir, output_dir)  # Run bundle adjustment if enabled

    # Create JSON metadata and collect sun angles
    json_dir = create_dataset_from_DFC2019_data(
        aoi_id, img_dir, dataset_dir, output_dir, use_ba=ba, sun_angles_list=sun_angles_list
    )

    # Split into train and test sets if required
    if splits:
        json_files = [os.path.basename(p) for p in glob.glob(os.path.join(json_dir, "*.json"))]
        train_samples, test_samples = create_train_test_splits(json_files)
        with open(os.path.join(json_dir, 'train.txt'), "w+") as f:
            f.write("\n".join(train_samples))
        with open(os.path.join(json_dir, 'test.txt'), "w+") as f:
            f.write("\n".join(test_samples))

    print(f'Finished processing {mode}ing images!\n')
    return output_dir, img_dir, json_dir


def config_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare DFC2019 Satellite Dataset")

    parser.add_argument("--aoi_id", type=str, required=True,
                        help='AOI ID, e.g., JAX_214')
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument("--output_dir", type=str, required=True,
                        help='Directory to save the output')

    return parser.parse_args()


if __name__ == '__main__':
    args = config_parser()

    # Initialize a list to collect sun angles
    sun_angles = []

    # Process training images
    _, img_dir_train, json_dir_train = create_satellite_dataset(
        args.aoi_id, args.dataset_dir, args.output_dir, mode='train',
        splits=True, sun_angles_list=sun_angles
    )

    # Process testing images
    dir_test, img_dir_test, json_dir_test = create_satellite_dataset(
        args.aoi_id, args.dataset_dir, args.output_dir,
        mode='test', splits=True, sun_angles_list=sun_angles
    )

    print(f'img_dir_train: {img_dir_train}')
    print(f'json_dir_train: {json_dir_train}')
    print(f'dir_test: {dir_test}')
    print(f'img_dir_test: {img_dir_test}')
    print(f'json_dir_test: {json_dir_test}')

    # Copy test images and JSON to training directory
    with open(os.path.join(args.dataset_dir, 'test.txt'), "r") as f:
        test_imgs = f.read().split("\n")[:-1]
    for test_img in test_imgs:
        src_img = os.path.join(img_dir_test, f"{test_img[:-5]}.tif")
        dst_img = os.path.join(img_dir_train, f"{test_img[:-5]}.tif")
        shutil.copyfile(src_img, dst_img)

        src_json = os.path.join(json_dir_test, test_img)
        dst_json = os.path.join(json_dir_train, test_img)
        shutil.copyfile(src_json, dst_json)

    # Copy train.txt and test.txt to training JSON directory
    shutil.copyfile(os.path.join(args.dataset_dir, 'train.txt'), os.path.join(json_dir_train, 'train.txt'))
    shutil.copyfile(os.path.join(args.dataset_dir, 'test.txt'), os.path.join(json_dir_train, 'test.txt'))

    # Remove temporary test directory
    shutil.rmtree(dir_test)

    print('Finished creating DFC2019 dataset!')

    # Write sun angles to <AOI_ID>_sunangles.txt
    sunangles_file = os.path.join(args.dataset_dir, f"{args.aoi_id}_sunangles.txt")
    with open(sunangles_file, "w") as f:
        for img, elevation, azimuth in sun_angles:
            f.write(f"{img} {elevation} {azimuth}\n")

    print(f'Sun angles file created at: {sunangles_file}')
