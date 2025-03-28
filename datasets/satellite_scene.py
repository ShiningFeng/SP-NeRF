"""
This script defines the dataloader for a dataset of multi-view satellite images， depth and semantic segmentation label
"""

import glob
import os

import numpy as np
import rasterio
import rpcm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from modules import utils
from modules.opt import SEMANTIC_CONFIG


def get_rays(cols, rows, rpc, min_alt, max_alt):
    """
    Draw a set of rays from a satellite image
    Each ray is defined by an origin 3d point + a direction vector
    First the bounds of each ray are found by localizing each pixel at min and max altitude
    Then the corresponding direction vector is found by the difference between such bounds
    Args:
        cols: 1d array with image column coordinates
        rows: 1d array with image row coordinates
        rpc: RPC model with the localization function associated to the satellite image
        min_alt: float, the minimum altitude observed in the image
        max_alt: float, the maximum altitude observed in the image
    Returns:
        rays: (h*w, 8) tensor of floats encoding h*w rays
        columns 0,1,2 correspond to the rays origin
        columns 3,4,5 correspond to the direction vector
        columns 6,7 correspond to the distance of the ray bounds with respect to the camera
    """
    min_alts = float(min_alt) * np.ones(cols.shape)
    max_alts = float(max_alt) * np.ones(cols.shape)

    # assume the points of maximum altitude are those closest to the camera
    lons, lats = rpc.localization(cols, rows, max_alts)
    x_near, y_near, z_near = utils.geodetic_to_ecef(lats, lons, max_alts)
    xyz_near = np.vstack([x_near, y_near, z_near]).T

    # similarly, the points of minimum altitude are the furthest away from the camera
    lons, lats = rpc.localization(cols, rows, min_alts)
    x_far, y_far, z_far = utils.geodetic_to_ecef(lats, lons, min_alts)
    xyz_far = np.vstack([x_far, y_far, z_far]).T

    # define the rays origin as the nearest point coordinates
    rays_o = xyz_near

    # define the unit direction vector
    d = xyz_far - xyz_near
    rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

    # assume the nearest points are at distance 0 from the camera
    # the furthest points are at distance Euclidean distance(far - near)
    fars = np.linalg.norm(d, axis=1)
    nears = float(0) * np.ones(fars.shape)

    # create a stack with the rays origin, direction vector and near-far bounds
    rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))
    rays = rays.type(torch.FloatTensor)

    return rays


def load_tensor_from_rgb_geotiff(img_path, downscale_factor, imethod=Image.BILINEAR):
    # Sat-NeRF method(Image.BICUBIC) will lead to noisy pixel errors
    with rasterio.open(img_path, 'r') as f:  # (3, h, w)
        img = np.transpose(f.read(), (1, 2, 0)) / 255.  # (h, w, 3)
    h, w = img.shape[:2]
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        img = np.transpose(img, (2, 0, 1))  # (h, w, 3) -> (3, h, w)
        img = T.Resize(size=(h, w), interpolation=imethod)(torch.Tensor(img))  # T.Resize need (channels, height, width)
        img = np.transpose(img.numpy(), (1, 2, 0))  # (3, h, w) -> (h, w, 3)
    img = T.ToTensor()(img)  # (3, h, w) & normalization
    rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    rgbs = rgbs.type(torch.FloatTensor)

    return rgbs


class SatelliteSceneDataset(Dataset):
    def __init__(self, depth_dir, json_dir, img_dir, sem_dir, aoi_id, cache_dir=None, split="train",
                 img_downscale=1.0, stdscale=1, margin=0, dense_ss=False, sem_downscale=8, sem=False,
                 num_sem_classes=5):
        """
        NeRF Satellite Dataset
        Args:
            json_dir: string, directory containing the json files with all relevant metadata per image
            img_dir: string, directory containing all the satellite images (may be different from json_dir)
            split: string, either 'train' or 'val'
            img_downscale: float, image downscale factor
            cache_dir: string, directory containing precomputed rays
        """
        self.depth_dir = depth_dir
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.aoi_id = aoi_id
        self.sem_dir = os.path.join(sem_dir, self.aoi_id + '_CLS.tif')
        self.cache_dir = cache_dir
        self.train = split == "train"
        self.img_downscale = float(img_downscale)
        self.stdscale = stdscale
        self.margin = margin
        self.sem = sem
        self.num_sem_classes = num_sem_classes
        self.dense_ss = dense_ss
        self.sem_downscale = sem_downscale

        # load scaling params
        if not os.path.exists(f"{self.json_dir}/scene.loc"):
            self.init_scaling_params()
        else:
            print(f"{self.json_dir}/scene.loc already exist, hence skipped scaling")
        d = utils.read_dict_from_json(os.path.join(self.json_dir, "scene.loc"))
        self.center = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
        self.range = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))

        # load dataset split
        if self.train:
            self.load_train_split()
        else:
            self.load_val_split()

    def load_train_split(self):
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        json_files = json_files[:-1]
        n_train_ims = len(json_files)
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        self.all_rays, self.all_rgbs, self.all_ids = self.load_data(self.json_files, verbose=True)
        self.all_deprays, self.all_depths, self.all_valid_depth, self.all_depth_stds = self.load_depth_data(
            self.json_files, self.depth_dir, verbose=True)
        if self.sem:
            self.all_semantics, self.all_valid_semantics = self.load_semantic_data(
                self.sem_dir, self.json_files, self.num_sem_classes, verbose=True)

    def load_val_split(self):
        with open(os.path.join(self.json_dir, "test.txt"), "r") as f:
            json_files = f.read().split("\n")
        json_files = json_files[:-1]  # # Remove the last empty element if any
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]

        # # add an extra image from the training set to the validation set (for debugging purposes)
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        json_files = json_files[:-1]
        n_train_ims = len(json_files)
        self.all_ids = [i + n_train_ims for i, j in enumerate(self.json_files)]
        self.json_files = [os.path.join(self.json_dir, json_files[0])] + self.json_files
        self.all_ids = [0] + self.all_ids

    def load_data(self, json_files, verbose=False):
        """
        Load all relevant information from a set of json files
        Args:
            json_files: list containing the path to the input json files
            verbose (bool, optional): If True, prints additional information during loading. Defaults to False
        Returns:
            all_rays: (N, 11) tensor of floats encoding all ray-related parameters corresponding to N rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
                      columns 8,9,10 correspond to the sun direction vectors
            all_rgbs: (N, 3) tensor of floats encoding all the rgb colors corresponding to N rays
        """
        all_rgbs, all_rays, all_sun_dirs, all_ids = [], [], [], []
        for t, json_p in enumerate(json_files):
            if os.path.exists(json_p) == False or os.path.isfile(json_p) == False:
                print(json_p, 'not exist or is not a file, hence skipped')
                continue

            d = utils.read_dict_from_json(json_p)
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = utils.get_file_id(d["img"])

            # get rgb colors
            rgbs = load_tensor_from_rgb_geotiff(img_p, self.img_downscale)

            # get rays
            cache_path = f"{self.cache_dir}/{img_id}.data"
            if self.cache_dir is not None and os.path.exists(cache_path):
                rays = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                # numpy flatten, default 'C', means in row-major (C-style) order.
                rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
                if self.cache_dir is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(rays, cache_path)
            rays = self.normalize_rays(rays)

            # get sun direction
            sun_dirs = self.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])

            all_ids += [t * torch.ones(rays.shape[0], 1)]
            all_rgbs += [rgbs]
            all_rays += [rays]
            all_sun_dirs += [sun_dirs]
            if verbose:
                print(f"Image {img_id} loaded ({t + 1} / {len(json_files)} )")

        all_ids = torch.cat(all_ids, 0)
        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        all_rays = torch.hstack([all_rays, all_sun_dirs])  # (len(json_files)*h*w, 11)
        all_rays = all_rays.type(torch.FloatTensor)
        all_rgbs = all_rgbs.type(torch.FloatTensor)

        return all_rays, all_rgbs, all_ids

    def load_depth_data(self, json_files, depth_dir, verbose=False):
        """
            Load depth data and process 2D/3D points and correlation scores
            Returns:
                all_deprays: (N, 8) tensor of ray parameters (origin, direction, bounds)
                all_depths: (N, 2) tensor of depth values and correlation scores
                all_valid_depth: (N,) tensor of binary values (1: valid, 0: invalid)
                all_depth_stds: (N,) tensor of depth uncertainty (standard deviation)
        """
        all_deprays, all_depths, all_weights, all_depth_stds, all_valid_depth = [[] for _ in range(5)]
        depth_min, depth_max = None, None

        for t, json_p in enumerate(json_files):
            d = utils.read_dict_from_json(json_p)
            img_id = utils.get_file_id(d["img"])
            height, width = d["height"], d["width"]

            # read 2D and 3D point data, and weight correlations
            pts2d = np.loadtxt(os.path.join(depth_dir, f"{img_id}_2DPts.txt"), dtype='int').reshape(-1, 2)
            pts3d = torch.FloatTensor(
                np.loadtxt(os.path.join(depth_dir, f"{img_id}_3DPts_ecef.txt"), dtype='float').reshape(-1, 3))
            current_weights = torch.FloatTensor(
                np.loadtxt(os.path.join(depth_dir, f"{img_id}_Correl.txt"), dtype='float'))

            # normalize current_weights to a range of [0, 1]
            current_weights = (current_weights - current_weights.min()) / (
                    current_weights.max() - current_weights.min())

            # create and rescale RPC model, adjust points for image downscale
            rpc = utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
            pts2d = pts2d / self.img_downscale
            cols, rows = pts2d.T
            rays = self.normalize_rays(get_rays(cols, rows, rpc, float(d["min_alt"]), float(d["max_alt"])))

            # normalize 3D points by subtracting center and dividing by range
            pts3d = (pts3d - torch.tensor(self.center)) / self.range

            # compute depth values and standard deviation
            depths = torch.linalg.norm(pts3d - rays[:, :3], axis=1)
            current_depth_std = self.stdscale * (1 - current_weights) + self.margin

            # update minimum and maximum depth values
            depth_min = depths.min() if depth_min is None else min(depth_min, depths.min())
            depth_max = depths.max() if depth_max is None else max(depth_max, depths.max())

            # create valid depth mask
            valid_depth = torch.zeros(height, width)
            valid_depth[pts2d[:, 1], pts2d[:, 0]] = 1
            valid_depth = valid_depth.flatten()
            valid_mask = np.where(valid_depth > 0)[0]  # Get indices of valid depths

            # pad, scale, and add data to respective lists
            all_deprays.append(self.prepare_padded_tensor(rays, valid_mask, height, width, depth=8))  # depth = 8
            all_depths.append(self.prepare_padded_tensor(depths, valid_mask, height, width)[:, np.newaxis])
            all_weights.append(self.prepare_padded_tensor(current_weights, valid_mask, height, width)[:, np.newaxis])
            all_depth_stds.append(self.prepare_padded_tensor(current_depth_std, valid_mask, height, width))
            all_valid_depth.append(self.scale_depth(valid_depth, height, width))

            if verbose:
                print(f"Depth {img_id} loaded ({t + 1} / {len(json_files)} )")
                print(
                    f'depth range: [{torch.min(depths):.5f}, {torch.max(depths):.5f}], mean: {torch.mean(depths):.5f}')
                print(
                    f'corr  range: [{torch.min(current_weights):.5f}, {torch.max(current_weights):.5f}], mean: {torch.mean(current_weights):.5f}')
                print(
                    f'std   range: [{torch.min(current_depth_std):.5f}, {torch.max(current_depth_std):.5f}], mean: {torch.mean(current_depth_std):.5f}')
                print(f'{depths.shape[0] * 100.0 / height / width:.5f} percent of pixels are valid in depth map.')

        all_deprays = torch.cat(all_deprays, 0).float()  # (len(json_files)*h*w, 8)
        all_depths = torch.hstack(
            [torch.cat(all_depths, 0), torch.cat(all_weights, 0)]).float()  # # (len(json_files)*h*w, 2)
        all_valid_depth = torch.cat(all_valid_depth, 0)
        all_depth_stds = torch.cat(all_depth_stds, 0) * (depth_max - depth_min)

        return all_deprays, all_depths, all_valid_depth, all_depth_stds

    def load_semantic_data(self, sem_dir, json_files, num_sem_classes, verbose=False, sem_downscale=8):
        """
            Load semantic data and process semantic labels
            Args:
                sem_downscale: Factor for downscaling and upscaling semantic labels (default: 8)
            Returns:
                all_semantics: (N, 1) tensor of class indices for semantic labels
                all_valid_semantics: (N,) tensor indicating valid (1) or void (0) labels
            Notes:
                Upscaling modes:
                1. Sparse: Supervision on every `sem_downscale` pixel, interpolated pixels set to void
                2. Dense: Supervision on all pixels, including interpolated ones
        """
        all_semantics, all_valid_semantics = [], []

        with rasterio.open(sem_dir) as src:
            semantic = src.read(1)
        semantic_labels = torch.from_numpy(semantic).type(torch.LongTensor)  # (H, W)

        mapped_labels = -100 * torch.ones_like(semantic_labels)  # Initialize mapped labels to -100

        # Define the label mapping
        label_mapping = SEMANTIC_CONFIG[num_sem_classes]["label_mapping"]

        # Apply the label mapping
        for original_label, mapped_label in label_mapping.items():
            mapped_labels[semantic_labels == original_label] = mapped_label
        sem_height, sem_width = mapped_labels.shape

        # Iterate over each JSON file and apply semantic data
        for t, json_p in enumerate(json_files):
            metadata = utils.read_dict_from_json(json_p)
            image_id = utils.get_file_id(metadata["img"])
            height, width = metadata["height"], metadata["width"]

            mapped_labels_ = mapped_labels.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

            # Dense or sparse supervision
            if self.dense_ss:
                # Dense supervision: downsample GT and upsample to the current image size
                downsampled_labels = F.interpolate(
                    mapped_labels_,
                    size=(sem_height // sem_downscale, sem_width // sem_downscale),
                    mode='nearest'
                )
                resized_mapped_labels = F.interpolate(
                    downsampled_labels,
                    size=(height, width),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()  # (H, W)
                valid_semantic_resized = (resized_mapped_labels != -100).float()
            else:
                # Sparse supervision: supervise on specific points after resizing
                resized_mapped_labels = F.interpolate(
                    mapped_labels_,
                    size=(height, width),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()  # (H, W)

                # Generate sparse supervision mask
                mask = torch.zeros((height, width), dtype=torch.float)

                # Calculate indices ensuring edge pixels are included
                indices_h = torch.arange(0, height, sem_downscale)
                indices_w = torch.arange(0, width, sem_downscale)
                grid_h, grid_w = torch.meshgrid(indices_h, indices_w)
                mask[grid_h.flatten(), grid_w.flatten()] = 1.0

                mask = mask * (resized_mapped_labels != -100).float()  # Ensure invalid pixels are not supervised
                valid_semantic_resized = mask  # [H, W]

                #  Set labels to -100 where the mask is 0 (i.e., regions that are not supervised)
                resized_mapped_labels[valid_semantic_resized == 0] = -100

            # Flatten labels and valid mask
            flattened_labels = resized_mapped_labels.view(-1, 1)  # (H*W, 1)
            flattened_valid_mask = valid_semantic_resized.view(-1).float()  # (H*W,)

            # Store the labels and masks
            all_semantics.append(flattened_labels)
            all_valid_semantics.append(flattened_valid_mask)

            if verbose:
                print(f"Semantic {image_id} loaded ({t + 1} / {len(json_files)} )")
                print(
                    f'{flattened_valid_mask.sum() * 100.0 / (height * width):.5f} percent of pixels are valid in semantic map.')

        all_semantics = torch.cat(all_semantics, 0).long()  # (N, 1)
        all_valid_semantics = torch.cat(all_valid_semantics, 0).float()  # (N,)

        return all_semantics, all_valid_semantics

    def init_scaling_params(self):
        print("Could not find a scene.loc file in the JSON directory, creating one...")
        all_json = glob.glob(f"{self.json_dir}/*.json")
        all_rays = []
        for json_p in all_json:
            d = utils.read_dict_from_json(json_p)
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            rpc = utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
            all_rays += [rays]
        all_rays = torch.cat(all_rays, 0)
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = utils.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = utils.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = utils.rpc_scaling_params(all_points[:, 2])
        utils.write_dict_to_json(d, f"{self.json_dir}/scene.loc")
        print("... done !")

    def normalize_rays(self, rays):
        rays[:, 0] -= self.center[0]
        rays[:, 1] -= self.center[1]
        rays[:, 2] -= self.center[2]
        rays[:, 0] /= self.range
        rays[:, 1] /= self.range
        rays[:, 2] /= self.range
        rays[:, 6] /= self.range  # near of the ray
        rays[:, 7] /= self.range  # far of the ray

        return rays

    def prepare_padded_tensor(self, data, valid_mask, height, width, depth=None, interp_mode='nearest'):
        if depth is not None:  # For 2D data (e.g., rays)
            padded_tensor = torch.zeros(height * width, depth)
            padded_tensor[valid_mask, :] = data
            return self.scale_depth(padded_tensor, height, width, depth, interp_mode)
        else:  # For 1D data (e.g., depths, weights, std)
            padded_tensor = torch.zeros(height * width)
            padded_tensor[valid_mask] = data
            return self.scale_depth(padded_tensor, height, width, interp_mode=interp_mode)

    def scale_depth(self, feature, height, width, depth=None, interp_mode='nearest'):
        new_height, new_width = int(height / self.img_downscale), int(width / self.img_downscale)

        if depth is not None:
            feature = feature.reshape(1, 1, height, width, depth)  # 转换为5D张量以适应插值操作
            new_feature = F.interpolate(feature, size=(new_height, new_width, depth), mode=interp_mode)
            return new_feature.squeeze().reshape(new_height * new_width, depth).squeeze()  # 返回2D张量
        else:
            feature = feature.view(1, 1, height, width)  # 转换为4D张量以适应插值操作
            new_feature = F.interpolate(feature, size=(new_height, new_width), mode=interp_mode)
            return new_feature.view(new_height * new_width)  # 返回1D张量

    def get_sun_dirs(self, sun_elevation_deg, sun_azimuth_deg, n_rays):
        """
        Get sun direction vectors
        Args:
            sun_elevation_deg: float, sun elevation in  degrees
            sun_azimuth_deg: float, sun azimuth in degrees
            n_rays: number of rays affected by the same sun direction
        Returns:
            sun_d: (n_rays, 3) 3-valued unit vector encoding the sun direction, repeated n_rays times
        """
        # cnvert degrees to radians
        sun_el = np.radians(sun_elevation_deg)
        sun_az = np.radians(sun_azimuth_deg)

        # Calculate the sun direction vector in Cartesian coordinates
        sun_d = np.array([
            np.sin(sun_az) * np.cos(sun_el),  # x component
            np.cos(sun_az) * np.cos(sun_el),  # y component
            np.sin(sun_el)  # z component
        ])

        sun_dirs = torch.from_numpy(np.tile(sun_d, (n_rays, 1)))
        sun_dirs = sun_dirs.type(torch.FloatTensor)

        return sun_dirs

    def get_latlonalt_from_nerf_prediction(self, rays, depth):
        """
        Compute an image of altitudes from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
        Returns:
            lats: numpy vector of length h*w with the latitudes of the predicted points
            lons: numpy vector of length h*w with the longitude of the predicted points
            alts: numpy vector of length h*w with the altitudes of the predicted points
        """

        # convert inputs to double (avoids loss of resolution later when the tensors are converted to numpy)
        rays = rays.double()
        depth = depth.double()

        # use input rays + predicted sigma to construct a point cloud
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        xyz_n = rays_o + rays_d * depth.view(-1, 1)

        # denormalize prediction to obtain ECEF coordinates
        xyz = xyz_n * self.range
        xyz[:, 0] += self.center[0]
        xyz[:, 1] += self.center[1]
        xyz[:, 2] += self.center[2]

        # convert to lat-lon-alt
        xyz = xyz.data.numpy()
        lats, lons, alts = utils.ecef_to_latlon_custom(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        return lats, lons, alts

    def get_dsm_from_nerf_prediction(self, rays, depth, dsm_path=None, roi_txt=None):
        """
        Compute a DSM from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
            dsm_path (optional): string, path to output DSM, in case you want to write it to disk
            roi_txt (optional): compute the DSM only within the bounds of the region of interest of the txt
        Returns:
            dsm: (h, w) numpy array with the output dsm
        """

        # get point cloud from nerf depth prediction
        lats, lons, alts = self.get_latlonalt_from_nerf_prediction(rays, depth)
        easts, norths = utils.utm_from_latlon(lats, lons)
        cloud = np.vstack([easts, norths, alts]).T

        # (optional) read region of interest, where lidar GT is available
        if roi_txt is not None:
            gt_roi_metadata = np.loadtxt(roi_txt)
            xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
            xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
            resolution = gt_roi_metadata[3]
            yoff += ysize * resolution  # weird but seems necessary ?
        else:
            resolution = 0.5
            xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
            ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
            xoff = np.floor(xmin / resolution) * resolution
            xsize = int(1 + np.floor((xmax - xoff) / resolution))
            yoff = np.ceil(ymax / resolution) * resolution
            ysize = int(1 - np.floor((ymin - yoff) / resolution))

        from plyflatten import plyflatten
        from plyflatten.utils import rasterio_crs, crs_proj
        import utm
        import affine
        import rasterio

        # run plyflatten
        dsm = plyflatten(cloud, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float("inf"))

        n = utm.latlon_to_zone_number(lats[0], lons[0])
        l = utm.latitude_to_zone_letter(lats[0])
        crs_proj = rasterio_crs(crs_proj(f"{n}{l}", crs_type="UTM"))

        # (optional) write dsm to disk
        if dsm_path is not None:
            os.makedirs(os.path.dirname(dsm_path), exist_ok=True)
            profile = {}
            profile["dtype"] = dsm.dtype
            profile["height"] = dsm.shape[0]
            profile["width"] = dsm.shape[1]
            profile["count"] = 1
            profile["driver"] = "GTiff"
            profile["nodata"] = float("nan")
            profile["crs"] = crs_proj
            profile["transform"] = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)
            with rasterio.open(dsm_path, "w", **profile) as f:
                f.write(dsm[:, :, 0], 1)

        return dsm

    def __len__(self):
        # compute length of dataset
        if self.train:
            return self.all_rays.shape[0]
        else:
            return len(self.json_files)

    def __getitem__(self, idx):
        if self.train:
            # rays_ref: depth rays padded to the same size of rgb rays, for debug only (to verify the correspondence)
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "ts": self.all_ids[idx].long(),
                "rays_ref": self.all_deprays[idx],
                "depths": self.all_depths[idx],
                "valid_depth": self.all_valid_depth[idx].long(),
                "depth_std": self.all_depth_stds[idx]
            }

            if self.sem:
                sample["sems"] = self.all_semantics[idx]
                sample["valid_sem"] = self.all_valid_semantics[idx].long()
        else:
            rays, rgbs, _ = self.load_data([self.json_files[idx]])
            ts = self.all_ids[idx] * torch.ones(rays.shape[0], 1)
            d = utils.read_dict_from_json(self.json_files[idx])
            img_id = utils.get_file_id(d["img"])
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)

            sample = {
                "rays": rays,
                "rgbs": rgbs,
                "ts": ts.long(),
                "src_id": img_id,
                "h": h,
                "w": w
            }

            if self.sem:
                sems, _ = self.load_semantic_data(self.sem_dir, [self.json_files[idx]], self.num_sem_classes,
                                                  verbose=False)
                sample["sems"] = sems

        return sample
