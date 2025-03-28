"""
This script defines the input parameters that can be customized from the command line
"""

import argparse
import json
import os
from datetime import datetime


def Train_parser():
    parser = argparse.ArgumentParser()

    # Input and output paths
    parser.add_argument('--project_dir', type=str, required=True,
                        help='path to the project directory')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="pretrained checkpoint path to load")

    # Basic stuff and dataset options
    parser.add_argument('--aoi_id', type=str, required=True,
                        help='aoi_id')
    parser.add_argument("--model", type=str, default='sp-nerf',
                        help="which model to use")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="experiment name")
    parser.add_argument("--gpu_id", type=int, required=True,
                        help="GPU that will be used")

    # Training and network configuration
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate')
    # parser.add_argument('--lr_decay', type=float, default=250e3,
    #                     help='Step size for learning rate decay')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size (number of input rays per iteration)')
    parser.add_argument('--img_downscale', type=float, default=1.0,
                        help='downscale factor for the input images')
    parser.add_argument('--max_train_steps', type=int, default=500000,
                        help='number of training iterations')
    parser.add_argument('--save_every_n_epochs', type=int, default=2,
                        help="save checkpoints and debug files every n epochs")
    parser.add_argument('--fc_units', type=int, default=512,
                        help='number of fully connected units in the main block of layers')
    parser.add_argument('--fc_layers', type=int, default=8,
                        help='number of fully connected layers in the main block of layers')
    parser.add_argument('--n_samples', type=int, default=64,
                        help='number of coarse scale discrete points per input ray')
    parser.add_argument('--n_importance', type=int, default=0,
                        help='number of fine scale discrete points per input ray')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='standard deviation of noise added to sigma to regularize')
    parser.add_argument('--chunk', type=int, default=1024 * 5,
                        help='maximum number of rays that can be processed at once without memory issues')

    # Solar correction
    parser.add_argument('--sc_lambda', type=float, default=0.,
                        help='float that multiplies the solar correction auxiliary loss')

    # Uncertainty aware loss
    parser.add_argument('--beta', action='store_true',
                        help='by default, do not use beta for transient uncertainty')
    parser.add_argument('--first_beta_epoch', type=int, default=2,
                        help='epoch from which transients are estimated')
    parser.add_argument('--t_embbeding_tau', type=int, default=4,
                        help='dimension of the image-dependent embedding')
    parser.add_argument('--t_embbeding_vocab', type=int, default=30,
                        help='number of image-dependent embeddings, it needs to be at least the number of training images')

    # Dense depth supervision
    parser.add_argument('--depth', action='store_true',
                        help='by default, do not use depth supervision loss')
    parser.add_argument('--ds_lambda', type=float, default=0.,
                        help='float that multiplies the depth supervision auxiliary loss')
    parser.add_argument('--ds_drop', type=float, default=0.25,
                        help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--GNLL', action='store_true',
                        help='by default, use MSE depth loss instead of Gaussian negative log likelihood loss')
    parser.add_argument('--usealldepth', action='store_true',
                        help='by default, use only a subset of depth which meets the condition of R_sub in equation 6 in SpS-NeRF article')
    parser.add_argument('--margin', type=float, default=0.0001,
                        help='so that the pts with correlation scores equal to 1 has the std value of margin, instead of 0. (m in equation 5 in SpS-NeRF article)')
    parser.add_argument('--stdscale', type=float, default=1,
                        help='so that the pts with correlation scores close to 0 has the std value of stdscale, instead of 1. (gama in equation 5 in SpS-NeRF article)')

    # Semantic label supervision
    parser.add_argument('--sem', action='store_true',
                        help='Enable semantic loss to guide the model with additional supervision')
    parser.add_argument('--num_sem_classes', type=int, default=5,
                        help='number of semantic classes to use')
    parser.add_argument('--s_embedding_factor', type=int, default=1,
                        help='factor used to embed the input semantic maps')
    parser.add_argument('--sem_downscale', type=float, default=8.0,
                        help='downscale factor for the semantic classification map')
    parser.add_argument('--ignore_label', type=int, default=-100,
                        help='ignore this label')
    parser.add_argument('--dense_ss', action='store_true',
                        help='whether to use dense or sparse labels for SR instead of dense(but coarse) labels')
    parser.add_argument('--ss_lambda', type=float, default=4e-2,
                        help='float that multiplies the semantic supervision auxiliary loss')
    parser.add_argument('--ss_drop', type=float, default=1,
                        help='portion of training steps at which the semantic supervision loss will be dropped')

    # Other strategy
    parser.add_argument('--mapping', action='store_true',
                        help='by default, do not use positional encoding')
    parser.add_argument('--guidedsample', action='store_true',
                        help='by default, do not apply depth-guided sampling')

    args = parser.parse_args()

    # args.dataset_dir = os.path.join(args.project_dir, 'dataset', 'DFC2019_214')
    # args.dataset_dir = os.path.join(args.project_dir, 'dataset', 'DFC2019-3V')
    args.dataset_dir = os.path.join(args.project_dir, 'dataset', 'DFC2019_269')

    # Automatically generate input data paths based on dataset_dir
    args.depth_dir = os.path.join(args.dataset_dir, 'Depth')
    args.json_dir = os.path.join(args.dataset_dir, 'JSON')
    args.img_dir = os.path.join(args.dataset_dir, 'RGB', args.aoi_id)
    args.sem_dir = os.path.join(args.dataset_dir, 'Semantic')
    args.gt_dir = os.path.join(args.dataset_dir, 'Truth')

    args.exp_name = f"{args.exp_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    print(f"\nRunning {args.exp_name} - Using gpu {args.gpu_id}\n")

    args.output_dir = os.path.join(args.project_dir, 'output', args.exp_name)

    # Automatically generate output paths based on output_dir
    args.cache_dir = os.path.join(args.output_dir, 'cache')  # store rays
    args.ckpts_dir = os.path.join(args.output_dir, 'ckpts')
    args.logs_dir = os.path.join(args.output_dir, 'logs')

    os.makedirs(args.logs_dir, exist_ok=True)
    opt_save_path = os.path.join(args.logs_dir, 'opts.json')
    with open(opt_save_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Training parameters have been saved to {opt_save_path}")

    return args


def Test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True,
                        help='path to the project directory')
    parser.add_argument("--exp_name", type=str, default=None,
                        help='exp_name when training SP-NeRF')
    parser.add_argument("--epoch_number", type=int, default=28,
                        help='epoch_number when training SP-NeRF')
    parser.add_argument("--split", type=str, default='val',
                        help='None')
    args = parser.parse_args()

    args.logs_dir = os.path.join(args.project_dir, 'output', args.exp_name, 'logs')
    args.output_dir = os.path.join(args.project_dir, 'output', args.exp_name, 'eval')

    return args


SEMANTIC_CONFIG = {
    3: {
        "color_mapping": {
            0: [0, 255, 0],  # Ground: Green
            1: [255, 0, 0],  # Buildings: Red
            2: [0, 0, 255],  # Water: Blue
        },
        "class_mapping": {
            0: 2,  # Ground
            1: 6,  # Buildings
            2: 9,  # Water
        },
        "semantic_names": {
            0: 'Ground',
            1: 'Buildings',
            2: 'Water',
        },
        "label_mapping": {
            2: 0,  # Ground
            6: 1,  # Buildings
            9: 2,  # Water
        }
    },
    4: {
        "color_mapping": {
            0: [0, 255, 0],  # Ground: Green
            1: [0, 128, 0],  # Trees: Dark Green
            2: [255, 0, 0],  # Buildings: Red
            3: [0, 0, 255],  # Water: Blue
        },
        "class_mapping": {
            0: 2,  # Ground
            1: 5,  # Trees
            2: 6,  # Buildings
            3: 9,  # Water
        },
        "semantic_names": {
            0: 'Ground',
            1: 'Trees',
            2: 'Buildings',
            3: 'Water',
        },
        "label_mapping": {
            2: 0,  # Ground
            5: 1,  # Trees
            6: 2,  # Buildings
            9: 3,  # Water
        }
    },
    5: {
        "color_mapping": {
            0: [0, 255, 0],  # Ground: Green
            1: [0, 128, 0],  # Trees: Dark Green
            2: [255, 0, 0],  # Buildings: Red
            3: [0, 0, 255],  # Water: Blue
            4: [255, 255, 0],  # Bridge / elevated road: Yellow
        },
        "class_mapping": {
            0: 2,  # Ground
            1: 5,  # Trees
            2: 6,  # Buildings
            3: 9,  # Water
            4: 17,  # Bridge / elevated road
        },
        "semantic_names": {
            0: 'Ground',
            1: 'Trees',
            2: 'Buildings',
            3: 'Water',
            4: 'Bridge/Elevated Road',
        },
        "label_mapping": {
            2: 0,  # Ground
            5: 1,  # Trees
            6: 2,  # Buildings
            9: 3,  # Water
            17: 4,  # Bridge / elevated road
        }
    }
}
