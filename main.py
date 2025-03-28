import datetime
import os
import shutil
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from eval import predefined_val_ts, save_nerf_output_to_images
from models import load_model
from modules import metrics, utils
from modules.opt import Train_parser
from modules.rendering import render_rays


class NeRF_pl(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_sem_classes = args.num_sem_classes
        self.loss = metrics.load_loss(args)

        self.depth = args.depth
        if self.depth:
            self.depth_loss = metrics.DepthLoss(lambda_ds=args.ds_lambda, GNLL=args.GNLL, usealldepth=args.usealldepth,
                                                margin=args.margin, stdscale=args.stdscale)
            self.ds_drop = np.round(args.ds_drop * args.max_train_steps)

        self.sem = args.sem
        if self.sem:
            self.semantic_loss = metrics.SemanticLoss(lambda_ss=args.ss_lambda, )
            self.ss_drop = np.round(args.ss_drop * args.max_train_steps)

        self.define_models()
        self.outdir = os.path.join(args.logs_dir)
        self.val_im_dir = os.path.join(args.logs_dir, "val")
        self.train_im_dir = os.path.join(args.logs_dir, "train")
        self.train_steps = 0

        self.use_ts = False
        if self.args.beta:
            self.loss_without_beta = metrics.SNerfLoss(lambda_sc=args.sc_lambda)
            self.use_ts = True

    def define_models(self):
        self.models = {}
        self.nerf_coarse = load_model(self.args)
        self.models['coarse'] = self.nerf_coarse
        if self.args.n_importance > 0:
            self.nerf_fine = load_model(self.args)
            self.models['fine'] = self.nerf_fine
        if self.args.beta:
            self.embedding_t = torch.nn.Embedding(self.args.t_embbeding_vocab, self.args.t_embbeding_tau)
            self.models["t"] = self.embedding_t

    def forward(self, rays, ts, mode='test', valid_depth=None, target_depths=None, target_std=None, semantics=None):
        chunk_size = self.args.chunk
        batch_size = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, batch_size, chunk_size):
            rendered_ray_chunks = render_rays(
                self.models,
                self.args,
                rays[i:i + chunk_size],
                ts[i:i + chunk_size] if ts is not None else None,
                semantics=semantics[i:i + chunk_size] if semantics is not None else None,
                mode=mode,
                valid_depth=valid_depth,
                target_depths=target_depths,
                target_std=target_std)
            for k, v in rendered_ray_chunks.items():
                if mode != 'train':
                    results[k].append(v.cpu())
                else:
                    results[k].append(v)

                # if mode != 'train':
                #     results[k] += [v.cpu()]
                # else:
                #     results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
            if mode == 'train':
                results[k] = results[k].to(rays.device)
        return results

    def prepare_data(self):
        self.train_dataset = [] + load_dataset(self.args, split="train")
        self.val_dataset = [] + load_dataset(self.args, split="val")

    def configure_optimizers(self):
        parameters = utils.get_parameters(self.models)
        self.optimizer = torch.optim.Adam(parameters, lr=self.args.lr, weight_decay=0)
        max_epochs = self.get_current_epoch(self.args.max_train_steps)
        scheduler = utils.get_scheduler(optimizer=self.optimizer, lr_scheduler='step', num_epochs=max_epochs)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def train_dataloader(self):
        a = DataLoader(self.train_dataset[0],
                       shuffle=True,
                       num_workers=4,
                       batch_size=self.args.batch_size,
                       pin_memory=True)
        loaders = {"color": a}
        return loaders

    def val_dataloader(self):
        b = DataLoader(self.val_dataset[0],
                       shuffle=False,
                       num_workers=4,
                       batch_size=1,  # validate one image (H*W rays) at a time
                       pin_memory=True)
        return b

    def training_step(self, batch, batch_nb):
        self.log("lr", utils.get_learning_rate(self.optimizer))
        self.train_steps += 1

        rays = batch["color"]["rays"]  # (B, 11)
        rgbs = batch["color"]["rgbs"]  # (B, 3)
        ts = batch["color"]["ts"].squeeze() if self.use_ts else None  # (B, 1)
        sems = batch["color"]["sems"].squeeze() if self.sem else None

        valid_depth = None
        depths = None
        target_std = None

        if self.args.model == 'sp-nerf':
            valid_depth = batch["color"]["valid_depth"]  # (B)
            depths = batch["color"]["depths"]  # (B,2)
            target_std = batch["color"]["depth_std"]  # (B)

        results = self(
            rays,
            ts,
            mode='train',
            valid_depth=valid_depth, target_depths=depths, target_std=target_std,
            semantics=sems)

        if 'beta_coarse' in results and self.get_current_epoch(self.train_steps) < 2:
            loss, loss_dict = self.loss_without_beta(results, rgbs)
        else:
            loss, loss_dict = self.loss(results, rgbs)

        self.args.noise_std *= 0.9

        if self.depth:
            kp_depths = depths[:, 0]
            kp_weights = depths[:, 1]
            loss_depth, depth_loss_dict = self.depth_loss(
                results, kp_depths, kp_weights, target_valid_depth=valid_depth, target_std=target_std)

            if self.train_steps < self.ds_drop:
                loss += loss_depth
            loss_dict.update(depth_loss_dict)
            # for k in depth_loss_dict.keys():
            #     loss_dict[k] = depth_loss_dict[k]

        if self.sem:
            sem_loss, sem_loss_dict = self.semantic_loss(results, sems)

            if self.train_steps < self.ss_drop:
                loss += sem_loss
            loss_dict.update(sem_loss_dict)

        self.log("train/loss", loss)
        typ = "fine" if "rgb_fine" in results else "coarse"

        with torch.no_grad():
            psnr_ = metrics.psnr(results[f"rgb_{typ}"], rgbs)
            self.log("train/psnr", psnr_)
        for k in loss_dict.keys():
            self.log(f"train/{k}", loss_dict[k])

        self.log('train_psnr', psnr_, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            rays = batch["rays"].squeeze()  # (H*W, 11)
            rgbs = batch["rgbs"].squeeze()  # (H*W, 3)
            sems = batch["sems"].squeeze() if "sems" in batch else None

            # Get time embedding if needed
            if self.args.model == "sp-nerf":
                t = predefined_val_ts(batch["src_id"][0])
                ts = t * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
            else:
                ts = None

            # Forward pass to get predictions
            results = self(rays, ts, mode='test', semantics=sems)
            for k in results.keys():
                results[k] = results[k].to(rays.device)
            loss, loss_dict = self.loss(results, rgbs)

        self.is_validation_image = True
        if batch_nb == 0:
            self.is_validation_image = False

        # Decide whether to use 'coarse' or 'fine' outputs
        typ = "fine" if "rgb_fine" in results else "coarse"

        # Get image dimensions
        if "h" in batch and "w" in batch:
            W, H = batch["w"], batch["h"]
        else:
            W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))  # assume squared images

        # Prepare RGB images and depth map for output_visual
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        depth = utils.visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)

        # Visualize semantics if available
        if f'sem_logits_{typ}' in results and sems is not None:
            sem_logits = results[f'sem_logits_{typ}']  # (N_rays, num_sem_classes)
            pred_sems = sem_logits.argmax(dim=-1).view(H, W).cpu()  # (H, W)
            gt_sems = sems.view(H, W).cpu()  # (H, W)

            pred_sems_np = pred_sems.numpy()
            gt_sems_np = gt_sems.numpy()

            # Convert semantic labels to color images
            pred_sems_color = utils.convert_semantic_to_color(pred_sems_np, self.num_sem_classes)  # (H, W, 3)
            gt_sems_color = utils.convert_semantic_to_color(gt_sems_np, self.num_sem_classes)  # (H, W, 3)

            # Convert numpy arrays to tensors and permute dimensions
            pred_sems_color = torch.from_numpy(pred_sems_color).permute(2, 0, 1)  # (3, H, W)
            gt_sems_color = torch.from_numpy(gt_sems_color).permute(2, 0, 1)  # (3, H, W)

            # Stack images for output_visual
            stack = torch.stack([img_gt, img, depth, gt_sems_color, pred_sems_color])  # (5, 3, H, W)
        else:
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)

        # Log images to TensorBoard
        split = 'val' if self.is_validation_image else 'train'
        sample_idx = batch_nb - 1 if self.is_validation_image else batch_nb
        self.logger.experiment.add_images(f'{split}_{sample_idx}/GT_pred_depth_sems', stack, self.global_step)

        # Save output for the first training image and all the validation images
        epoch = self.get_current_epoch(self.train_steps)
        save = not bool(epoch % self.args.save_every_n_epochs)
        if save:
            out_dir = self.val_im_dir if self.is_validation_image else self.train_im_dir
            save_nerf_output_to_images(self.val_dataset[0], batch, results, out_dir, epoch, self.num_sem_classes)

        # Compute evaluation metrics
        psnr_ = metrics.psnr(results[f"rgb_{typ}"], rgbs)
        ssim_ = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W), rgbs.view(1, 3, H, W))

        # Compute semantic metrics if semantics are available
        # if f'sem_logits_{typ}' in results:
        #     miou_ = metrics.miou(pred_sems.squeeze(), gt_sems.squeeze(), self.num_sem_classes)
        #     oa_ = metrics.overall_accuracy(pred_sems.squeeze(), gt_sems.squeeze())
        # else:
        #     miou_ = torch.tensor(0.0)
        #     oa_ = torch.tensor(0.0)

        if True:  # compute MAE
            try:
                aoi_id = self.args.aoi_id
                gt_roi_path = os.path.join(self.args.gt_dir, aoi_id + "_DSM.txt")
                gt_dsm_path = os.path.join(self.args.gt_dir, aoi_id + "_DSM.tif")
                assert os.path.exists(gt_roi_path), f"{gt_roi_path} not found"
                assert os.path.exists(gt_dsm_path), f"{gt_dsm_path} not found"
                depth = results[f"depth_{typ}"]
                unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                out_path = os.path.join(self.val_im_dir, f"dsm/tmp_pred_dsm_{unique_identifier}.tif")

                _ = self.val_dataset[0].get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)
                mae_ = utils.compute_mae_and_save_dsm_diff(out_path, batch["src_id"][0], aoi_id, self.args.gt_dir,
                                                           self.val_im_dir, 0, save=False)
                os.remove(out_path)
            except:
                mae_ = np.nan

            self.log("val/loss", loss)
            self.log("val/psnr", psnr_)
            self.log("val/ssim", ssim_)
            self.log("val/mae", mae_)
            # if sems is not None:
            #     self.log("val/mious", miou_)
            #     self.log("val/oa", oa_)
            for k in loss_dict.keys():
                self.log(f"val/{k}", loss_dict[k])

        return {"loss": loss}

    def get_current_epoch(self, tstep):
        return utils.get_epoch_number_from_train_step(tstep, len(self.train_dataset[0]), self.args.batch_size)


def main():
    torch.cuda.empty_cache()
    args = Train_parser()

    system = NeRF_pl(args)

    shutil.copyfile(os.path.join(args.json_dir, "train.txt"), os.path.join(system.outdir, "train.txt"))
    shutil.copyfile(os.path.join(args.json_dir, "test.txt"), os.path.join(system.outdir, "test.txt"))

    logger = pl.loggers.TensorBoardLogger(save_dir=args.logs_dir, name=None, default_hp_metric=True)
    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath=args.ckpts_dir,
                                                 filename="{epoch:d}",
                                                 monitor="val/psnr",
                                                 mode="max",
                                                 save_top_k=-1,
                                                 every_n_val_epochs=args.save_every_n_epochs)

    trainer = pl.Trainer(max_steps=args.max_train_steps,
                         logger=logger,
                         callbacks=[ckpt_callback],
                         resume_from_checkpoint=args.ckpt_path,
                         gpus=[args.gpu_id],
                         auto_select_gpus=False,
                         deterministic=True,
                         benchmark=True,
                         weights_summary=None,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=2,
                         profiler="simple",
                         precision=16,  # Enable 16-bit precision
                         amp_backend='native',  # Use native AMP
                         amp_level='O1'  # Set AMP optimization level
                         )
    trainer.fit(system)


if __name__ == "__main__":
    main()
