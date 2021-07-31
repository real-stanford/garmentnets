import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import wandb
import torch_scatter
from torch_geometric.data import Batch

from components.unet3d import Abstract3DUNet, DoubleConv
from components.mlp import MLP
from networks.pointnet2_nocs import PointNet2NOCS
from components.gridding import batch_to_volume
from common.torch_util import to_numpy
from common.visualization_util import (
    get_vis_idxs, render_nocs_pair, 
    render_wnf_pair, render_wnf_points_pair)
from components.gridding import VirtualGrid

class VolumeFeatureAggregator(pl.LightningModule):
    def __init__(self,
            nn_channels=[1024,1024,128],
            batch_norm=True,
            lower_corner=(0,0,0), 
            upper_corner=(1,1,1), 
            grid_shape=(32, 32, 32),
            reduce_method='mean',
            include_point_feature=True,
            include_confidence_feature=False):
        super().__init__()
        self.save_hyperparameters()
        self.local_nn = MLP(nn_channels, batch_norm=batch_norm)
        self.lower_corner = tuple(lower_corner)
        self.upper_corner = tuple(upper_corner)
        self.grid_shape = tuple(grid_shape)
        self.reduce_method = reduce_method
        self.include_point_feature = include_point_feature
        self.include_confidence_feature = include_confidence_feature
    
    def forward(self, nocs_data):
        local_nn = self.local_nn
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        include_point_feature = self.include_point_feature
        include_confidence_feature = self.include_confidence_feature
        reduce_method = self.reduce_method
        batch_size = nocs_data.num_graphs

        sim_points = nocs_data.sim_points
        points = nocs_data.pos
        nocs_features = nocs_data.x
        batch_idx = nocs_data.batch
        confidence = nocs_data.pred_confidence
        device = points.device
        float_dtype = points.dtype
        int_dtype = torch.int64

        vg = VirtualGrid(
            lower_corner=lower_corner, 
            upper_corner=upper_corner, 
            grid_shape=grid_shape, 
            batch_size=batch_size,
            device=device,
            int_dtype=int_dtype,
            float_dtype=float_dtype)
        
        # get aggregation target index
        points_grid_idxs = vg.get_points_grid_idxs(points, batch_idx=batch_idx)
        flat_idxs = vg.flatten_idxs(points_grid_idxs, keepdim=False)

        # get features
        features_list = [nocs_features]
        if include_point_feature:
            points_grid_points = vg.idxs_to_points(points_grid_idxs)
            local_offset = points - points_grid_points
            features_list.append(local_offset)
            features_list.append(sim_points)
        
        if include_confidence_feature:
            features_list.append(confidence)
        features = torch.cat(features_list, axis=-1)
        
        # per-point transform
        if local_nn is not None:
            features = local_nn(features)

        # scatter
        volume_feature_flat = torch_scatter.scatter(
            src=features.T, index=flat_idxs, dim=-1, 
            dim_size=vg.num_grids, reduce=reduce_method)
        
        # reshape to volume
        feature_size = features.shape[-1]
        volume_feature = volume_feature_flat.reshape(
            (feature_size, batch_size) + grid_shape).permute((1,0,2,3,4))
        return volume_feature


# TODO: UNet 3D
class UNet3D(pl.LightningModule):
    def __init__(self, in_channels, out_channels, f_maps=64, 
            layer_order='gcr', num_groups=8, num_levels=4):
        super().__init__()
        self.save_hyperparameters()
        self.abstract_3d_unet = Abstract3DUNet(
            in_channels=in_channels, out_channels=out_channels,
            final_sigmoid=False, basic_module=DoubleConv, f_maps=f_maps,
            layer_order=layer_order, num_groups=num_groups, 
            num_levels=num_levels, is_segmentation=False)
    
    def forward(self, data):
        result = self.abstract_3d_unet(data)
        return result


# TODO: Deocder Network
class ImplicitWNFDecoder(pl.LightningModule):
    def __init__(self, nn_channels=(128, 512, 512, 1), 
            batch_norm=True):
        super().__init__()
        self.save_hyperparameters()
        self.mlp = MLP(nn_channels, batch_norm=batch_norm)

    def forward(self, features_grid, query_points):
        """
        features_grid: (N,C,D,H,W)
        query_points: (N,M,3)
        """
        # normalize query points to (-1, 1), which is 
        # requried by grid_sample
        query_points_normalized = 2.0 * query_points - 1.0
        # shape (N,C,M,1,1)
        sampled_features = F.grid_sample(
            input=features_grid, 
            grid=query_points_normalized.view(
                *(query_points_normalized.shape[:2] + (1,1,3))), 
            mode='bilinear', padding_mode='border',
            align_corners=True)
        # shape (N,M,C)
        sampled_features = sampled_features.view(
            sampled_features.shape[:3]).permute(0,2,1)
        
        # shape (N,M,C)
        out_features = self.mlp(sampled_features)
        return out_features


class ConvImplicitWNFPipeline(pl.LightningModule):
    def __init__(self, 
        # pointnet params
        pointnet2_params,
        # VolumeFeaturesAggregator params
        volume_agg_params,
        # unet3d params
        unet3d_params,
        # ImplicitWNFDecoder params
        volume_decoder_params,
        surface_decoder_params,
        mc_surface_decoder_params=None,
        # training params
        learning_rate=1e-4,
        loss_type='l2',
        volume_loss_weight=1.0,
        surface_loss_weight=1.0,
        mc_surface_loss_weight=0,
        volume_classification=False,
        volume_task_space=False,
        # vis params
        vis_per_items=0,
        max_vis_per_epoch_train=0,
        max_vis_per_epoch_val=0,
        batch_size=None
        ):
        super().__init__()
        self.save_hyperparameters()

        criterion = None
        if loss_type == 'l2':
            criterion = nn.MSELoss(reduction='mean')
        elif loss_type == 'smooth_l1':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise RuntimeError("Invalid loss_type: {}".format(loss_type))

        self.pointnet2_nocs = PointNet2NOCS(**pointnet2_params)
        self.volume_agg = VolumeFeatureAggregator(**volume_agg_params)
        self.unet_3d = UNet3D(**unet3d_params)
        self.volume_decoder = ImplicitWNFDecoder(**volume_decoder_params)
        self.surface_decoder = ImplicitWNFDecoder(**surface_decoder_params)
        self.mc_surface_decoder = None
        if mc_surface_loss_weight > 0:
            self.mc_surface_decoder = ImplicitWNFDecoder(**mc_surface_decoder_params)
        self.criterion = criterion
        self.binary_criterion = nn.BCEWithLogitsLoss()

        self.volume_loss_weight = volume_loss_weight
        self.surface_loss_weight = surface_loss_weight
        self.mc_surface_loss_weight = mc_surface_loss_weight
        self.volume_classification = volume_classification
        self.volume_task_space = volume_task_space
        self.learning_rate = learning_rate
        self.vis_per_items = vis_per_items
        self.max_vis_per_epoch_train = max_vis_per_epoch_train
        self.max_vis_per_epoch_val = max_vis_per_epoch_val
        self.batch_size = batch_size
        
    # forward function for each stage
    # ===============================
    def pointnet2_forward(self, data):
        self.pointnet2_nocs.eval()
        self.pointnet2_nocs.requires_grad_(False)
        # pointnet2
        pointnet2_result = self.pointnet2_nocs(data)

        # generate prediction
        nocs_bins = self.pointnet2_nocs.nocs_bins
        pred_logits = pointnet2_result['per_point_logits']
        pred_logits_bins = pred_logits.reshape(
            (pred_logits.shape[0], nocs_bins, 3))
        nocs_bin_idx_pred = torch.argmax(pred_logits_bins, dim=1)
        pred_confidence_bins = F.softmax(pred_logits_bins, dim=1)
        pred_confidence = torch.squeeze(torch.gather(
            pred_confidence_bins, dim=1, 
            index=torch.unsqueeze(nocs_bin_idx_pred, dim=1)))

        vg = self.pointnet2_nocs.get_virtual_grid()
        pred_nocs = vg.idxs_to_points(nocs_bin_idx_pred)
        nocs_data = Batch(
            x=pointnet2_result['per_point_features'],
            pos=pred_nocs, 
            batch=pointnet2_result['per_point_batch_idx'],
            sim_points=data.pos,
            pred_confidence=pred_confidence)
        
        pointnet2_result['nocs_data'] = nocs_data
        return pointnet2_result
    
    def unet3d_forward(self, pointnet2_result):
        nocs_data = pointnet2_result['nocs_data']
        # volume agg
        in_feature_volume = self.volume_agg(nocs_data)
        # unet3d
        out_feature_volume = self.unet_3d(in_feature_volume)
        unet3d_result = {
            'out_feature_volume': out_feature_volume
        }
        return unet3d_result

    def volume_decoder_forward(self, unet3d_result, query_points):
        out_feature_volume = unet3d_result['out_feature_volume']
        out_features = self.volume_decoder(out_feature_volume, query_points)
        pred_volume_value = out_features.view(*out_features.shape[:-1])
        decoder_result = {
            'out_features': out_features,
            'pred_volume_value': pred_volume_value
        }
        return decoder_result
    
    def surface_decoder_forward(self, unet3d_result, query_points):
        out_feature_volume = unet3d_result['out_feature_volume']
        out_features = self.surface_decoder(out_feature_volume, query_points)
        decoder_result = {
            'out_features': out_features
        }
        return decoder_result
    
    def mc_surface_decoder_forward(self, unet3d_result, query_points):
        out_feature_volume = unet3d_result['out_feature_volume']
        out_features = self.mc_surface_decoder(out_feature_volume, query_points)
        decoder_result = {
            'out_features': out_features
        }
        return decoder_result
    
    def apply_volume_task_space(self, data, pointnet2_result):
        cloth_sim_aabb = data.cloth_sim_aabb

        nocs_data = pointnet2_result['nocs_data']
        new_nocs_data = copy.copy(nocs_data)
        new_pointnet2_result = copy.copy(pointnet2_result)

        # apply aabb scaling
        scale, offset = self.get_aabb_scale_offset(cloth_sim_aabb)
        # assume the same scaling for now
        scale = scale[0]
        offset = offset[0]
        # replace nocs with normalized sim coordinate
        new_pos = (data.pos * scale) + offset
        new_nocs_data.pos = new_pos
        new_pointnet2_result['nocs_data'] = new_nocs_data
        return new_pointnet2_result

    @staticmethod
    def get_aabb_scale_offset(aabb, padding=0.05):
        nocs_radius = 0.5 - padding
        radius = torch.max(torch.abs(aabb), dim=1)[0][:, :2]
        radius_scale = torch.min(nocs_radius / radius, dim=1)[0]
        nocs_z = nocs_radius * 2
        z_length = aabb[:,1,2] - aabb[:,0,2]
        z_scale = nocs_z / z_length
        scale = torch.minimum(radius_scale, z_scale)

        z_max = aabb[:,1,2] * scale
        offset = torch.ones((len(aabb),3), dtype=aabb.dtype, device=aabb.device) * 0.5
        offset[:,2] = 1-padding-z_max
        return scale, offset

    # forward
    # =======
    def forward(self, data):
        volume_task_space = self.volume_task_space
        pointnet2_result = self.pointnet2_forward(data)

        volume_query_points = data.volume_query_points
        surface_query_points = data.surf_query_points
        if volume_task_space:
            pointnet2_result = self.apply_volume_task_space(
                data, pointnet2_result)
        unet3d_result = self.unet3d_forward(pointnet2_result)
        volume_decoder_result = self.volume_decoder_forward(
            unet3d_result, volume_query_points)
        surface_decoder_result = self.surface_decoder_forward(
            unet3d_result, surface_query_points)
        result = {
            'pointnet2_result': pointnet2_result,
            'unet3d_result': unet3d_result,
            'volume_decoder_result': volume_decoder_result,
            'surface_decoder_result': surface_decoder_result
        }
        if self.mc_surface_decoder is not None:
            mc_surface_query_points = data.mc_surf_query_points
            result['mc_surface_decoder_result'] = self.mc_surface_decoder_forward(
                unet3d_result, mc_surface_query_points)
        return result

    # training
    # ========
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def vis_batch(self, batch, batch_idx, result, is_train=False, img_size=256):
        volume_classification = self.volume_classification
        decoder_result = result['volume_decoder_result']
        pointnet2_result = result['pointnet2_result']

        nocs_data = pointnet2_result['nocs_data']
        pred_nocs = nocs_data.pos
        gt_nocs = batch.y
        batch_idxs = batch.batch
        this_batch_size = batch.num_graphs
        input_pc = batch.pos
        nocs_grip_point_gt = batch.nocs_grip_point
        query_points = batch.volume_query_points
        gt_volume_value = batch.gt_volume_value
        pred_volume_value = decoder_result['pred_volume_value']

        vis_per_items = self.vis_per_items
        batch_size = self.batch_size
        max_vis_per_epoch = None
        prefix = None
        if is_train:
            max_vis_per_epoch = self.max_vis_per_epoch_train
            prefix = 'train_'
        else:
            max_vis_per_epoch = self.max_vis_per_epoch_val
            prefix = 'val_'

        _, selected_idxs, vis_idxs = get_vis_idxs(batch_idx, 
            batch_size=batch_size, this_batch_size=this_batch_size, 
            vis_per_items=vis_per_items, max_vis_per_epoch=max_vis_per_epoch)
        
        log_data = dict()
        for i, vis_idx in zip(selected_idxs, vis_idxs):
            label = prefix + str(vis_idx)
            is_this_item = (batch_idxs == i)
            this_gt_nocs = to_numpy(gt_nocs[is_this_item])
            this_pred_nocs = to_numpy(pred_nocs[is_this_item])

            this_query_points = to_numpy(query_points[i])
            this_gt_volume_value = to_numpy(gt_volume_value[i])
            this_pred_volume_value = to_numpy(pred_volume_value[i])
            if volume_classification:
                this_pred_volume_value = to_numpy(torch.sigmoid(pred_volume_value[i]))

            # point cloud is in gripper frame, therefore gripper is at 0,0,0
            this_pc = input_pc[is_this_item]
            this_pc_dist_pred = torch.norm(this_pc, p=None, dim=1)
            this_grip_idx_pred = torch.argmin(this_pc_dist_pred)
            this_grip_nocs_pred = this_pred_nocs[to_numpy(this_grip_idx_pred)]
            this_grip_nocs_gt = to_numpy(nocs_grip_point_gt[i])

            nocs_img = render_nocs_pair(this_gt_nocs, this_pred_nocs, 
                this_grip_nocs_gt, this_grip_nocs_pred, img_size=img_size)
            wnf_img = render_wnf_points_pair(this_query_points, 
                this_gt_volume_value, this_pred_volume_value, img_size=img_size)
            img = np.concatenate([nocs_img, wnf_img], axis=0)

            log_data[label] = [wandb.Image(img, caption=label)]
        return log_data

    def infer(self, batch, batch_idx, is_train=True):
        volume_loss_weight = self.volume_loss_weight
        surface_loss_weight = self.surface_loss_weight
        mc_surface_loss_weight = self.mc_surface_loss_weight
        volume_classification = self.volume_classification

        result = self(batch)
        volume_decoder_result = result['volume_decoder_result']
        surface_decoder_result = result['surface_decoder_result']
        pred_volume_value = volume_decoder_result['pred_volume_value']
        pred_sim_points = surface_decoder_result['out_features']
        
        gt_volume_value = batch.gt_volume_value
        gt_sim_points = batch.gt_sim_points

        volume_criterion = self.criterion
        surface_criterion = self.criterion
        if volume_classification:
            volume_criterion = self.binary_criterion

        loss_dict = {
            'volume_loss': volume_loss_weight * volume_criterion(pred_volume_value, gt_volume_value),
            'surface_loss': surface_loss_weight * surface_criterion(pred_sim_points, gt_sim_points)
        }
        if mc_surface_loss_weight > 0:
            mc_surface_decoder_result = result['mc_surface_decoder_result']
            pred_is_point_on_surface_logits = mc_surface_decoder_result['out_features']
            gt_is_point_on_surface = batch.is_query_point_on_surf
            loss_dict['mc_surface_loss'] = mc_surface_loss_weight * self.binary_criterion(
                pred_is_point_on_surface_logits, gt_is_point_on_surface)

        metrics = dict(loss_dict)
        metrics['loss'] = sum(loss_dict.values())

        for key, value in metrics.items():
            log_key = ('train_' if is_train else 'val_') + key
            self.log(log_key, value)
        log_data = self.vis_batch(batch, batch_idx, result, is_train=is_train)
        self.logger.log_metrics(log_data, step=self.global_step)
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.infer(batch, batch_idx, is_train=True)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self.infer(batch, batch_idx, is_train=False)
        return metrics['loss']
