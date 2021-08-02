import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
import torch_scatter
from torch.cuda.amp import autocast
import pytorch_lightning as pl
import wandb
import numpy as np

from components.gridding import VirtualGrid
from components.mlp import MLP
from components.pointnet2 import SAModule, GlobalSAModule, FPModule
from components.symmetry import mirror_nocs_points_by_axis
from components.loss import MirrorMSELoss
from common.torch_util import to_numpy
from common.visualization_util import get_vis_idxs, render_nocs_pair, render_confidence_pair

# helper functions
# ================
def local_idx_to_batch_idx(data, batch_idxs, local_idxs):
    batch_global_idxs = torch.arange(
        len(data), dtype=torch.int64, device=data.device)
    global_idxs = list()
    for i in range(len(local_idxs)):
        is_this_batch = (batch_idxs == i)
        this_idxs = batch_global_idxs[is_this_batch]
        global_idx = this_idxs[local_idxs[i]]
        global_idxs.append(global_idx)
    global_idxs_tensor = torch.tensor(
        global_idxs, dtype=local_idxs.dtype, device=local_idxs.device)
    return global_idxs_tensor


def predict_grip_point_nocs(point_cloud, pred_nocs, batch_idxs, batch_size):
    # point cloud is in gripper frame, therefore gripper is at 0,0,0
    batch_global_idxs = torch.arange(len(point_cloud), 
        dtype=torch.int64, device=point_cloud.device)
    pc_dist_to_gripper = torch.norm(point_cloud, p=None, dim=1)
    global_grip_idxs = list()
    for i in range(batch_size):
        is_this_batch = (batch_idxs == i)
        this_dist = pc_dist_to_gripper[is_this_batch]
        this_idxs = batch_global_idxs[is_this_batch]
        local_grip_idx = torch.argmin(this_dist)
        global_grip_idx = this_idxs[local_grip_idx]
        global_grip_idxs.append(global_grip_idx)
    global_grip_idxs_tensor = torch.tensor(
        global_grip_idxs, dtype=torch.int64, device=pred_nocs.device)

    pred_grip_nocs = pred_nocs[global_grip_idxs_tensor]
    return pred_grip_nocs

# modules
# =======
class PointNet2NOCS(pl.LightningModule):
    def __init__(self, 
            # architecture params
            feature_dim, batch_norm, dropout,
            sa1_ratio, sa1_r,
            sa2_ratio, sa2_r,
            fp3_k, fp2_k, fp1_k,
            symmetry_axis=None,
            nocs_bins=None,
            # training params
            learning_rate=1e-4,
            nocs_loss_weight=1,
            grip_point_loss_weight=1,
            # vis params
            vis_per_items=0,
            max_vis_per_epoch_train=0,
            max_vis_per_epoch_val=0,
            batch_size=None
            ):
        super().__init__()
        self.save_hyperparameters()
        self.sa1_module = SAModule(
            sa1_ratio, sa1_r, 
            MLP([3 + 3, 64, 64, 128], batch_norm=batch_norm))
        self.sa2_module = SAModule(
            sa2_ratio, sa2_r, 
            MLP([128 + 3, 128, 128, 256], batch_norm=batch_norm))
        self.sa3_module = GlobalSAModule(
            nn=MLP([256 + 3, 256, 512, 1024], batch_norm=batch_norm))

        self.fp3_module = FPModule(
            k=fp3_k, 
            nn=MLP([1024 + 256, 256, 256], batch_norm=batch_norm))
        self.fp2_module = FPModule(
            k=fp2_k, 
            nn=MLP([256 + 128, 256, 128], batch_norm=batch_norm))
        self.fp1_module = FPModule(
            k=fp1_k, 
            nn=MLP([128 + 3, 128, 128, 128], batch_norm=batch_norm))

        # per-point prediction
        output_dim = 3
        if nocs_bins is not None:
            output_dim = nocs_bins * 3

        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, feature_dim)
        self.lin3 = nn.Linear(feature_dim, output_dim)
        
        self.dp1 = nn.Dropout(p=0.5, inplace=False) if dropout else lambda x: x
        self.dp2 = nn.Dropout(p=0.5, inplace=False) if dropout else lambda x: x

        # global prediction
        self.global_lin1 = nn.Linear(1024, 1024)
        self.global_lin2 = nn.Linear(1024, output_dim)
        self.global_dp1 = nn.Dropout(p=0.5, inplace=False) if dropout else lambda x: x
        self.global_dp2 = nn.Dropout(p=0.5, inplace=False) if dropout else lambda x: x

        criterion = None
        if symmetry_axis is None:
            criterion = nn.MSELoss()
        else:
            criterion = MirrorMSELoss()
        self.criterion = criterion
        self.grip_point_criterion = nn.MSELoss()

        self.nocs_bins = nocs_bins
        self.learning_rate = learning_rate
        self.nocs_loss_weight = nocs_loss_weight
        self.grip_point_loss_weight = grip_point_loss_weight
        self.vis_per_items = vis_per_items
        self.max_vis_per_epoch_train = max_vis_per_epoch_train
        self.max_vis_per_epoch_val = max_vis_per_epoch_val
        self.batch_size = batch_size
        self.symmetry_axis = symmetry_axis

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        # pre-point prediction
        x = F.relu(self.lin1(x))
        x = self.dp1(x)
        x = self.lin2(x)
        features = self.dp2(x)
        logits = self.lin3(features)

        # global prediction
        global_feature, _, _ = sa3_out
        x = F.relu(global_feature)
        x = self.global_dp1(x)
        x = self.global_lin1(x)
        x = self.global_dp2(x)
        global_logits = self.global_lin2(x)

        result = {
            'per_point_features': features,
            'per_point_logits': logits,
            'per_point_batch_idx': data.batch,
            'global_logits': global_logits,
            'global_feature': global_feature
        }
        return result
    
    def logits_to_nocs(self, logits):
        nocs_bins = self.nocs_bins
        if nocs_bins is None:
            # directly regress from nn
            return logits

        # reshape
        logits_bins = None
        if len(logits.shape) == 2:
            logits_bins = logits.reshape((logits.shape[0], nocs_bins, 3))
        elif len(logits.shape) == 1:
            logits_bins = logits.reshape((nocs_bins, 3))

        bin_idx_pred = torch.argmax(logits_bins, dim=1, keepdim=False)

        # turn into per-channel classification problem
        vg = self.get_virtual_grid()
        points_pred = vg.idxs_to_points(bin_idx_pred)
        return points_pred
    
    def get_virtual_grid(self):
        nocs_bins = self.nocs_bins
        device = self.device
        vg = VirtualGrid(lower_corner=(0,0,0), upper_corner=(1,1,1),
            grid_shape=(nocs_bins,)*3, batch_size=1, 
            device=device, int_dtype=torch.int64, 
            float_dtype=torch.float32)
        return vg
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def vis_batch(self, batch, nocs_data, batch_idx, is_train=False):
        pred_nocs = nocs_data.pos
        gt_nocs = batch.y
        batch_idxs = batch.batch
        this_batch_size = batch.num_graphs
        input_pc = batch.pos
        nocs_grip_point_gt = batch.nocs_grip_point
        nocs_grip_point_pred_nn = nocs_data.grip_point

        vis_per_items = self.vis_per_items
        batch_size = self.batch_size
        max_vis_per_epoch = None
        prefix = None
        if vis_per_items <= 0:
            return dict()

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

            # point cloud is in gripper frame, therefore gripper is at 0,0,0
            this_pc = input_pc[is_this_item]
            this_pc_dist_pred = torch.norm(this_pc, p=None, dim=1)
            this_grip_idx_pred = torch.argmin(this_pc_dist_pred)
            this_grip_nocs_pred = this_pred_nocs[to_numpy(this_grip_idx_pred)]
            this_grip_nocs_gt = to_numpy(nocs_grip_point_gt[i])
            this_grip_nocs_pred_nn = to_numpy(nocs_grip_point_pred_nn[i])

            img = render_nocs_pair(this_gt_nocs, this_pred_nocs, 
                this_grip_nocs_gt, this_grip_nocs_pred, this_grip_nocs_pred_nn)
            
            if hasattr(nocs_data, 'pred_confidence'):
                this_pred_confidence = to_numpy(nocs_data.pred_confidence[is_this_item])
                this_x_confidence = this_pred_confidence[:, 0]
                confidence_img = render_confidence_pair(this_gt_nocs, this_pred_nocs, this_x_confidence)
                img = np.concatenate([img, confidence_img], axis=0)

            log_data[label] = [wandb.Image(img, caption=label)]
        return log_data
    
    def get_metrics_regression(self, result, batch):
        criterion = self.criterion
        pred_nocs = result['per_point_logits']
        gt_nocs = batch.y
        nocs_loss = criterion(pred_nocs, gt_nocs)
    
        pred_grip_point = result['global_logits']
        gt_grip_point = batch.nocs_grip_point
        grip_point_loss = criterion(
            pred_grip_point, gt_grip_point)
        
        nocs_data = Batch(x=result['per_point_features'], 
            pos=pred_nocs, grip_point=pred_grip_point, 
            batch=result['per_point_batch_idx'])
        
        loss = self.nocs_loss_weight * nocs_loss \
            + self.grip_point_loss_weight *grip_point_loss

        nocs_err_dist = torch.norm(nocs_data.pos - batch.y, dim=-1).mean()
        grip_err_dist = torch.norm(
            nocs_data.grip_point - batch.nocs_grip_point, dim=-1).mean()
        
        metrics = {
            'loss': loss,
            'nocs_loss': nocs_loss,
            'grip_point_loss': grip_point_loss,
            'nocs_err_dist': nocs_err_dist,
            'grip_point_err_dist': grip_err_dist
        }
        return metrics, nocs_data
    
    def get_metrics_bin_simple(self, result, batch):
        nocs_bins = self.nocs_bins

        criterion = nn.CrossEntropyLoss()
        vg = self.get_virtual_grid()

        pred_logits = result['per_point_logits']
        pred_logits_bins = pred_logits.reshape(
            (pred_logits.shape[0], nocs_bins, 3))
        gt_nocs = batch.y
        gt_nocs_idx = vg.get_points_grid_idxs(gt_nocs)
        nocs_loss = criterion(pred_logits_bins, gt_nocs_idx)

        pred_global_logits = result['global_logits']
        pred_global_bins = pred_global_logits.reshape(
            (pred_global_logits.shape[0], nocs_bins, 3))
        gt_grip_point = batch.nocs_grip_point
        gt_grip_point_idx = vg.get_points_grid_idxs(gt_grip_point)
        grip_point_loss = criterion(
            pred_global_bins, gt_grip_point_idx)
        
        # compute confidence
        nocs_bin_idx_pred = torch.argmax(pred_logits_bins, dim=1)
        pred_confidence_bins = F.softmax(pred_logits_bins, dim=1)
        pred_confidence = torch.squeeze(torch.gather(
            pred_confidence_bins, dim=1, 
            index=torch.unsqueeze(nocs_bin_idx_pred, dim=1)))

        pred_nocs = vg.idxs_to_points(nocs_bin_idx_pred)
        grip_bin_idx_pred = torch.argmax(pred_global_bins, dim=1)
        pred_grip_point = vg.idxs_to_points(grip_bin_idx_pred)

        nocs_data = Batch(x=result['per_point_features'],
            pos=pred_nocs, grip_point=pred_grip_point,
            batch=result['per_point_batch_idx'],
            pred_confidence=pred_confidence)
        
        loss = self.nocs_loss_weight * nocs_loss \
            + self.grip_point_loss_weight *grip_point_loss

        nocs_err_dist = torch.norm(nocs_data.pos - batch.y, dim=-1).mean()
        grip_err_dist = torch.norm(
            nocs_data.grip_point - batch.nocs_grip_point, dim=-1).mean()
        
        metrics = {
            'loss': loss,
            'nocs_loss': nocs_loss,
            'grip_point_loss': grip_point_loss,
            'nocs_err_dist': nocs_err_dist,
            'grip_point_err_dist': grip_err_dist
        }
        return metrics, nocs_data
    
    def get_metrics_bin_symmetry_helper(self, result, batch, mirror_axis=None):
        nocs_bins = self.nocs_bins

        # mirroring
        gt_nocs = batch.y
        gt_grip_point = batch.nocs_grip_point

        if mirror_axis is not None:
            gt_nocs = mirror_nocs_points_by_axis(gt_nocs, axis=mirror_axis)
            gt_grip_point = mirror_nocs_points_by_axis(gt_grip_point, axis=mirror_axis)

        criterion = nn.CrossEntropyLoss()
        vg = self.get_virtual_grid()

        pred_logits = result['per_point_logits']
        pred_logits_bins = pred_logits.reshape(
            (pred_logits.shape[0], nocs_bins, 3))
        gt_nocs_idx = vg.get_points_grid_idxs(gt_nocs)
        nocs_loss = criterion(pred_logits_bins, gt_nocs_idx)

        pred_global_logits = result['global_logits']
        pred_global_bins = pred_global_logits.reshape(
            (pred_global_logits.shape[0], nocs_bins, 3))
        gt_grip_point_idx = vg.get_points_grid_idxs(gt_grip_point)
        grip_point_loss = criterion(
            pred_global_bins, gt_grip_point_idx)
        
        nocs_bin_idx_pred = torch.argmax(pred_logits_bins, dim=1)
        pred_confidence_bins = F.softmax(pred_logits_bins, dim=1)
        pred_confidence = torch.squeeze(torch.gather(
            pred_confidence_bins, dim=1, 
            index=torch.unsqueeze(nocs_bin_idx_pred, dim=1)))
        pred_nocs = vg.idxs_to_points(nocs_bin_idx_pred)
        grip_bin_idx_pred = torch.argmax(pred_global_bins, dim=1)
        pred_grip_point = vg.idxs_to_points(grip_bin_idx_pred)

        nocs_data = Batch(x=result['per_point_features'],
            pos=pred_nocs, grip_point=pred_grip_point,
            batch=result['per_point_batch_idx'],
            pred_confidence=pred_confidence)
        
        loss = self.nocs_loss_weight * nocs_loss \
            + self.grip_point_loss_weight * grip_point_loss

        nocs_err_dist = torch.norm(pred_nocs - gt_nocs, dim=-1).mean()
        grip_err_dist = torch.norm(
            pred_grip_point - gt_grip_point, dim=-1).mean()
        
        metrics = {
            'loss': loss,
            'nocs_loss': nocs_loss,
            'grip_point_loss': grip_point_loss,
            'nocs_err_dist': nocs_err_dist,
            'grip_point_err_dist': grip_err_dist
        }
        return metrics, nocs_data
    
    def get_metrics_bin_symmetry(self, result, batch):
        symmetry_axis = self.symmetry_axis

        normal_metrics, normal_nocs_data = self.get_metrics_bin_symmetry_helper(
            result, batch, mirror_axis=None)
        mirrored_metrics, mirror_nocs_data = self.get_metrics_bin_symmetry_helper(
            result, batch, mirror_axis=symmetry_axis)
        
        normal_loss = normal_metrics['loss']
        mirrored_loss = mirrored_metrics['loss']
        final_loss = torch.min(normal_loss, mirrored_loss)
        final_metrics = None
        final_nocs_data = None
        if normal_loss <= mirrored_loss:
            final_metrics = normal_metrics
            final_nocs_data = normal_nocs_data
        else:
            final_metrics = mirrored_metrics
            final_nocs_data = mirror_nocs_data
        final_metrics['loss'] = final_loss
        return final_metrics, final_nocs_data

    
    def infer(self, batch, batch_idx, is_train=True):
        nocs_bins = self.nocs_bins
        symmetry_axis = self.symmetry_axis
        vis_per_items = self.vis_per_items
        result = self(batch)

        metrics, nocs_data = None, None
        if nocs_bins is None:
            metrics, nocs_data = self.get_metrics_regression(result, batch)
        elif symmetry_axis is None:
            metrics, nocs_data = self.get_metrics_bin_simple(result, batch)
        else:
            metrics, nocs_data = self.get_metrics_bin_symmetry(result, batch)
        for key, value in metrics.items():
            log_key = ('train_' if is_train else 'val_') + key
            self.log(log_key, value)
        if vis_per_items > 0:
            log_data = self.vis_batch(batch, nocs_data, batch_idx, is_train=is_train)
            self.logger.log_metrics(log_data, step=self.global_step)
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.infer(batch, batch_idx, is_train=True)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self.infer(batch, batch_idx, is_train=False)
        return metrics['loss']
