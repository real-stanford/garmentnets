# %%
# Change CWD
import os
import sys
project_dir = os.path.expanduser("~/dev/garmentnets")
os.chdir(project_dir)
sys.path.append(project_dir)

# %%
# import
import pathlib

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

import pandas as pd
import numpy as np
import scipy.ndimage as ni
from skimage.measure import marching_cubes
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from datasets.conv_implicit_wnf_dataset import ConvImplicitWNFDataModule
from networks.pointnet2_nocs import PointNet2NOCS
from networks.conv_implicit_wnf import ConvImplicitWNFPipeline
from components.gridding import VirtualGrid, ArraySlicer
from common.torch_util import to_numpy
from common.geometry_util import AABBGripNormalizer


# %%
# helper functions
def get_checkpoint_df(checkpoint_dir):
    all_checkpoint_paths = sorted(pathlib.Path(checkpoint_dir).glob('*.ckpt'))
    rows = list()
    for path in all_checkpoint_paths:
        fname = path.stem
        row = dict()
        for item in fname.split('-'):
            key, value = item.split('=')
            row[key] = float(value)
        row['path'] = str(path.absolute())
        rows.append(row)
    checkpoint_df = pd.DataFrame(rows)
    return checkpoint_df


# %%
# main script
@hydra.main(config_path="../config", 
    config_name="predict_conv_implicit_iccv_pipeline_default")
def main(cfg: DictConfig) -> None:
    # hydra creates working directory automatically
    pred_output_dir = os.getcwd()
    print(pred_output_dir)

    # determine checkpoint
    checkpoint_path = os.path.expanduser(cfg.main.checkpoint_path)
    assert(pathlib.Path(checkpoint_path).exists())

    # load datamodule
    datamodule = ConvImplicitWNFDataModule(**cfg.datamodule)
    datamodule.prepare_data()
    batch_size = datamodule.kwargs['batch_size']
    assert(batch_size == 1)
    # val and test dataloader both uses val_dataset
    val_dataset = datamodule.val_dataset
    # subset = getattr(datamodule, '{}_subset'.format(cfg.prediction.subset))
    dataloader = getattr(datamodule, '{}_dataloader'.format(cfg.prediction.subset))()
    num_samples = len(dataloader)

    # load input zarr
    input_zarr_path = os.path.expanduser(cfg.datamodule.zarr_path)
    input_root = zarr.open(input_zarr_path, 'r')
    input_samples_group = input_root['samples']

    # create output zarr
    output_zarr_path = os.path.join(pred_output_dir, 'prediction.zarr')
    store = zarr.DirectoryStore(output_zarr_path)
    compressor = Blosc(cname='zstd', clevel=6, shuffle=Blosc.BITSHUFFLE)
    output_root = zarr.group(store=store, overwrite=False)
    output_samples_group = output_root.require_group('samples', overwrite=False)

    root_attrs = {
        'subset': cfg.prediction.subset
    }
    output_root.attrs.put(root_attrs)

    # init wandb
    wandb_path = os.path.join(pred_output_dir, 'wandb')
    os.mkdir(wandb_path)
    wandb_run = wandb.init(
        project=os.path.basename(__file__),
        **cfg.logger)
    wandb_meta = {
        'run_name': wandb_run.name,
        'run_id': wandb_run.id
    }
    meta = {
        'script_path': __file__
    }

    # load module to gpu
    model_cpu = ConvImplicitWNFPipeline.load_from_checkpoint(checkpoint_path)
    device = torch.device('cuda:{}'.format(cfg.main.gpu_id))
    model = model_cpu.to(device)
    model.eval()
    model.requires_grad_(False)

    # dump final cfg
    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': pred_output_dir,
        'wandb': wandb_meta,
        'meta': meta
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    wandb.config.update(all_config)

    # loop
    for batch_idx, batch_cpu in enumerate(tqdm(dataloader)):
        # locate raw info
        dataset_idx = int(batch_cpu.dataset_idx[0])
        val_group_row = val_dataset.groups_df.iloc[dataset_idx]
        group_key = val_group_row.group_key
        attr_keys = ['scale', 'gender', 'sample_id', 'garment_name', 'grip_vertex_idx']
        attrs = dict((x, val_group_row[x]) for x in attr_keys)
        int_keys = ['gender', 'grip_vertex_idx']
        for key in int_keys:
            attrs[key] = int(attrs[key])
        attrs['batch_idx'] = batch_idx

        # load input zarr
        input_group = input_samples_group[group_key]

        # create zarr group
        output_group = output_samples_group.require_group(
            group_key, overwrite=False)
        output_group.attrs.put(attrs)

        batch = batch_cpu.to(device=device)
        # stage 1/1.5
        pointnet2_result = model.pointnet2_forward(batch)
        unet3d_result = model.unet3d_forward(pointnet2_result)
        nocs_data = pointnet2_result['nocs_data']

        # stage 2 generate volume
        vg = VirtualGrid(grid_shape=(cfg.prediction.volume_size,)*3)
        grid_points = vg.get_grid_points(include_batch=False)
        array_slicer = ArraySlicer(grid_points.shape, (64,64,64))
        result_volume = torch.zeros(grid_points.shape[:-1], dtype=torch.float32, device=device)

        for i in range(len(array_slicer)):
            slices = array_slicer[i]
            query_points = grid_points[slices]
            query_points_gpu = query_points.to(device).view(1,-1,3)
            decoder_result = model.volume_decoder_forward(unet3d_result, query_points_gpu)
            pred_volume_value = decoder_result['pred_volume_value'].view(*query_points.shape[:-1])
            result_volume[slices] = pred_volume_value
        pred_volume = result_volume
        wnf_volume = to_numpy(pred_volume)

        # stage 2.5 marching cubes
        volume_size = wnf_volume.shape[-1]
        wnf_ggm = ni.gaussian_gradient_magnitude(
            wnf_volume, sigma=cfg.prediction.gradient_sigma, mode="nearest")
        voxel_spacing = 1 / (volume_size - 1)
        mc_verts = np.ones((1,3), dtype=np.float32) * np.nan
        mc_faces = np.zeros((1,3), dtype=np.int64)
        mc_normals =np.ones((1,3), dtype=np.float32) * np.nan
        mc_values = np.ones((1,), dtype=np.float32) * np.nan
        mc_verts_ggm = np.ones((1,), dtype=np.float32) * np.nan
        mc_warp_field = np.ones((1,3), dtype=np.float32) * np.nan
        try:
            mc_verts, mc_faces, mc_normals, mc_values = marching_cubes(
                wnf_volume, 
                level=cfg.prediction.iso_surface_level, 
                spacing=(voxel_spacing,)*3, 
                gradient_direction=cfg.prediction.gradient_direction,
                method='lewiner')
            
            mc_verts_nn_idx = (mc_verts / voxel_spacing).astype(np.uint32)
            mc_verts_ggm = wnf_ggm[
                mc_verts_nn_idx[:,0], mc_verts_nn_idx[:,1], mc_verts_nn_idx[:,2]]
            
            # stage 3
            surface_query_points = torch.from_numpy(mc_verts.astype(np.float32)).view(1,-1,3).to(device)
            surface_decoder_result = model.surface_decoder_forward(
                unet3d_result, surface_query_points)
            mc_warp_field = to_numpy(surface_decoder_result['out_features'].view(-1, 3))
        except ValueError as e:
            pass

        # write data to disk
        mc_data = {
            'verts': mc_verts.astype(np.float32),
            'faces': mc_faces.astype(np.int32),
            'normals': mc_normals.astype(np.float32),
            'volume_value': mc_values.astype(np.float32),
            'volume_gradient_magnitude': mc_verts_ggm.astype(np.float32),
            'warp_field': mc_warp_field.astype(np.float32)
        }

        # stage 3.5 hole prediction
        if cfg.prediction.use_hole_prediction:
            mc_surface_decoder_result = model.mc_surface_decoder_forward(
                unet3d_result, surface_query_points)
            is_on_surface_logits = to_numpy(
                mc_surface_decoder_result['out_features']).squeeze()
            is_on_surface = is_on_surface_logits > 0
            mc_data['is_on_surface'] = is_on_surface
            mc_data['is_on_surface_logits'] = is_on_surface_logits

        output_mc_group = output_group.require_group(
            'marching_cubes_mesh', overwrite=False)
        for key, data in mc_data.items():
            output_mc_group.array(
                name=key, data=data, chunks=data.shape, 
                compressor=compressor, overwrite=True)

        nocs_data = pointnet2_result['nocs_data']
        pred_nocs_logits = pointnet2_result['per_point_logits']
        pc_data_torch = {
            'pred_nocs': nocs_data.pos,
            'pred_nocs_confidence': nocs_data.pred_confidence,
            'pred_nocs_logits': pred_nocs_logits,
            'input_points': batch.pos,
            'input_rgb': (batch.x * 255).to(torch.uint8),
            'gt_nocs': batch.y
        }
        pc_data = dict((x[0], to_numpy(x[1])) for x in pc_data_torch.items())
        output_pc_group = output_group.require_group(
            'point_cloud', overwrite=False)
        for key, data in pc_data.items():
            output_pc_group.array(
                name=key, data=data, chunks=data.shape, 
                compressor=compressor, overwrite=True)
        
        # copy mesh data
        input_gt_mc_group = input_group['marching_cube_mesh']
        zarr.copy(input_gt_mc_group, output_group, 
            name='gt_marching_cubes_mesh', if_exists='replace')
    
        rot_mat = np.squeeze(to_numpy(batch_cpu.input_aug_rot_mat))
        aug_keys = ['cloth_verts']
        input_mesh_group = input_group['mesh']
        output_mesh_group = output_group.require_group('gt_mesh', overwrite=False)
        for key, value in input_mesh_group.arrays():
            data = value[:]
            if key in aug_keys:
                data = data @ rot_mat.T
            output_mesh_group.array(
                name=key, data=data, chunks=data.shape, 
                compressor=compressor, overwrite=True)
        
        # handel grip point predicition
        global_feature = pointnet2_result['global_feature']
        pred_global_logits = pointnet2_result['global_logits']
        pred_global_bins = pred_global_logits.reshape(
            (pred_global_logits.shape[0], 
            pred_global_logits.shape[-1]//3, 
            3))
        grip_bin_idx_pred = torch.argmax(pred_global_bins, dim=1)
        pred_grip_point = vg.idxs_to_points(grip_bin_idx_pred)
        pred_global_bins_confidence = torch.softmax(pred_global_bins, dim=1)

        this_pc_dist_pred = torch.norm(batch.pos, p=None, dim=1)
        this_grip_idx_pred = torch.argmin(this_pc_dist_pred)
        this_grip_nocs_pred = nocs_data.pos[this_grip_idx_pred]

        misc_data = {
            'gt_nocs_grip_point': to_numpy(batch.nocs_grip_point)[0],
            'pred_nocs_grip_point': to_numpy(this_grip_nocs_pred),
            'pred_global_nocs_grip_point': to_numpy(pred_grip_point)[0],
            'pred_global_confidence': to_numpy(pred_global_bins_confidence)[0],
            'global_feature': to_numpy(global_feature)[0]
        }
        output_misc_group = output_group.require_group('misc', overwrite=False)
        for key, data in misc_data.items():
            output_misc_group.array(
                name=key, data=data, chunks=data.shape, 
                compressor=compressor, overwrite=True)

        # logging
        log_data = {
            'prediction_batch_idx': batch_idx
        }
        wandb.log(
            data=log_data,
            step=batch_idx)

# %%
# driver
if __name__ == "__main__":
    main()
