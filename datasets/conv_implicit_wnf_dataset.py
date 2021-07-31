from typing import Tuple, Optional
import os
import pathlib
import copy

import igl
import numpy as np
import pandas as pd
import zarr
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, DataLoader
from torch.utils.data import Subset
from scipy.spatial.transform import Rotation
import pytorch_lightning as pl
from torch.utils.data import Subset

from common.cache import file_attr_cache
from components.gridding import nocs_grid_sample
from common.geometry_util import (
    barycentric_interpolation, mesh_sample_barycentric, AABBGripNormalizer)

# helper functions
# ================
def _get_groups_df(samples_group):
    rows = dict()
    for key, group in samples_group.items():
        rows[key] = group.attrs.asdict()
    groups_df = pd.DataFrame(data=list(rows.values()), index=rows.keys())
    groups_df.drop_duplicates(inplace=True)
    groups_df['group_key'] = groups_df.index
    return groups_df

# data sets
# =========
class ConvImplicitWNFDataset(Dataset):
    def __init__(self, 
            # zarr
            zarr_path: str, 
            metadata_cache_dir: str,
            # sample size
            num_pc_sample: int = 6000,
            num_volume_sample: int = 0,
            num_surface_sample: int = 0,
            num_mc_surface_sample: int = 0,
            # mixed sampling config
            surface_sample_ratio: float = 0,
            surface_sample_std: float = 0.05,
            # surface sample noise
            surface_normal_noise_ratio: float = 0,
            surface_normal_std: float = 0,
            # data augumentaiton
            enable_augumentation: bool = True,
            random_rot_range: Tuple[float, float] = (-90, 90),
            num_views: int = 4,
            pc_noise_std: float = 0,
            # volume config
            volume_size: int = 128,
            volume_group: str = 'nocs_winding_number_field',
            tsdf_clip_value: Optional[float] = None,
            volume_absolute_value: bool = False,
            include_volume: bool = False,
            # random seed
            static_epoch_seed: bool = False,
            # catch all
            **kwargs):
        """
        If static_point_sample is True, the points sampled for each index
        will be identical each time being called.
        """
        super().__init__()
        path = pathlib.Path(os.path.expanduser(zarr_path))
        assert(path.exists())
        root = zarr.open(str(path.absolute()), mode='r')
        samples_group = root['samples']

        # extract common info from sample group
        _, sample_group = next(samples_group.groups())
        print(sample_group.tree())
        if volume_size is not None:
            assert(str(volume_size) in 
                sample_group['volume'][volume_group])

        # load group metadata
        groups_df = file_attr_cache(zarr_path, 
            cache_dir=metadata_cache_dir)(_get_groups_df)(samples_group)
        # check if index is sorted
        assert(groups_df.index.is_monotonic_increasing)
        groups_df['idx'] = np.arange(len(groups_df))

        volume_task_space = False
        if volume_group == 'sim_nocs_winding_number_field':
            # don't make mistance twice
            volume_task_space = True
            assert(num_mc_surface_sample == 0)
        
        # global state
        self.samples_group = samples_group
        self.groups_df = groups_df
        # sample size
        self.num_pc_sample = num_pc_sample
        self.num_volume_sample = num_volume_sample
        self.num_surface_sample = num_surface_sample
        self.num_mc_surface_sample = num_mc_surface_sample
        # mixed sampling config
        self.surface_sample_ratio = surface_sample_ratio
        self.surface_sample_std = surface_sample_std
        # surface sample noise
        self.surface_normal_noise_ratio = surface_normal_noise_ratio
        self.surface_normal_std = surface_normal_std
        # data augumentaiton
        self.enable_augumentation = enable_augumentation
        self.random_rot_range = random_rot_range
        self.num_views = num_views
        assert(num_views > 0)
        self.pc_noise_std = pc_noise_std
        # volume config
        self.volume_size = volume_size
        self.volume_group = volume_group
        self.tsdf_clip_value = tsdf_clip_value
        self.volume_absolute_value = volume_absolute_value
        self.include_volume = include_volume
        self.volume_task_space = volume_task_space
        # random seed
        self.static_epoch_seed = static_epoch_seed

        # aabb
        cloth_sim_aabb = root['summary/cloth_aabb_union'][:].astype(np.float32)
        self.cloth_sim_aabb = cloth_sim_aabb

    def __len__(self):
        return len(self.groups_df)

    def data_io(self, idx: int) -> dict:
        dataset_idx = idx
        row = self.groups_df.iloc[dataset_idx]
        group = self.samples_group[row.group_key]
        volume_size = self.volume_size
        volume_group = self.volume_group
        tsdf_clip_value = self.tsdf_clip_value
        volume_absolute_value = self.volume_absolute_value
        num_volume_sample = self.num_volume_sample
        num_mc_surface_sample = self.num_mc_surface_sample

        # io
        attrs = group.attrs.asdict()
        pc_group = group['point_cloud']
        mesh_group = group['mesh']
        data = {
            'cloth_sim_verts': mesh_group['cloth_verts'][:],
            'cloth_nocs_verts': mesh_group['cloth_nocs_verts'][:],
            'cloth_faces_tri': mesh_group['cloth_faces_tri'][:],
            'pc_nocs': pc_group['nocs'][:],
            'pc_sim': pc_group['point'][:],
            'pc_sim_rgb': pc_group['rgb'][:],
            'pc_sizes': pc_group['sizes'][:],
            'scale': attrs['scale'],
            'grip_vertex_idx': attrs['grip_vertex_idx']
        }

        # mc surface io
        if num_mc_surface_sample > 0:
            mc_mesh_group = group['marching_cube_mesh']
            data['marching_cube_verts'] = mc_mesh_group['marching_cube_verts'][:]
            data['marching_cube_faces'] = mc_mesh_group['marching_cube_faces'][:]
            data['is_vertex_on_surface'] = mc_mesh_group['is_vertex_on_surface'][:]

        # volume io
        if num_volume_sample > 0:
            volume_group = group['volume'][volume_group]
            raw_volume = volume_group[str(volume_size)][:]
            volume = np.expand_dims(raw_volume, (0, 1)).astype(np.float32)
            if tsdf_clip_value is not None:
                scaled_volume = volume / tsdf_clip_value
                volume = np.clip(scaled_volume, -1, 1)
            if volume_absolute_value:
                volume = np.abs(volume)
            data['volume'] = volume

        return data
    
    def get_base_data(self, idx:int, data_in: dict) -> dict:
        """
        Get non-volumetric data as numpy arrays
        """
        num_pc_sample = self.num_pc_sample
        static_epoch_seed = self.static_epoch_seed
        num_views = self.num_views
        cloth_sim_aabb = self.cloth_sim_aabb

        seed = idx if static_epoch_seed else None
        rs = np.random.RandomState(seed=seed)
        all_idxs = np.arange(len(data_in['pc_sim']))
        all_num_views = len(data_in['pc_sizes'])
        if num_views < all_num_views:
            idxs_mask = np.zeros_like(all_idxs, dtype=np.bool)
            selected_view_idxs = np.sort(rs.choice(all_num_views, size=num_views, replace=False))
            view_idxs = np.concatenate([[0], np.cumsum(data_in['pc_sizes'])])
            for i in selected_view_idxs:
                idxs_mask[view_idxs[i]: view_idxs[i+1]] = True
            all_idxs = all_idxs[idxs_mask]

        selected_idxs = rs.choice(all_idxs, size=num_pc_sample, replace=False)

        pc_sim_rgb = data_in['pc_sim_rgb'][selected_idxs].astype(np.float32) / 255
        pc_sim = data_in['pc_sim'][selected_idxs].astype(np.float32)
        pc_nocs = data_in['pc_nocs'][selected_idxs].astype(np.float32)
        grip_idx = data_in['grip_vertex_idx']
        sim_grip_point = data_in['cloth_sim_verts'][grip_idx].reshape((1,3))
        nocs_grip_point = data_in['cloth_nocs_verts'][grip_idx].reshape((1,3))

        dists = np.linalg.norm(pc_sim - sim_grip_point[0], axis=1)
        grip_pc_idx = np.array([np.argmin(dists)])
        dataset_idx = np.array([idx])
        scale = np.array([data_in['scale']])
        cloth_sim_aabb = cloth_sim_aabb.reshape((1,)+cloth_sim_aabb.shape)

        data = {
            'x': pc_sim_rgb,
            'y': pc_nocs,
            'pos': pc_sim,
            'scale': scale,
            'sim_grip_point': sim_grip_point,
            'nocs_grip_point': nocs_grip_point,
            'grip_pc_idx': grip_pc_idx,
            'dataset_idx': dataset_idx,
            'cloth_sim_aabb': cloth_sim_aabb
        }
        return data

    def get_volume_sample(self, idx: int, data_in: dict) -> dict:
        """
        Sample points by interpolating the volume.
        """
        volume_group = self.volume_group
        num_volume_sample = self.num_volume_sample
        static_epoch_seed = self.static_epoch_seed
        surface_sample_ratio = self.surface_sample_ratio
        surface_sample_std = self.surface_sample_std
        
        seed = idx if static_epoch_seed else None
        rs = np.random.RandomState(seed=seed)
        query_points = None
        if surface_sample_ratio == 0:
            query_points = rs.uniform(low=0, high=1, size=(num_volume_sample, 3)).astype(np.float32)
        else:
            # combine uniform and near-surface sample
            num_uniform_sample = int(num_volume_sample * surface_sample_ratio)
            num_surface_sample = num_volume_sample - num_uniform_sample
            uniform_query_points = rs.uniform(
                low=0, high=1, size=(num_uniform_sample, 3)).astype(np.float32)
            
            cloth_nocs_verts = data_in['cloth_nocs_verts']
            cloth_faces_tri = data_in['cloth_faces_tri']
            sampled_barycentric, sampled_face_idxs = mesh_sample_barycentric(
                verts=cloth_nocs_verts, faces=cloth_faces_tri, 
                num_samples=num_surface_sample, seed=seed)
            sampled_faces = cloth_faces_tri[sampled_face_idxs]
            sampled_nocs_points = barycentric_interpolation(
                sampled_barycentric, cloth_nocs_verts, sampled_faces)
            surface_noise = rs.normal(loc=(0,)*3, scale=(surface_sample_std,)*3, 
                size=(num_surface_sample, 3))
            surface_query_points = sampled_nocs_points + surface_noise
            mixed_query_points = np.concatenate(
                [uniform_query_points, surface_query_points], axis=0).astype(np.float32)
            query_points = np.clip(mixed_query_points, 0, 1)

        sample_values_torch = nocs_grid_sample(
            torch.from_numpy(data_in['volume']), 
            torch.from_numpy(query_points))
        sample_values_numpy = sample_values_torch.view(
            sample_values_torch.shape[:-1]).numpy()
        if volume_group == 'nocs_occupancy_grid':
            # make sure number is either 0 or 1 for occupancy
            sample_values_numpy = (sample_values_numpy > 0.1).astype(np.float32)
        data = {
            'volume_query_points': query_points,
            'gt_volume_value': sample_values_numpy
        }
        data = self.reshape_for_batching(data)
        return data

    def get_surface_sample(self, idx: int, data_in: dict) -> dict:
        num_surface_sample = self.num_surface_sample
        static_epoch_seed = self.static_epoch_seed
        cloth_sim_aabb = self.cloth_sim_aabb
        volume_task_space = self.volume_task_space
        surface_normal_noise_ratio = self.surface_normal_noise_ratio
        surface_normal_std = self.surface_normal_std

        cloth_nocs_verts = data_in['cloth_nocs_verts']
        cloth_sim_verts = data_in['cloth_sim_verts']
        cloth_faces_tri = data_in['cloth_faces_tri']
        if volume_task_space:
            # flip nocs and sim
            normalizer = AABBGripNormalizer(cloth_sim_aabb)
            old_nocs_verts = cloth_nocs_verts
            cloth_nocs_verts = normalizer(cloth_sim_verts)
            cloth_sim_verts = old_nocs_verts

        seed = idx if static_epoch_seed else None
        sampled_barycentric, sampled_face_idxs = mesh_sample_barycentric(
            verts=cloth_nocs_verts, faces=cloth_faces_tri, 
            num_samples=num_surface_sample, seed=seed)
        sampled_faces = cloth_faces_tri[sampled_face_idxs]

        sampled_nocs_points = barycentric_interpolation(
            sampled_barycentric, cloth_nocs_verts, sampled_faces)
        sampled_sim_points = barycentric_interpolation(
            sampled_barycentric, cloth_sim_verts, sampled_faces)

        if surface_normal_noise_ratio != 0:
            # add noise in normal direction
            num_points_with_noise = int(num_surface_sample * surface_normal_noise_ratio)
            nocs_vert_normals = igl.per_vertex_normals(cloth_nocs_verts, cloth_faces_tri)
            sampled_nocs_normals = barycentric_interpolation(
                sampled_barycentric[:num_points_with_noise], nocs_vert_normals, 
                sampled_faces[:num_points_with_noise])
            rs = np.random.RandomState(seed)
            offset = rs.normal(0, surface_normal_std, size=num_points_with_noise)
            offset_vectors = (sampled_nocs_normals.T * offset).T
            aug_sampled_nocs_points = sampled_nocs_points[:num_points_with_noise] + offset_vectors
            sampled_nocs_points[:num_points_with_noise] = aug_sampled_nocs_points

        # !!!
        # Pytorch Geometric concatinate all elements by dim 0 except
        # attributes with word (face/index), which will be concnatinated by last dimention
        # face is in surface. Use surf instead.
        data = {
            'surf_query_points': sampled_nocs_points,
            'gt_sim_points': sampled_sim_points
        }
        data = self.reshape_for_batching(data)
        return data
    
    def get_mc_surface_sample(self, idx: int, data_in: dict) -> dict:
        num_surface_sample = self.num_surface_sample
        static_epoch_seed = self.static_epoch_seed

        mc_verts = data_in['marching_cube_verts']
        mc_faces = data_in['marching_cube_faces']
        is_mc_vert_on_surface = data_in['is_vertex_on_surface'].astype(np.float32)

        seed = idx if static_epoch_seed else None
        sampled_barycentric, sampled_face_idxs = mesh_sample_barycentric(
            verts=mc_verts, faces=mc_faces, 
            num_samples=num_surface_sample, seed=seed)
        sampled_faces = mc_faces[sampled_face_idxs]

        sampled_nocs_points = barycentric_interpolation(
            sampled_barycentric, mc_verts, sampled_faces)

        is_mc_vert_on_surface_float = np.expand_dims(
            is_mc_vert_on_surface.astype(np.float32), axis=-1)
        is_query_point_on_surface_float = barycentric_interpolation(
            sampled_barycentric, is_mc_vert_on_surface_float, sampled_faces)
        # to be used with BCELoss or BCEWithLogitsLoss, which takes a float 
        # of either 0 or 1 as target
        is_query_point_on_surface = (is_query_point_on_surface_float > 0.5).astype(np.float32)
        # Pytorch Geometric concatinate all elements by dim 0 except
        # attributes with word (face/index), which will be concnatinated by last dimention
        # face is in surface. Use surf instead.
        data = {
            'mc_surf_query_points': sampled_nocs_points,
            'is_query_point_on_surf': is_query_point_on_surface
        }
        data = self.reshape_for_batching(data)
        return data

    def rotation_augumentation(self, idx: int, data: dict) -> dict:
        static_epoch_seed = self.static_epoch_seed
        random_rot_range = self.random_rot_range
        volume_task_space = self.volume_task_space
        assert(len(random_rot_range) == 2)
        assert(random_rot_range[0] <= random_rot_range[-1])

        seed = idx if static_epoch_seed else None
        rs = np.random.RandomState(seed=seed)
        rot_angle = rs.uniform(*random_rot_range)
        rot_mat = Rotation.from_euler(
            'z', rot_angle, degrees=True
            ).as_matrix().astype(np.float32)
    
        if not volume_task_space:
            sim_point_keys = ['pos', 'sim_grip_point', 'gt_sim_points']
            out_data = dict(data)
            for key in sim_point_keys:
                if key in data:
                    out_data[key] = (data[key] @ rot_mat.T).astype(np.float32)
        else:
            out_data = dict(data)
            sim_point_keys = ['pos', 'sim_grip_point']
            for key in sim_point_keys:
                if key in data:
                    out_data[key] = (data[key] @ rot_mat.T).astype(np.float32)
            nocs_sim_point_keys = ['volume_query_points', 'surf_query_points']
            offset_vec = np.array([0.5, 0.5, 0],dtype=np.float32)
            for key in nocs_sim_point_keys:
                if key in data:
                    out_data[key] = (
                        (data[key] - offset_vec) @ rot_mat.T 
                        + offset_vec).astype(np.float32)

        # record augumentation matrix for eval
        out_data['input_aug_rot_mat'] = rot_mat.reshape((1,) + rot_mat.shape)
        return out_data
    
    def noise_augumentation(self, idx: int, data: dict) -> dict:
        pc_noise_std = self.pc_noise_std
        static_epoch_seed = self.static_epoch_seed

        pc_sim = data['pos']
        seed = idx if static_epoch_seed else None
        rs = np.random.RandomState(seed=seed)
        noise = rs.normal(
            loc=(0,)*3,
            scale=(pc_noise_std,)*3,
            size=pc_sim.shape)
        pc_sim_aug = pc_sim + noise
        out_data = dict(data)
        out_data['pos'] = pc_sim_aug
        return out_data
        
    def reshape_for_batching(self, data: dict) -> dict:
        out_data = dict()
        for key, value in data.items():
            out_data[key] = value.reshape((1,) + value.shape)
        return out_data
    
    def __getitem__(self, idx: int) -> Data:
        include_volume = self.include_volume
        num_volume_sample = self.num_volume_sample
        num_surface_sample = self.num_surface_sample
        num_mc_surface_sample = self.num_mc_surface_sample
        enable_augumentation = self.enable_augumentation
        pc_noise_std = self.pc_noise_std

        data_in = self.data_io(idx)
        data = self.get_base_data(idx, data_in=data_in)
        if num_volume_sample > 0:
            volume_sample_data = self.get_volume_sample(idx, data_in=data_in)
            data.update(volume_sample_data)
        if num_surface_sample > 0:
            surface_sample_data = self.get_surface_sample(idx, data_in=data_in)
            data.update(surface_sample_data) 
        if num_mc_surface_sample > 0:
            mc_surface_sample_data = self.get_mc_surface_sample(idx, data_in=data_in)
            data.update(mc_surface_sample_data)
        data['input_aug_rot_mat'] = np.expand_dims(np.eye(3, dtype=np.float32), axis=0)
        if pc_noise_std > 0:
            data = self.noise_augumentation(idx, data=data)
        if enable_augumentation:  
            data = self.rotation_augumentation(idx, data=data)

        if include_volume:
            data['volume'] = data_in['volume']
        
        data_torch = dict(
            (x[0], torch.from_numpy(x[1])) for x in data.items())
        pg_data = Data(**data_torch)
        return pg_data


# data modules
# ============
class ConvImplicitWNFDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        dataset_split: tuple of (train, val, test)
        """
        super().__init__()
        assert(len(kwargs['dataset_split']) == 3)
        self.kwargs = kwargs

        self.train_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        kwargs = self.kwargs
        split_seed = kwargs['split_seed']
        dataset_split = kwargs['dataset_split']

        train_args = dict(kwargs)
        train_args['static_epoch_seed'] = False
        train_dataset = ConvImplicitWNFDataset(**train_args)
        val_dataset = copy.deepcopy(train_dataset)
        val_dataset.static_epoch_seed = True

        groups_df = train_dataset.groups_df
        instances_df = groups_df.groupby('sample_id').agg({'idx': lambda x: sorted(x)})

        # split for train/val/test
        num_instances = len(instances_df)
        normalized_split = np.array(dataset_split)
        normalized_split = normalized_split / np.sum(normalized_split)
        instance_split = (normalized_split * num_instances).astype(np.int64)

        # add leftover instance to training set
        instance_split[0] += num_instances - np.sum(instance_split)

        # generate index for each
        all_idxs = np.arange(num_instances)
        rs = np.random.RandomState(seed=split_seed)
        perm_all_idxs = rs.permutation(all_idxs)

        split_instance_idx_list = list()
        prev_idx = 0
        for x in instance_split:
            next_idx = prev_idx + x
            split_instance_idx_list.append(perm_all_idxs[prev_idx: next_idx])
            prev_idx = next_idx
        assert(np.allclose([len(x) for x in split_instance_idx_list], instance_split))

        split_idx_list = list()
        for instance_idxs in split_instance_idx_list:
            idxs = np.sort(np.concatenate(instances_df.iloc[instance_idxs].idx))
            split_idx_list.append(idxs)
        assert(sum(len(x) for x in split_idx_list) == len(groups_df))

        # generate subsets
        train_idxs, val_idxs, test_idxs = split_idx_list
        train_subset = Subset(train_dataset, train_idxs)
        val_subset = Subset(val_dataset, val_idxs)
        test_subset = Subset(val_dataset, test_idxs)

        self.groups_df = groups_df
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset
    
    def train_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        train_subset = self.train_subset
        dataloader = DataLoader(train_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers)
        return dataloader

    def val_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        val_subset = self.val_subset
        dataloader = DataLoader(val_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers)
        return dataloader

    def test_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        test_subset = self.test_subset
        dataloader = DataLoader(test_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers)
        return dataloader
