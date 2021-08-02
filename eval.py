# %%
# set numpy threads
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# currently requires custom-built igl-python binding
os.environ["IGL_PARALLEL_FOR_NUM_THREADS"] = "1"
import numpy as np
import igl

# %%
# import
import pathlib
from pprint import pprint
import json

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import zarr
from numcodecs import Blosc
from tqdm import tqdm

import pandas as pd
import numpy as np
import scipy.ndimage as ni
from skimage.measure import marching_cubes
from scipy.spatial import ckdtree
import igl

from common.parallel_util import parallel_map
from common.geometry_util import (
    barycentric_interpolation, mesh_sample_barycentric,
    AABBNormalizer, AABBGripNormalizer)
from common.marching_cubes_util import delete_invalid_verts
from common.potpourri3d_util import geodesic_matrix
from common.rendering_util import get_wnf_cmap


# %%
# helper functions
def write_dict_to_group(data, group, compressor):
    for key, data in data.items():
        if isinstance(data, np.ndarray):
            group.array(
                name=key, data=data, chunks=data.shape, 
                compressor=compressor, overwrite=True)
        else:
            group[key] = data


# %%
# worker functions
def compute_optimal_gradient_treshold(
        sample_key, samples_group, precision_weight=0.85, **kwargs):
    sample_group = samples_group[sample_key]
    # io
    gt_mc_group = sample_group['gt_marching_cubes_mesh']
    # gt_mc_faces = gt_mc_group['marching_cube_faces'][:]
    gt_mc_verts = gt_mc_group['marching_cube_verts'][:]
    gt_mc_is_on_surface = gt_mc_group['is_vertex_on_surface'][:]

    pred_mc_group = sample_group['marching_cubes_mesh']
    # pred_mc_faces = pred_mc_group['faces'][:]
    pred_mc_verts = pred_mc_group['verts'][:]
    pred_mc_gm = pred_mc_group['volume_gradient_magnitude'][:]
    
    gt_verts_tree = ckdtree.cKDTree(gt_mc_verts)
    nn_dist, nn_vert_idx = gt_verts_tree.query(pred_mc_verts, k=1)
    nn_is_on_surface = gt_mc_is_on_surface[nn_vert_idx]

    # decision stump to maximize accuracy
    sorted_idx = np.argsort(pred_mc_gm)
    sorted_nn_is_on_surface = nn_is_on_surface[sorted_idx]
    # if greater than threshold, on surface
    # padded = np.concatenate([[False], sorted_nn_is_on_surface,[False]])
    # true_negative = np.cumsum(~sorted_nn_is_on_surface)
    false_negative = np.cumsum(sorted_nn_is_on_surface)
    true_positive = np.cumsum(sorted_nn_is_on_surface[::-1])[::-1]
    false_positive = np.cumsum(~sorted_nn_is_on_surface[::-1])[::-1]
    # accuracy = (true_negative + true_positive) / len(sorted_nn_is_on_surface)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    # f1_score = 2 * (precision * recall) / (precision + recall)
    # positive_rate = sorted_nn_is_on_surface.sum() / len(sorted_nn_is_on_surface)

    score = precision * precision_weight + recall * (1-precision_weight)
    max_score_threshold = None
    if np.any(np.isfinite(score)):
        max_score_idx = np.argmax(score)
        max_score_threshold = pred_mc_gm[sorted_idx[max_score_idx]]
    else:
        max_score_threshold = pred_mc_gm.min()

    metrics = {
        'optimal_wnf_gradient_threshold': max_score_threshold
    }
    return metrics


def compute_pc_metrics(sample_key, samples_group, nocs_aabb, **kwargs):
    sample_group = samples_group[sample_key]
    # io
    pc_group = sample_group['point_cloud']
    gt_nocs = pc_group['gt_nocs'][:]
    pred_nocs = pc_group['pred_nocs'][:]

    # transform
    normalizer = AABBNormalizer(nocs_aabb)
    gt_nocs = normalizer.inverse(gt_nocs)
    pred_nocs = normalizer.inverse(pred_nocs)

    # compute
    nocs_diff = pred_nocs - gt_nocs
    nocs_error_mean_per_dim = np.mean(np.abs(nocs_diff), axis=0)
    nocs_diff_std_per_dim = np.std(nocs_diff, axis=0)

    mirror_gt_nocs = gt_nocs.copy()
    mirror_gt_nocs[:, 0] = -mirror_gt_nocs[:, 0] 
    mirror_nocs_error = pred_nocs - mirror_gt_nocs
    nocs_error_dist = np.linalg.norm(nocs_diff, axis=1)
    mirror_nocs_error_dist = np.linalg.norm(mirror_nocs_error, axis=1)
    mirror_min_nocs_error_dist = np.minimum(nocs_error_dist, mirror_nocs_error_dist)
    
    metrics = {
        'nocs_pc_error_distance': np.mean(nocs_error_dist),
        'nocs_pc_mirror_error_distance': np.mean(mirror_nocs_error_dist),
        'nocs_pc_min_agg_error_distance': np.mean(mirror_min_nocs_error_dist),
        'nocs_pc_agg_min_error_distance': np.minimum(np.mean(nocs_error_dist), np.mean(mirror_nocs_error_dist))
    }
    axis_order = ['x', 'y', 'z']
    per_dim_features = {
        'nocs_pc_diff_std': nocs_diff_std_per_dim,
        'nocs_pc_error': nocs_error_mean_per_dim,
    }
    for key, value in per_dim_features.items():
        for i in range(3):
            metrics['_'.join([key, axis_order[i]])] = value[i]
    return metrics


def compute_grip_point_metrics(sample_key, samples_group, nocs_aabb, **kwargs):
    sample_group = samples_group[sample_key]
    # io
    misc_group = sample_group['misc']
    gt_nocs_grip_point = misc_group['gt_nocs_grip_point'][:]
    pred_nocs_grip_point = misc_group['pred_nocs_grip_point'][:]
    pred_global_nocs_grip_point = misc_group['pred_global_nocs_grip_point'][:]

    # transform
    normalizer = AABBNormalizer(nocs_aabb)
    gt_nocs_grip_point = normalizer.inverse(gt_nocs_grip_point)
    pred_nocs_grip_point = normalizer.inverse(pred_nocs_grip_point)
    pred_global_nocs_grip_point = normalizer.inverse(pred_global_nocs_grip_point)

    pred_points = {
        'pc':  pred_nocs_grip_point,
        'global': pred_global_nocs_grip_point
    }

    metrics = dict()
    for key, value in pred_points.items():
        pred = value
        mirror = value.copy()
        mirror[0] = -mirror[0]

        pred_error = np.linalg.norm(pred - gt_nocs_grip_point)
        mirror_error = np.linalg.norm(mirror - gt_nocs_grip_point)
        min_error = min(pred_error, mirror_error)
        this_metric = {
            'error_distance': pred_error,
            'mirror_error_distanc': mirror_error,
            'min_error_distanc': min_error
        }
        for _key, _value in this_metric.items():
            metric_key = '_'.join(['grip_point', _key, key])
            metrics[metric_key] = _value
    return metrics


def compute_chamfer(sample_key, samples_group, nocs_aabb, 
        num_points=1e4,
        value_threshold=0.13,
        value_key='marching_cubes_mesh/volume_gradient_magnitude',
        seed=0,
        predict_holes=True,
        volume_task_space=False,
        **kwargs):
    sample_group = samples_group[sample_key]
    # io
    pred_mc_group = sample_group['marching_cubes_mesh']
    pred_mc_verts = pred_mc_group['verts'][:]
    pred_mc_faces = pred_mc_group['faces'][:]
    pred_mc_sim_verts = pred_mc_group['warp_field'][:]

    gt_mesh_group = sample_group['gt_mesh']
    gt_faces = gt_mesh_group['cloth_faces_tri'][:]
    gt_nocs_verts = gt_mesh_group['cloth_nocs_verts'][:]
    gt_sim_verts = gt_mesh_group['cloth_verts'][:]

    gt_mc_group = sample_group['gt_marching_cubes_mesh']
    gt_mc_verts = gt_mc_group['marching_cube_verts'][:]
    gt_mc_faces = gt_mc_group['marching_cube_faces'][:]
    gt_is_vertex_on_surface = gt_mc_group['is_vertex_on_surface'][:]

    if volume_task_space:
        # in task space predition, verts are in sim space
        # and warp field is in nocs space
        pred_mc_sim_verts, pred_mc_verts \
            = pred_mc_verts, pred_mc_sim_verts
    
    # transform
    normalizer = AABBNormalizer(nocs_aabb)
    gt_nocs_verts = normalizer.inverse(gt_nocs_verts)
    pred_mc_verts = normalizer.inverse(pred_mc_verts)
    gt_mc_verts = normalizer.inverse(gt_mc_verts)

    # point sample
    num_samples = int(num_points)
    pred_sample_bc, pred_sample_face_idx = mesh_sample_barycentric(
        pred_mc_verts, pred_mc_faces, 
        num_samples=num_samples, seed=seed)
    pred_sample_nocs_points = barycentric_interpolation(
        pred_sample_bc, 
        pred_mc_verts, 
        pred_mc_faces[pred_sample_face_idx])
    pred_sample_sim_points = barycentric_interpolation(
        pred_sample_bc, 
        pred_mc_sim_verts, 
        pred_mc_faces[pred_sample_face_idx])

    gt_sample_bc, gt_sample_face_idx = mesh_sample_barycentric(
        gt_nocs_verts, gt_faces, 
        num_samples=num_samples, seed=seed)
    gt_sample_nocs_points = barycentric_interpolation(
        gt_sample_bc, 
        gt_nocs_verts, 
        gt_faces[gt_sample_face_idx])
    gt_sample_sim_points = barycentric_interpolation(
        gt_sample_bc, 
        gt_sim_verts, 
        gt_faces[gt_sample_face_idx])
    
    surf_gt_mc_verts, surf_gt_mc_faces = delete_invalid_verts(
        gt_mc_verts, gt_mc_faces, gt_is_vertex_on_surface)
    gt_mc_sample_bc, gt_mc_sample_face_idx = mesh_sample_barycentric(
        surf_gt_mc_verts, surf_gt_mc_faces, 
        num_samples=num_samples, seed=seed)
    gt_sample_mc_points = barycentric_interpolation(
        gt_mc_sample_bc, 
        surf_gt_mc_verts, 
        surf_gt_mc_faces[gt_mc_sample_face_idx])

    # compute chamfer distance
    def get_chamfer(pred_points, gt_points):
        pred_tree = ckdtree.cKDTree(pred_points)
        gt_tree = ckdtree.cKDTree(gt_points)
        forward_distance, forward_nn_idx = gt_tree.query(pred_points, k=1)
        backward_distance, backward_nn_idx = pred_tree.query(gt_points, k=1)
        forward_chamfer = np.mean(forward_distance)
        backward_chamfer = np.mean(backward_distance)
        symmetrical_chamfer = np.mean([forward_chamfer, backward_chamfer])
        result = {
            # 'chamfer_forward': forward_chamfer,
            # 'chamfer_backward': backward_chamfer,
            'chamfer_symmetrical': symmetrical_chamfer
        }
        return result
    
    in_data = {
        'nocs_no_hole': {
            'pred_points': pred_sample_nocs_points,
            'gt_points': gt_sample_nocs_points,
        },
        'sim_no_hole': {
            'pred_points': pred_sample_sim_points,
            'gt_points': gt_sample_sim_points
        },
        'nocs_mc': {
            'pred_points': gt_sample_mc_points,
            'gt_points': gt_sample_nocs_points
        }
    }
    if predict_holes:
        pred_value = sample_group[value_key][:]
        pred_sample_value = np.squeeze(barycentric_interpolation(
            pred_sample_bc, 
            np.expand_dims(pred_value, axis=1), 
            pred_mc_faces[pred_sample_face_idx]))
        is_valid_sample = pred_sample_value > value_threshold
        valid_pred_sample_nocs_points = pred_sample_nocs_points[is_valid_sample]
        valid_pred_sample_sim_points = pred_sample_sim_points[is_valid_sample]
        holes_in_data = {
            'nocs': {
                'pred_points': valid_pred_sample_nocs_points,
                'gt_points': gt_sample_nocs_points,
            },
            'sim': {
                'pred_points': valid_pred_sample_sim_points,
                'gt_points': gt_sample_sim_points
            },
        }
        in_data.update(holes_in_data)
    key_order = ['nocs', 'sim', 'nocs_no_hole', 'sim_no_hole', 'nocs_mc']
    old_in_data = in_data
    in_data = dict([(x, old_in_data[x]) for x in key_order if x in old_in_data])
    
    result = dict()
    for category, kwargs in in_data.items():
        out_data = get_chamfer(**kwargs)
        for key, value in out_data.items():
            result['_'.join([key, category])] = value
    return result


def compute_hybrid_chamfer(
        sample_key,
        samples_group, 
        nocs_aabb, 
        num_points=1e4,
        value_threshold=0.13,
        value_key='marching_cubes_mesh/volume_gradient_magnitude',
        seed=0,
        predict_holes=True,
        volume_task_space=False,
        **kwargs):
    sample_group = samples_group[sample_key]
    # io
    pred_mc_group = sample_group['marching_cubes_mesh']
    pred_mc_verts = pred_mc_group['verts'][:]
    pred_mc_faces = pred_mc_group['faces'][:]
    pred_mc_sim_verts = pred_mc_group['warp_field'][:]

    gt_mesh_group = sample_group['gt_mesh']
    gt_faces = gt_mesh_group['cloth_faces_tri'][:]
    gt_nocs_verts = gt_mesh_group['cloth_nocs_verts'][:]
    gt_sim_verts = gt_mesh_group['cloth_verts'][:]

    if volume_task_space:
        # in task space predition, verts are in sim space
        # and warp field is in nocs space
        pred_mc_sim_verts, pred_mc_verts \
            = pred_mc_verts, pred_mc_sim_verts
    
    # transform
    normalizer = AABBNormalizer(nocs_aabb)
    gt_nocs_verts = normalizer.inverse(gt_nocs_verts)
    pred_mc_verts = normalizer.inverse(pred_mc_verts)

    # point sample
    num_samples = int(num_points)
    pred_sample_bc, pred_sample_face_idx = mesh_sample_barycentric(
        pred_mc_verts, pred_mc_faces, 
        num_samples=num_samples, seed=seed)
    pred_sample_nocs_points = barycentric_interpolation(
        pred_sample_bc, 
        pred_mc_verts, 
        pred_mc_faces[pred_sample_face_idx])
    pred_sample_sim_points = barycentric_interpolation(
        pred_sample_bc, 
        pred_mc_sim_verts, 
        pred_mc_faces[pred_sample_face_idx])

    gt_sample_bc, gt_sample_face_idx = mesh_sample_barycentric(
        gt_nocs_verts, gt_faces, 
        num_samples=num_samples, seed=seed)
    gt_sample_nocs_points = barycentric_interpolation(
        gt_sample_bc, 
        gt_nocs_verts, 
        gt_faces[gt_sample_face_idx])
    gt_sample_sim_points = barycentric_interpolation(
        gt_sample_bc, 
        gt_sim_verts, 
        gt_faces[gt_sample_face_idx])

    # compute chamfer distance
    def get_chamfer(pred_nocs_points, gt_nocs_points, pred_sim_points, gt_sim_points):
        pred_tree = ckdtree.cKDTree(pred_nocs_points)
        gt_tree = ckdtree.cKDTree(gt_nocs_points)
        forward_distance, forward_nn_idx = gt_tree.query(pred_nocs_points, k=1)
        backward_distance, backward_nn_idx = pred_tree.query(gt_nocs_points, k=1)

        # forward (for each pred)
        forward_diff = pred_sim_points - gt_sim_points[forward_nn_idx]
        forward_distance = np.linalg.norm(forward_diff, axis=1)

        # backward (for each gt)
        backward_diff = gt_sim_points - pred_sim_points[backward_nn_idx]
        backward_distance = np.linalg.norm(backward_diff, axis=1)

        forward_chamfer = np.mean(forward_distance)
        backward_chamfer = np.mean(backward_distance)
        symmetrical_chamfer = np.mean([forward_chamfer, backward_chamfer])
        result = {
            'hybrid_chamfer_forward': forward_chamfer,
            'hybrid_chamfer_backward': backward_chamfer,
            'hybrid_chamfer_symmetrical': symmetrical_chamfer
        }
        return result
    
    in_data = {
        'no_hole': {
            'pred_nocs_points': pred_sample_nocs_points,
            'gt_nocs_points': gt_sample_nocs_points,
            'pred_sim_points': pred_sample_sim_points,
            'gt_sim_points': gt_sample_sim_points
        }
    }
    if predict_holes:
        pred_value = sample_group[value_key][:]
        pred_sample_value = np.squeeze(barycentric_interpolation(
            pred_sample_bc, 
            np.expand_dims(pred_value, axis=1), 
            pred_mc_faces[pred_sample_face_idx]))
        is_valid_sample = pred_sample_value > value_threshold
        valid_pred_sample_nocs_points = pred_sample_nocs_points[is_valid_sample]
        valid_pred_sample_sim_points = pred_sample_sim_points[is_valid_sample]

        holes_in_data = {
            'regular': {
                'pred_nocs_points': valid_pred_sample_nocs_points,
                'gt_nocs_points': gt_sample_nocs_points,
                'pred_sim_points': valid_pred_sample_sim_points,
                'gt_sim_points': gt_sample_sim_points
            }
        }
        in_data.update(holes_in_data)
    key_order = ['regular', 'no_hole']
    in_data = dict([(x, in_data[x]) for x in key_order if x in in_data])

    result = dict()
    for category, kwargs in in_data.items():
        mirror_kwargs = dict(kwargs)
        pred_nocs = mirror_kwargs['pred_nocs_points'].copy()
        pred_nocs[:, 0] = -pred_nocs[:, 0]
        mirror_kwargs['pred_nocs_points'] = pred_nocs

        out_data = get_chamfer(**kwargs)
        mirror_out_data = get_chamfer(**mirror_kwargs)
        min_out_data = dict()
        for key in out_data:
            min_out_data[key] = min(out_data[key], mirror_out_data[key])
        this_metrics = {
            'pred': out_data,
            'mirror': mirror_out_data,
            'min': min_out_data
        }
        for aug_key, out_data in this_metrics.items():
            for key, value in out_data.items():
                result['_'.join([key, category, aug_key])] = value
    return result


def compute_hausdorff(
        sample_key,
        samples_group, 
        nocs_aabb, 
        value_threshold=0.13,
        value_key='marching_cubes_mesh/volume_gradient_magnitude',
        predict_holes=True,
        volume_task_space=False,
        **kwargs):
    sample_group = samples_group[sample_key]
    # io
    pred_mc_group = sample_group['marching_cubes_mesh']
    pred_mc_verts = pred_mc_group['verts'][:]
    pred_mc_faces = pred_mc_group['faces'][:]
    pred_mc_sim_verts = pred_mc_group['warp_field'][:]

    gt_mesh_group = sample_group['gt_mesh']
    gt_faces = gt_mesh_group['cloth_faces_tri'][:]
    gt_nocs_verts = gt_mesh_group['cloth_nocs_verts'][:]
    gt_sim_verts = gt_mesh_group['cloth_verts'][:]

    gt_mc_group = sample_group['gt_marching_cubes_mesh']
    gt_mc_verts = gt_mc_group['marching_cube_verts'][:]
    gt_mc_faces = gt_mc_group['marching_cube_faces'][:]
    gt_is_vertex_on_surface = gt_mc_group['is_vertex_on_surface'][:]

    if volume_task_space:
        # in task space predition, verts are in sim space
        # and warp field is in nocs space
        pred_mc_sim_verts, pred_mc_verts \
            = pred_mc_verts, pred_mc_sim_verts
    
    # transform
    normalizer = AABBNormalizer(nocs_aabb)
    gt_nocs_verts = normalizer.inverse(gt_nocs_verts)
    pred_mc_verts = normalizer.inverse(pred_mc_verts)
    gt_mc_verts = normalizer.inverse(gt_mc_verts)

    # predict gt mc mesh
    surf_gt_mc_verts, surf_gt_mc_faces = delete_invalid_verts(
        gt_mc_verts, gt_mc_faces, gt_is_vertex_on_surface)
    adj_mat = igl.adjacency_matrix(surf_gt_mc_faces)
    num_cc, cc_idxs, cc_sizes = igl.connected_components(adj_mat)
    max_cc_idx = np.argmax(cc_sizes)
    is_cc_vert = (cc_idxs == max_cc_idx)
    valid_gt_mc_verts, valid_gt_mc_faces = delete_invalid_verts(
        surf_gt_mc_verts, surf_gt_mc_faces, is_cc_vert)
    valid_gt_mc_verts, _ = delete_invalid_verts(
        surf_gt_mc_verts, surf_gt_mc_faces, is_cc_vert)
    
    in_data = {
        'nocs_no_hole': {
            'va': gt_nocs_verts, 
            'fa': gt_faces, 
            'vb': pred_mc_verts,
            'fb': pred_mc_faces
        },
        'sim_no_hole': {
            'va': gt_sim_verts, 
            'fa': gt_faces, 
            'vb': pred_mc_sim_verts,
            'fb': pred_mc_faces
        },
        'nocs_mc': {
            'va': gt_nocs_verts, 
            'fa': gt_faces, 
            'vb': valid_gt_mc_verts,
            'fb': valid_gt_mc_faces
        }
    }

    if predict_holes:
        pred_value = sample_group[value_key][:]
        # remove invalid faces
        is_surface_mc_vert = pred_value > value_threshold
        surface_pred_nocs_verts, surface_pred_faces = delete_invalid_verts(
            pred_mc_verts, pred_mc_faces, is_surface_mc_vert)
        surface_pred_sim_verts, _ = delete_invalid_verts(
            pred_mc_sim_verts, pred_mc_faces, is_surface_mc_vert)
        
        adj_mat = igl.adjacency_matrix(surface_pred_faces)
        num_cc, cc_idxs, cc_sizes = igl.connected_components(adj_mat)
        max_cc_idx = np.argmax(cc_sizes)
        is_cc_vert = (cc_idxs == max_cc_idx)

        cc_pred_nocs_verts, cc_pred_faces = delete_invalid_verts(
            surface_pred_nocs_verts, surface_pred_faces, is_cc_vert)
        cc_pred_sim_verts, _ = delete_invalid_verts(
            surface_pred_sim_verts, surface_pred_faces, is_cc_vert)
        
        holes_in_data = {
            'nocs': {
                'va': gt_nocs_verts, 
                'fa': gt_faces, 
                'vb': cc_pred_nocs_verts,
                'fb': cc_pred_faces
            },
            'sim': {
                'va': gt_sim_verts, 
                'fa': gt_faces, 
                'vb': cc_pred_sim_verts,
                'fb': cc_pred_faces
            }
        }
        in_data.update(holes_in_data)

    key_order = ['nocs', 'sim', 'nocs_no_hole', 'sim_no_hole', 'nocs_mc']
    old_in_data = in_data
    in_data = dict([(x, old_in_data[x]) for x in key_order if x in old_in_data])
    
    def get_hausdorff(**kwargs):
        value = igl.hausdorff(**kwargs)
        return {
            'hausdorff': value
        }

    result = dict()
    for category, kwargs in in_data.items():
        out_data = get_hausdorff(**kwargs)
        for key, value in out_data.items():
            result['_'.join([key, category])] = value
    return result



def compute_geodesic(
        sample_key,
        samples_group,
        nocs_aabb,
        num_points=100, 
        value_threshold=0.13,
        value_key='marching_cubes_mesh/volume_gradient_magnitude',
        seed=0, 
        predict_holes=True,
        volume_task_space=False,
        **kwargs):
    sample_group = samples_group[sample_key]
    # io
    pred_mc_group = sample_group['marching_cubes_mesh']
    pred_mc_verts = pred_mc_group['verts'][:]
    pred_mc_faces = pred_mc_group['faces'][:]
    pred_mc_sim_verts = pred_mc_group['warp_field'][:]

    gt_mesh_group = sample_group['gt_mesh']
    gt_faces = gt_mesh_group['cloth_faces_tri'][:]
    gt_nocs_verts = gt_mesh_group['cloth_nocs_verts'][:]
    gt_sim_verts = gt_mesh_group['cloth_verts'][:]

    gt_mc_group = sample_group['gt_marching_cubes_mesh']
    gt_mc_verts = gt_mc_group['marching_cube_verts'][:]
    gt_mc_faces = gt_mc_group['marching_cube_faces'][:]
    gt_is_vertex_on_surface = gt_mc_group['is_vertex_on_surface'][:]

    if volume_task_space:
        # in task space predition, verts are in sim space
        # and warp field is in nocs space
        pred_mc_sim_verts, pred_mc_verts \
            = pred_mc_verts, pred_mc_sim_verts
    
    # transform
    normalizer = AABBNormalizer(nocs_aabb)
    gt_nocs_verts = normalizer.inverse(gt_nocs_verts)
    pred_mc_verts = normalizer.inverse(pred_mc_verts)
    gt_mc_verts = normalizer.inverse(gt_mc_verts)

    # predict gt mc mesh
    surf_gt_mc_verts, surf_gt_mc_faces = delete_invalid_verts(
        gt_mc_verts, gt_mc_faces, gt_is_vertex_on_surface)
    adj_mat = igl.adjacency_matrix(surf_gt_mc_faces)
    num_cc, cc_idxs, cc_sizes = igl.connected_components(adj_mat)
    max_cc_idx = np.argmax(cc_sizes)
    is_cc_vert = (cc_idxs == max_cc_idx)
    valid_gt_mc_verts, valid_gt_mc_faces = delete_invalid_verts(
        surf_gt_mc_verts, surf_gt_mc_faces, is_cc_vert)

    # point sample
    rs = np.random.RandomState(seed=seed)
    selected_gt_vert_idxs = rs.choice(
        len(gt_nocs_verts), num_points, replace=False)
    selected_gt_nocs = gt_nocs_verts[selected_gt_vert_idxs]
    
    pred_no_hole_tree = ckdtree.cKDTree(pred_mc_verts)
    nn_dist, nn_idxs = pred_no_hole_tree.query(selected_gt_nocs, k=1)
    selected_pred_no_hole_vert_idxs = nn_idxs

    gt_mc_tree = ckdtree.cKDTree(valid_gt_mc_verts)
    nn_dist, nn_idxs = gt_mc_tree.query(selected_gt_nocs, k=1)
    selected_gt_mc_vert_idxs = nn_idxs


    in_data = {
        'gt_nocs': {
            'verts': gt_nocs_verts,
            'faces': gt_faces,
            'vert_idxs': selected_gt_vert_idxs,
        },
        'gt_sim': {
            'verts': gt_sim_verts,
            'faces': gt_faces,
            'vert_idxs': selected_gt_vert_idxs,
        },
        'pred_nocs_no_hole': {
            'verts': pred_mc_verts,
            'faces': pred_mc_faces,
            'vert_idxs': selected_pred_no_hole_vert_idxs,
        },
        'pred_sim_no_hole': {
            'verts': pred_mc_sim_verts,
            'faces': pred_mc_faces,
            'vert_idxs': selected_pred_no_hole_vert_idxs
        },
        'gt_nocs_mc': {
            'verts': valid_gt_mc_verts,
            'faces': valid_gt_mc_faces,
            'vert_idxs': selected_gt_mc_vert_idxs
        }
    }

    rms_pairs_dict = {
        'geodesic_rms_sim_no_hole': ('pred_sim_no_hole', 'gt_sim'),
        'geodesic_rms_nocs_no_hole': ('pred_nocs_no_hole', 'gt_nocs'),
        'geodesic_rms_nocs_mc': ('gt_nocs_mc', 'gt_nocs')
    }

    if predict_holes:
        pred_value = sample_group[value_key][:]
        # remove invalid faces
        is_surface_mc_vert = pred_value > value_threshold
        surface_pred_nocs_verts, surface_pred_faces = delete_invalid_verts(
            pred_mc_verts, pred_mc_faces, is_surface_mc_vert)
        surface_pred_sim_verts, _ = delete_invalid_verts(
            pred_mc_sim_verts, pred_mc_faces, is_surface_mc_vert)
        
        adj_mat = igl.adjacency_matrix(surface_pred_faces)
        num_cc, cc_idxs, cc_sizes = igl.connected_components(adj_mat)
        max_cc_idx = np.argmax(cc_sizes)
        is_cc_vert = (cc_idxs == max_cc_idx)

        cc_pred_nocs_verts, cc_pred_faces = delete_invalid_verts(
            surface_pred_nocs_verts, surface_pred_faces, is_cc_vert)
        cc_pred_sim_verts, _ = delete_invalid_verts(
            surface_pred_sim_verts, surface_pred_faces, is_cc_vert)

        valid_pred_nocs_verts = cc_pred_nocs_verts
        valid_pred_sim_verts = cc_pred_sim_verts
        valid_pred_faces = cc_pred_faces

        pred_tree = ckdtree.cKDTree(valid_pred_nocs_verts)
        nn_dist, nn_idxs = pred_tree.query(selected_gt_nocs, k=1)
        selected_pred_vert_idxs = nn_idxs

        holes_in_data = {
            'pred_nocs': {
                'verts': valid_pred_nocs_verts,
                'faces': valid_pred_faces,
                'vert_idxs': selected_pred_vert_idxs,
            },
            'pred_sim': {
                'verts': valid_pred_sim_verts,
                'faces': valid_pred_faces,
                'vert_idxs': selected_pred_vert_idxs
            },
        }
        holes_rms_pairs_dict = {
            'geodesic_rms_sim': ('pred_sim', 'gt_sim'),
            'geodesic_rms_nocs': ('pred_nocs', 'gt_nocs'),
        }
        in_data.update(holes_in_data)
        rms_pairs_dict.update(holes_rms_pairs_dict)

    key_order = ['gt_nocs', 'gt_sim', 'pred_nocs', 'pred_sim', 'pred_nocs_no_hole', 'pred_sim_no_hole', 'gt_nocs_mc']
    in_data = dict([(x, in_data[x]) for x in key_order if x in in_data])

    key_order = ['geodesic_rms_sim', 'geodesic_rms_nocs', 'geodesic_rms_sim_no_hole', 'geodesic_rms_nocs_no_hole', 'geodesic_rms_nocs_mc']
    rms_pairs_dict = dict([(x, rms_pairs_dict[x]) for x in key_order if x in rms_pairs_dict])

    out_data = dict()
    for key, args in in_data.items():
        out_data[key] = geodesic_matrix(**args)

    def get_rms(mat0, mat1):
        diff = mat0 - mat1
        rms = np.mean(np.abs(diff))
        return rms

    result = dict()
    for key, args in rms_pairs_dict.items():
        result[key] = get_rms(*[out_data[x] for x in args])
    return result


# %%
# visualization functions
def get_task_mesh_vis(
        sample_key, 
        samples_group, 
        value_threshold=0.13,
        value_key='marching_cubes_mesh/volume_gradient_magnitude',
        offset=(0.6,0,0),
        predict_holes=True,
        volume_task_space=False,
        **kwargs):
    """
    Visualizes task space result as a point cloud
    Order:  GT sim mesh Pred sim mesh Sim point cloud
    """
    sample_group = samples_group[sample_key]
    # io
    pred_mc_group = sample_group['marching_cubes_mesh']
    pred_mc_verts = pred_mc_group['verts'][:]
    pred_mc_sim_verts = pred_mc_group['warp_field'][:]
    
    gt_mesh_group = sample_group['gt_mesh']
    gt_nocs_verts = gt_mesh_group['cloth_nocs_verts'][:]
    gt_sim_verts = gt_mesh_group['cloth_verts'][:]

    pc_group = sample_group['point_cloud']
    gt_input_pc = pc_group['input_points'][:]
    gt_input_rgb = pc_group['input_rgb'][:].astype(np.float32)

    if volume_task_space:
        # in task space predition, verts are in sim space
        # and warp field is in nocs space
        pred_mc_sim_verts, pred_mc_verts \
            = pred_mc_verts, pred_mc_sim_verts

    # filter
    if predict_holes:
        pred_value = sample_group[value_key][:]
        is_valid_mc_vert = pred_value > value_threshold
        valid_mc_nocs = pred_mc_verts[is_valid_mc_vert]
        valid_mc_sim = pred_mc_sim_verts[is_valid_mc_vert]
    else:
        valid_mc_nocs = pred_mc_verts
        valid_mc_sim = pred_mc_sim_verts

    # vis
    offset_vec = np.array(offset)
    gt_sim_pc = np.concatenate([gt_sim_verts - offset_vec, gt_nocs_verts * 255], axis=1)
    pred_sim_pc = np.concatenate([valid_mc_sim, valid_mc_nocs * 255], axis=1)
    gt_rgb_pc = np.concatenate([gt_input_pc + offset_vec, gt_input_rgb], axis=1)
    all_pc = np.concatenate([gt_sim_pc, pred_sim_pc, gt_rgb_pc], axis=0).astype(np.float32)
    vis_obj = wandb.Object3D(all_pc)
    return vis_obj

def get_nocs_mesh_vis(
        sample_key,
        samples_group, 
        value_threshold=0.13,
        value_key='marching_cubes_mesh/volume_gradient_magnitude',
        offset=[0.5,0,0],
        value_delta=0.1,
        predict_holes=True,
        volume_task_space=False,
        **kwargs):
    """
    Visualizes nocs space result as a point cloud
    Order:  GT nocs mesh    Pred nocs mesh (colored)
    """
    sample_group = samples_group[sample_key]
    # io
    pred_mc_group = sample_group['marching_cubes_mesh']
    pred_mc_verts = pred_mc_group['verts'][:]

    gt_mesh_group = sample_group['gt_mesh']
    gt_nocs_verts = gt_mesh_group['cloth_nocs_verts'][:]

    if volume_task_space:
        pred_mc_verts = pred_mc_group['warp_field'][:]

    # vis
    offset_vec = np.array(offset)
    gt_nocs_pc = np.concatenate([gt_nocs_verts - offset_vec, gt_nocs_verts * 255], axis=1)
    if predict_holes:
        pred_value = sample_group[value_key][:]
        cmap = get_wnf_cmap(
            min_value=value_threshold-value_delta, 
            max_value=value_threshold+value_delta)
        pred_colors = cmap(pred_value)[:,:3]
    else:
        pred_colors = np.ones((len(pred_mc_verts), 3), dtype=np.float32)
    pred_nocs_pc = np.concatenate([pred_mc_verts + offset_vec, pred_colors * 255], axis=1)
    all_pc = np.concatenate([gt_nocs_pc, pred_nocs_pc], axis=0).astype(np.float32)
    vis_obj = wandb.Object3D(all_pc)
    return vis_obj

def get_nocs_pc_vis(
        sample_key, 
        samples_group, 
        offset=[1.0,0,0], **kwargs):
    """
    GT nocs pc Pred nocs pc (colored with gt nocs)
    """
    sample_group = samples_group[sample_key]
    # io
    pc_group = sample_group['point_cloud']
    gt_nocs_pc = pc_group['gt_nocs'][:]
    pred_nocs_pc = pc_group['pred_nocs'][:]
    pred_nocs_confidence = pc_group['pred_nocs_confidence'][:]

    # vis
    offset_vec = np.array(offset)
    gt_nocs_vis = np.concatenate([gt_nocs_pc - offset_vec, gt_nocs_pc * 255], axis=1)
    pred_nocs_vis = np.concatenate([pred_nocs_pc, gt_nocs_pc * 255], axis=1)
    pred_confidence_vis = np.concatenate([pred_nocs_pc + offset_vec, pred_nocs_confidence * 255], axis=1)
    all_pc = np.concatenate([gt_nocs_vis, pred_nocs_vis, pred_confidence_vis])
    vis_obj = wandb.Object3D(all_pc)
    return vis_obj


# %%
# main script
@hydra.main(config_path="config", 
    config_name="eval_default.yaml")
def main(cfg: DictConfig) -> None:
    # load datase
    pred_output_dir = os.path.expanduser(cfg.main.prediction_output_dir)
    pred_config_path = os.path.join(pred_output_dir, 'config.yaml')
    pred_config_all = OmegaConf.load(pred_config_path)

    # setup wandb
    output_dir = os.getcwd()
    # output_dir = "/home/cchi/dev/cloth_tracking/data/eval_output_test"
    print(output_dir)

    wandb_path = os.path.join(output_dir, 'wandb')
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
    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'prediction_config': OmegaConf.to_container(pred_config_all, resolve=True),
        'output_dir': output_dir,
        'wandb': wandb_meta,
        'meta': meta
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    wandb.config.update(all_config)

    # setup zarr
    pred_zarr_path = os.path.join(pred_output_dir, 'prediction.zarr')
    pred_root = zarr.open(pred_zarr_path, 'r+')
    samples_group = pred_root['samples']
    summary_group = pred_root.require_group('summary', overwrite=False)
    compressor = Blosc(cname='zstd', clevel=6, shuffle=Blosc.BITSHUFFLE)

    sample_key, sample_group = next(iter(samples_group.groups()))
    print(sample_group.tree())
    all_sample_keys = list()
    all_sample_groups = list()
    for sample_key, sample_group in samples_group.groups():
        all_sample_keys.append(sample_key)
        all_sample_groups.append(sample_group)

    global_metrics_group = summary_group.require_group('metrics', overwrite=False)
    global_per_sample_group = global_metrics_group.require_group('per_sample', overwrite=False)
    global_agg_group = global_metrics_group.require_group('aggregate', overwrite=False)

    # write instance order
    sample_keys_arr = np.array(all_sample_keys)
    global_per_sample_group.array('sample_keys', sample_keys_arr, 
        chunks=sample_keys_arr.shape, compressor=compressor, overwrite=True)

    # load aabb
    input_zarr_path = os.path.expanduser(
        pred_config_all.config.datamodule.zarr_path)
    input_root = zarr.open(input_zarr_path, 'r')
    input_samples_group = input_root['samples']
    input_summary_group = input_root['summary']
    nocs_aabb = input_summary_group['cloth_canonical_aabb_union'][:]
    sim_aabb = input_summary_group['cloth_aabb_union'][:]

    # determine null groups
    def is_null(sample_key, samples_group, null_key='marching_cubes_mesh/verts'):
        sample_group = samples_group[sample_key]
        if null_key not in sample_group:
            return True
        arr = sample_group[null_key][:]
        if len(arr) == 0:
            return True
        if np.isnan(arr.flatten()[0]):
            return True
        return False
    null_key='marching_cubes_mesh/volume_gradient_magnitude'
    num_workers = cfg.main.num_workers
    sample_keys_series = pd.Series(all_sample_keys)
    result_df = parallel_map(
            lambda x: is_null(
                x, samples_group,
                null_key=null_key),
            sample_keys_series,
            num_workers=num_workers,
            preserve_index=True)
    is_sample_null = result_df.result
    not_null_sample_keys_series = sample_keys_series.loc[~is_sample_null]

    # compute metrics
    metric_func_dict = {
        'compute_optimal_gradient_treshold': compute_optimal_gradient_treshold,
        'compute_pc_metrics': compute_pc_metrics,
        'compute_grip_point_metrics': compute_grip_point_metrics,
        'compute_chamfer': compute_chamfer,
        'compute_hybrid_chamfer': compute_hybrid_chamfer,
        'compute_geodesic': compute_geodesic,
        'compute_hausdorff': compute_hausdorff
    }
    no_override_keys = ['compute_optimal_gradient_treshold', 'compute_pc_metrics']
    derefrence_keys = ['value_threshold']

    cfg_override_all = OmegaConf.to_container(cfg.override_all, resolve=True)
    num_workers = cfg.main.num_workers
    all_metrics = dict()
    for func_key, func in metric_func_dict.items():
        print("Running {}".format(func_key))
        metric_args = OmegaConf.to_container(cfg.eval[func_key], resolve=True)
        if not metric_args['enabled']:
            print("Disabled, skipping")
            continue
        if func_key not in no_override_keys:
            for key, value in cfg_override_all.items():
                if key in derefrence_keys:
                    if isinstance(value, str):
                        value = float(np.array(pred_root[value]))
                metric_args[key] = value
        print("Config:")
        pprint(metric_args)
        result_df = parallel_map(
            lambda x: func(
                sample_key=x, 
                samples_group=samples_group, 
                input_samples_group=input_samples_group,
                nocs_aabb=nocs_aabb,
                sim_aabb=sim_aabb,
                **metric_args),
            not_null_sample_keys_series,
            num_workers=num_workers,
            preserve_index=True)
        # print error
        errors_series = result_df.loc[result_df.error.notnull()].error
        if len(errors_series) > 0:
            print("Errors:")
            print(errors_series)

        result_dict = dict()
        for key in sample_keys_series.index:
            data = dict()
            if key in result_df.index:
                value = result_df.result.loc[key]
                if value is not None:
                    data = value
            result_dict[key] = data
        this_metric_df = pd.DataFrame(
            list(result_dict.values()),
            index=sample_keys_series.index)

        for column in this_metric_df:
            all_metrics[column] = this_metric_df[column]
            value = np.array(this_metric_df[column])
            global_per_sample_group.array(
                name=column, data=value, chunks=value.shape, 
                compressor=compressor, overwrite=True)
            value_agg = np.nanmean(value)
            global_agg_group[column] = value_agg

    # import pickle
    # pickle.dump(all_metrics, open('/home/cchi/dev/cloth_tracking/data/prediction_output_test/all_metrics.pk', 'wb'))

    all_metrics_df = pd.DataFrame(
        all_metrics, 
        index=sample_keys_series.index)
    all_metrics_df['null_percentage'] = is_sample_null.astype(np.float32)

    all_metrics_agg = all_metrics_df.mean()
    print(all_metrics_agg)
    # save metric to disk
    all_metrics_path = os.path.join(output_dir, 'all_metrics.csv')
    agg_path = os.path.join(output_dir, 'all_metrics_agg.csv')
    summary_path = os.path.join(output_dir, 'summary.json')
    all_metrics_df.to_csv(all_metrics_path)
    all_metrics_df.describe().to_csv(agg_path)
    json.dump(dict(all_metrics_agg), open(summary_path, 'w'), indent=2)

    if cfg.vis.samples_per_instance <= 0:
        print("Done!")
        return

    # visualization
    # pick best and worst
    rank_column = all_metrics_df[cfg.vis.rank_metric]
    sorted_rank_column = rank_column.sort_values()
    best_idxs = sorted_rank_column.index[:cfg.vis.num_best]
    worst_idxs = sorted_rank_column.index[-cfg.vis.num_best:][::-1]
    vis_idxs = np.arange(cfg.vis.num_normal) * cfg.vis.samples_per_instance

    vis_idx_dict = dict()
    for i, idx in enumerate(vis_idxs):
        vis_idx_dict[idx] = "regular_{0:02d}".format(i)
    for i, idx in enumerate(best_idxs):
        vis_idx_dict[idx] = "best_{0:02d}".format(i)
    for i, idx in enumerate(worst_idxs):
        vis_idx_dict[idx] = "worst_{0:02d}".format(i)
    
    vis_func_dict = {
        'task_mesh_vis': get_task_mesh_vis,
        'nocs_mesh_vis': get_nocs_mesh_vis,
        'nocs_pc_vis': get_nocs_pc_vis
    }
    no_override_keys = list()
    # all_log_data = list()
    print("Logging visualization to wandb")
    for i in tqdm(range(len(all_metrics_df))):
        log_data = dict(all_metrics_df.loc[i])
        if i in vis_idx_dict:
            vis_key = vis_idx_dict[i]
            for func_key, func in vis_func_dict.items():
                metric_args = OmegaConf.to_container(cfg.vis[func_key], resolve=True)
                if func_key not in no_override_keys:
                    for key, value in cfg_override_all.items():
                        if key in derefrence_keys:
                            if isinstance(value, str):
                                value = float(np.array(pred_root[value]))
                        metric_args[key] = value
                sample_key = sample_keys_series.loc[i]
                vis_obj = func(sample_key, samples_group,
                    nocs_aabb=nocs_aabb,
                    sim_aabb=sim_aabb,
                    **metric_args)
                vis_name = '_'.join([func_key, vis_key])
                log_data[vis_name] = vis_obj
        # all_log_data.append(log_data)
        wandb_run.log(log_data, step=i)

    print("Logging summary to wandb")
    for key, value in tqdm(all_metrics_agg.items()):
        wandb_run.summary[key] = value
    print("Done!")

# %%
# driver
if __name__ == "__main__":
    main()
