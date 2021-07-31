import numpy as np
import scipy.ndimage as ni
from skimage.measure import marching_cubes_lewiner

def wnf_to_mesh(wnf_volume, iso_surface_level=0.5, gradient_threshold=0.25, sigma=0.5):
    volume_size = wnf_volume.shape[-1]
    wnf_ggm = ni.gaussian_gradient_magnitude(
        wnf_volume, sigma=0.5, mode="nearest")
    voxel_spacing = 1 / (volume_size - 1)
    mc_verts, mc_faces, mc_normals, mc_values = marching_cubes_lewiner(
        wnf_volume, 
        level=iso_surface_level, 
        spacing=(voxel_spacing,)*3, 
        gradient_direction='ascent')
    
    mc_verts_nn_idx = (mc_verts / voxel_spacing).astype(np.uint32)
    mc_verts_ggm = wnf_ggm[
        mc_verts_nn_idx[:,0], mc_verts_nn_idx[:,1], mc_verts_nn_idx[:,2]]
    is_vert_on_surface = mc_verts_ggm > gradient_threshold

    is_face_valid = np.ones(len(mc_faces), dtype=np.bool)
    for i in range(3):
        is_face_valid_i = is_vert_on_surface[mc_faces[:, i]]
        is_face_valid = is_face_valid & is_face_valid_i
    
    # delete invalid verts
    raw_valid_faces = mc_faces[is_face_valid]
    raw_valid_vert_idx = np.unique(raw_valid_faces.flatten())
    valid_verts = mc_verts[raw_valid_vert_idx]
    
    valid_vert_idx = np.arange(len(valid_verts))
    vert_raw_idx_valid_idx_map = np.zeros(len(mc_verts), dtype=mc_faces.dtype)
    vert_raw_idx_valid_idx_map[raw_valid_vert_idx] = valid_vert_idx
    valid_faces = vert_raw_idx_valid_idx_map[raw_valid_faces]
    return valid_verts, valid_faces


def delete_invalid_verts(mc_verts, mc_faces, is_vert_on_surface):
    is_face_valid = np.ones(len(mc_faces), dtype=np.bool)
    for i in range(3):
        is_face_valid_i = is_vert_on_surface[mc_faces[:, i]]
        is_face_valid = is_face_valid & is_face_valid_i

    raw_valid_faces = mc_faces[is_face_valid]
    raw_valid_vert_idx = np.unique(raw_valid_faces.flatten())
    valid_verts = mc_verts[raw_valid_vert_idx]
    
    valid_vert_idx = np.arange(len(valid_verts))
    vert_raw_idx_valid_idx_map = np.zeros(len(mc_verts), dtype=mc_faces.dtype)
    vert_raw_idx_valid_idx_map[raw_valid_vert_idx] = valid_vert_idx
    valid_faces = vert_raw_idx_valid_idx_map[raw_valid_faces]
    return valid_verts, valid_faces

