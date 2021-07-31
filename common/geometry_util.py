from typing import Optional, Tuple

import numpy as np
import igl


def build_line(length=1.0, num_nodes=50):
    verts = np.zeros((num_nodes, 3), dtype=np.float32)
    verts[:, 0] = np.linspace(0, length, num_nodes)
    edges = np.empty((num_nodes - 1, 2), dtype=np.uint32)
    edges[:, 0] = range(0, num_nodes - 1)
    edges[:, 1] = range(1, num_nodes)
    return verts, edges


def build_rectangle(width=0.45, height=0.32, width_num_node=23, height_num_node=17):
    """
    Row major, row corresponds to width
    """
    #width_num_node = int(np.round(width / grid_size)) + 1
    #height_num_node = int(np.round(height / grid_size)) + 1

    print("Creating a rectangular grid with the following parameters:")
    print("Width:", width)
    print("Height:", height)
    print("W nodes::", width_num_node)
    print("H nodes:", height_num_node)

    def xy_to_index(x_idx, y_idx):
        # Assumes the following layout in imagespace - 0 is to the top left of the image
        #
        #        0          cloth_x_size+0    ...  cloth_y_size*cloth_x_size - cloth_x_size + 0
        #        1          cloth_x_size+1    ...  cloth_y_size*cloth_x_size - cloth_x_size + 1
        #        2          cloth_x_size+2    ...  cloth_y_size*cloth_x_size - cloth_x_size + 2
        #       ...
        #  cloth_x_size-1   cloth_x_size*2-1  ...  cloth_y_size*cloth_x_size - 1
        # return x_idx * width_num_node + y_idx
        return y_idx * height_num_node + x_idx

    verts = np.zeros((width_num_node * height_num_node, 3), dtype=np.float32)
    uv = np.zeros((width_num_node * height_num_node, 2), dtype=np.float32)
    edges_temp = []
    faces_temp = []
    for x in range(height_num_node):
        for y in range(width_num_node):
            curr_idx = xy_to_index(x, y)
            verts[curr_idx, 0] = x * height / (height_num_node - 1)
            verts[curr_idx, 1] = y * width / (width_num_node - 1)
            uv[curr_idx, 0] = x / (height_num_node - 1)
            uv[curr_idx, 1] = y / (width_num_node - 1)

            if x + 1 < height_num_node:
                edges_temp.append([curr_idx, xy_to_index(x + 1, y)])
            if y + 1 < width_num_node:
                edges_temp.append([curr_idx, xy_to_index(x, y + 1)])
            if x + 1 < height_num_node and y + 1 < width_num_node:
                faces_temp.append([curr_idx, xy_to_index(x + 1, y), xy_to_index(x + 1, y + 1), xy_to_index(x, y + 1)])

    edges = np.array(edges_temp, dtype=np.uint32)
    faces = np.array(faces_temp, dtype=np.uint32)
    return verts, edges, faces, uv

def faces_to_edges(faces):
    edges_set = set()
    for face in faces:
        for i in range(1, len(face)):
            edge_pair = (face[i-1], face[i])
            edge_pair = tuple(sorted(edge_pair))
            edges_set.add(edge_pair)
    edges = np.array(list(edges_set), dtype=np.int)
    return edges

class AABBNormalizer:
    def __init__(self, aabb):
        center = np.mean(aabb, axis=0)
        edge_lengths = aabb[1] - aabb[0]
        scale = 1 / np.max(edge_lengths)
        result_center = np.ones((3,), dtype=aabb.dtype) / 2

        self.center = center
        self.scale = scale
        self.result_center = result_center
    
    def __call__(self, data):
        center = self.center
        scale = self.scale
        result_center = self.result_center

        result = (data - center) * scale + result_center
        return result
    
    def inverse(self, data):
        center = self.center
        scale = self.scale
        result_center = self.result_center

        result = (data - result_center) / scale + center
        return result

class AABBGripNormalizer:
    """
    Assumes that the origin is gripping point. 
    Only translate the aabb in z direction and scale to fit.
    """
    def __init__(self, aabb, padding=0.05):
        nocs_radius = 0.5 - padding
        radius = np.max(np.abs(aabb), axis=0)[:2]
        radius_scale = np.min(nocs_radius / radius)
        nocs_z = nocs_radius * 2
        z_length = aabb[1,2] - aabb[0,2]
        z_scale = nocs_z / z_length
        scale = min(radius_scale, z_scale)

        z_max = aabb[1,2] * scale
        offset = np.array([0.5, 0.5, 1-padding-z_max], dtype=aabb.dtype)
        self.scale = scale
        self.offset = offset
    
    def __call__(self, data):
        scale = self.scale
        offset = self.offset
        result = (data * scale) + offset
        return result
    
    def inverse(self, data):
        scale = self.scale
        offset = self.offset
        result = (data - offset) / scale
        return result


def get_aabb(coords):
    """
    Axis Aligned Bounding Box
    Input:
    coords: (N, C) array
    Output:
    aabb: (2, C) array
    """
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    aabb = np.stack([min_coords, max_coords])
    return aabb


def buffer_aabb(aabb, buffer):
    result_aabb = aabb.copy()
    result_aabb[0] -= buffer
    result_aabb[1] += buffer
    return result_aabb


def quads2tris(quads):
    assert(isinstance(quads, np.ndarray))
    assert(len(quads.shape) == 2)
    assert(quads.shape[1] == 4)

    # allocate new array
    tris = np.zeros((quads.shape[0] * 2, 3), dtype=quads.dtype)
    tris[0::2] = quads[:, [0,1,2]]
    tris[1::2] = quads[:, [0,2,3]]
    return tris


def barycentric_interpolation(query_coords: np.array, verts: np.array, faces: np.array) -> np.array:
    """
    Input:
    query_coords: np.array[M, 3] float barycentric coorindates
    verts: np.array[N, 3] float vertecies
    faces: np.array[M, 3] int face index into verts, 1:1 coorespondace to query_coords

    Output
    result: np.array[M, 3] float interpolated points
    """
    assert(len(verts.shape) == 2)
    result = np.zeros((len(query_coords), verts.shape[1]), dtype=verts.dtype)
    for c in range(verts.shape[1]):
        for i in range(query_coords.shape[1]):
            result[:, c] += \
                query_coords[:, i] * verts[:,c][faces[:,i]]
    return result


def mesh_sample_barycentric(
        verts: np.ndarray, faces: np.ndarray, 
        num_samples: int, seed: Optional[int] = None,
        face_areas: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample points (as their barycentric coordinate) on suface

    Input:
    verts: np.array[N, 3] float mesh vertecies
    faces: np.array[M, 3] int mesh face index into verts
    num_sampels: int
    seed: int random seed
    face_areas: np.array[M, 3] per-face areas

    Output:
    barycentric_all: np.array[num_samples, 3] float sampled barycentric coordinates
    selected_face_idx: np.array[num_samples,3] int sampled faces, 1:1 coorespondance to barycentric_all
    """
    # generate face area
    if face_areas is None:
        face_areas = igl.doublearea(verts, faces)
    face_areas = face_areas / np.sum(face_areas)
    assert(len(face_areas) == len(faces))

    rs = np.random.RandomState(seed=seed)
    # select faces
    selected_face_idx = rs.choice(
        len(faces), size=num_samples, 
        replace=True, p=face_areas).astype(faces.dtype)
    
    # generate random barycentric coordinate
    barycentric_uv = rs.uniform(0, 1, size=(num_samples, 2))
    not_triangle = (np.sum(barycentric_uv, axis=1) >= 1)
    barycentric_uv[not_triangle] = 1 - barycentric_uv[not_triangle]

    barycentric_all = np.zeros((num_samples, 3), dtype=barycentric_uv.dtype)
    barycentric_all[:, :2] = barycentric_uv
    barycentric_all[:, 2] = 1 - np.sum(barycentric_uv, axis=1)

    return barycentric_all, selected_face_idx

