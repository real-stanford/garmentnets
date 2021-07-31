import numpy as np
import potpourri3d as pp3d


def geodesic_matrix(verts, faces, vert_idxs):
    """
    Pair-wise geodesic distance between all vertecies
    """
    solver = pp3d.MeshHeatMethodDistanceSolver(verts, faces)
    length = len(vert_idxs)
    result_mat = np.zeros((length, length))
    for i, vert_idx in enumerate(vert_idxs):
        all_dists = solver.compute_distance(vert_idx)
        result_mat[i] = all_dists[vert_idxs]
    return result_mat
