import numpy as np
from numba import jit
from matplotlib.cm import get_cmap
from skimage.transform import resize

# helper functions
# ================
@jit(nopython=True, nogil=True)
def _render_points_idx(xy_idx, z, idx_img, min_img, kernel_size, kernel_offset):
    for i in range(len(z)):
        x, y = xy_idx[i]
        this_z = z[i]
        min_z = min_img[y, x]
        for dy in range(kernel_offset, kernel_offset + kernel_size):
            ny = min(max(y + dy, 0), idx_img.shape[0] - 1)
            for dx in range(kernel_offset, kernel_offset + kernel_size):
                nx = min(max(x + dx, 0), idx_img.shape[1] - 1)
                min_z = min_img[ny, nx]
                if this_z < min_z:
                    min_img[ny, nx] = this_z
                    idx_img[ny, nx] = i


# low-level API
# =============
def render_points_idx(points, img_size=256, kernel_size=4):
    # assumes points are normized bewteen 0 and 1
    # assume s colros are rgb 0 to 1
    # images are in cv coordiante: (y, x)
    idx_dtype = np.uint32
    default_idx = np.iinfo(idx_dtype).max
    idx_img = np.full(shape=(img_size, img_size), 
        fill_value=default_idx, dtype=idx_dtype)
    min_img = np.full(shape=(img_size, img_size),
        fill_value=float('inf'), dtype=points.dtype)
    xy_idx = np.clip(
        (points[:,:2] * (img_size-1)).astype(idx_dtype), 
        0, img_size-1)
    z = points[:, 2]
    kernel_offset = -(kernel_size // 2)
    _render_points_idx(xy_idx, z, idx_img, min_img, kernel_size, kernel_offset)
    return idx_img


def color_idx_img(idx_img, colors, default_color=np.array([1,1,1])):
    h, w = idx_img.shape
    default_idx = np.iinfo(idx_img.dtype).max
    img_not_null = idx_img < default_idx
    idxs = idx_img[img_not_null]
    color_img = np.zeros((h, w, len(default_color)), dtype=np.float32)
    color_img[:, :] = default_color
    color_img[img_not_null] = colors[idxs]
    return color_img


def get_extrinsic(side='front'):
    # world to camera
    if side == 'front':
        return np.array([
            [1, 0, 0, 0],
            [0, 0,-1, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
    elif side == 'top':
        return np.array([
            [1, 0, 0, 0],
            [0,-1, 0, 1],
            [0, 0,-1, 1],
            [0, 0, 0, 1]
        ])
    elif side == 'left':
        return np.array([
            [0,-1, 0, 1],
            [0, 0,-1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
    else:
        assert(False)


def to_camera(points, extrinsic):
    rx = extrinsic[:3, :3]
    tx = extrinsic[:3, 3]
    result = points @ rx.T + tx
    return result


# high-level API
# ==============
def render_nocs(
    points, colors=None,
    side='front', img_size=256, kernel_size=4, 
    default_color=np.array([1,1,1,0])):
    extrinsic = get_extrinsic(side)
    camera_points = to_camera(points, extrinsic)
    if colors is None:
        colors = np.concatenate(
            [points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)

    idx_img = render_points_idx(camera_points, 
        img_size=img_size, kernel_size=kernel_size)
    color_img = color_idx_img(
        idx_img, colors, 
        default_color=default_color)
    return color_img


def render_wnf(wnf_img, img_size=256, cmap='viridis', min_value=-0.5, max_value=1.5):
    cmap = get_cmap(cmap)
    value_img = (wnf_img - min_value) / (max_value - min_value)
    color_img = cmap(value_img)
    final_img = resize(color_img, (img_size, img_size), anti_aliasing=False)
    return final_img

def get_wnf_cmap(cmap='viridis', min_value=-0.5, max_value=1.5):
    cmap = get_cmap(cmap)
    def cmap_func(x):
        values = (x - min_value) / (max_value - min_value)
        colors = cmap(values)
        return colors
    return cmap_func

def render_wnf_points(query_points, wnf_values, slice_range=(0.5, 0.6), side='front', **kwargs):
    cmap = get_wnf_cmap()
    colors = cmap(wnf_values)
    # TODO:
    assert side == 'front'
    dim_idx = 1
    is_selected = (slice_range[0] < query_points[...,dim_idx]) \
        & (query_points[...,dim_idx] < slice_range[1])

    color_img = render_nocs(
        points=query_points[is_selected], 
        colors=colors[is_selected], side=side, **kwargs)
    return color_img

def render_points_confidence(points, confidence, side='front', **kwargs):
    cmap = get_wnf_cmap(min_value=0.0, max_value=1.0)
    colors = cmap(confidence)
    color_img = render_nocs(
        points=points, 
        colors=colors, side=side, **kwargs)
    return color_img

