import numpy as np

from common.rendering_util import render_nocs, render_wnf, render_wnf_points, render_points_confidence

# high-level API
# ==============
def overlay_grip(img, grip_nocs, color=(1,0,0,1), side='front', kernel_size=4):
    assert(img.shape[0] == img.shape[1])
    img_size = img.shape[0]
    grip_img = render_nocs(
        points=np.expand_dims(grip_nocs, axis=0), colors=np.array([color]), 
        side=side, img_size=img_size, kernel_size=kernel_size)
    is_grip = grip_img[:,:,3] > 0
    new_img = img.copy()
    new_img[is_grip] = grip_img[is_grip]
    return new_img


def render_nocs_pair(gt_nocs, pred_nocs, gt_grip_nocs=None, 
    pred_grip_nocs=None, pred_grip_nocs_nn=None,
    side='front', img_size=256, kernel_size=4):
    """
    Both colored using gt_nocs's nocs as rgb
    """

    dtype = gt_nocs.dtype
    colors = np.concatenate(
        [gt_nocs, np.ones(
            (len(gt_nocs), 1), dtype=dtype)], axis=1)
    gt_img = render_nocs(
        gt_nocs, colors=colors, 
        side=side, img_size=img_size, kernel_size=kernel_size)
    pred_img = render_nocs(
        pred_nocs, colors=colors, 
        side=side, img_size=img_size, kernel_size=kernel_size)
    if gt_grip_nocs is not None:
        gt_img = overlay_grip(gt_img, gt_grip_nocs, 
            side=side, kernel_size=kernel_size*2)
    if pred_grip_nocs is not None:
        pred_img = overlay_grip(pred_img, pred_grip_nocs, 
            side=side, kernel_size=kernel_size*2)
    if pred_grip_nocs_nn is not None:
        pred_img = overlay_grip(pred_img, pred_grip_nocs_nn, color=(0,1,0,1),
            side=side, kernel_size=kernel_size*2)
    pair_img = np.concatenate([gt_img, pred_img], axis=1)
    return pair_img


def render_confidence_pair(gt_nocs, pred_nocs, confidence, 
    side='front', img_size=256, kernel_size=4):
    gt_img = render_points_confidence(
        gt_nocs, confidence)
    pred_img = render_points_confidence(
        pred_nocs, confidence)
    pair_img = np.concatenate([gt_img, pred_img], axis=1)
    return pair_img


def render_wnf_pair(gt_wnf_img, pred_wnf_img, img_size=256):
    gt_img = render_wnf(gt_wnf_img, img_size=img_size)
    pred_img = render_wnf(pred_wnf_img, img_size=img_size)
    pair_img = np.concatenate([gt_img, pred_img], axis=1)
    return pair_img

def render_wnf_points_pair(query_points, gt_wnf, pred_wnf, img_size=256):
    gt_img = render_wnf_points(
        query_points=query_points, wnf_values=gt_wnf, img_size=img_size)
    pred_img = render_wnf_points(
        query_points=query_points, wnf_values=pred_wnf, img_size=img_size)
    pair_img = np.concatenate([gt_img, pred_img], axis=1)
    return pair_img

def get_vis_idxs(batch_idx, 
        batch_size=None, this_batch_size=None, 
        vis_per_items=1, max_vis_per_epoch=None):
    assert((batch_size is not None) or (this_batch_size is not None))
    if this_batch_size is None:
        this_batch_size = batch_size
    if batch_size is None:
        batch_size = this_batch_size
    
    global_idxs = list()
    selected_idxs = list()
    vis_idxs = list()
    for i in range(this_batch_size):
        global_idx = batch_size * batch_idx + i
        global_idxs.append(global_idx)
        vis_idx = global_idx // vis_per_items
        vis_modulo = global_idx % vis_per_items
        if (vis_modulo == 0) and (vis_idx < max_vis_per_epoch):
            selected_idxs.append(i)
            vis_idxs.append(vis_idx)
    return global_idxs, selected_idxs, vis_idxs

