from typing import Union

import numpy as np
from skimage import draw

from echotk.utils.config import Label


def masks_from_meshes(meshes, image_shape, nb_pts=36, label:Label = None):
    masks = []
    for i in range(len(meshes)):
        masks.append(compute_mask_from_mesh(meshes[i], nb_pts, image_shape, label))
    return np.array(masks)


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=int)
    mask[fill_row_coords, fill_col_coords] = 1
    return mask


def compute_mask_from_mesh(mesh: np.ndarray, nb_pts: int, shape, label:Label = None):
    """

    Args:
        mesh:
        nb_pts:
        bmode_img:
        labels:
    Returns:

    """

    endo_points = mesh[0:nb_pts, :]
    epi_points = mesh[-nb_pts:, :]

    polygon = np.concatenate([endo_points, np.flip(epi_points, axis=0), endo_points[0][None]], axis=0)

    myo = poly2mask(polygon[:, 1], polygon[:, 0], shape)
    lv = poly2mask(endo_points[:, 1], endo_points[:, 0], shape)

    # single label --> binary mask
    if label is not None:
        return ((lv if int(label) == Label.LV else myo) == 1).astype(np.uint8)

    # multi-label --> use both masks
    mask = np.zeros(shape, dtype=np.uint8)
    mask[myo == 1] = Label.MYO
    mask[lv == 1] = Label.LV
    return mask