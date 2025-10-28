import numpy as np
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
from skimage import draw
from matplotlib import pyplot as plt
from matplotlib import path


def masks_from_meshes(meshes, image_shape, nb_pts=36):
    masks = []
    for i in range(len(meshes)):
        masks.append(compute_mask_from_mesh(meshes[i], nb_pts, image_shape))
    return np.array(masks)


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=int)
    mask[fill_row_coords, fill_col_coords] = 1
    return mask


def compute_mask_from_mesh(mesh: np.ndarray, nb_pts: int, shape):
    """

    Args:
        mesh:
        nb_pts:
        bmode_img:

    Returns:

    """

    endo_points = mesh[0:nb_pts, :]
    epi_points = mesh[-nb_pts:, :]

    polygon = np.concatenate([endo_points, np.flip(epi_points, axis=0), endo_points[0][None]], axis=0)

    mask = poly2mask(polygon[:, 1], polygon[:, 0], shape)

    # print(polygon.shape)
    #
    #
    #
    # from matplotlib import pyplot as plt
    # plt.imshow(mask)
    # plt.scatter(polygon[:, 0], polygon[:, 1])
    # plt.show()

    return mask