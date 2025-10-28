import numpy as np
from skimage.morphology import convex_hull_image

from echotk.utils.config import Label
from echotk.utils.measure import ContourMeasure


# MESH
def lv_contour(segmentation, nb_points, identify_apex=True):
    lv_edge = ContourMeasure.structure_edge(segmentation=segmentation, label=Label.LV)

    base = np.array(ContourMeasure._endo_base(segmentation, Label.LV, Label.MYO))

    if identify_apex:
        apex = np.array(ContourMeasure.lv_apex(segmentation))

        path1 = ContourMeasure.get_path(lv_edge, tuple(apex), tuple(base[0]))
        path2 = ContourMeasure.get_path(lv_edge, tuple(apex), tuple(base[1]))

        points_per_side = (nb_points + 1) // 2

        path1_points_idx = np.linspace(0, len(path1) - 1, points_per_side).astype(int)
        path2_points_idx = np.linspace(0, len(path2) - 1, points_per_side).astype(int)

        points = np.concatenate(
            (
                base[0][None],
                path1[path1_points_idx[1:-1]],
                apex[None],
                path2[-path2_points_idx[1:-1]],
                base[1][None],
            ),
            axis=0,
        )
    else:

        # TODO simply concat path1 and path2 with apex
        base_mid = np.array([base[0], base[1]]).mean(axis=0).round().astype(int)
        lv_edge[base_mid[0] - 10:base_mid[0] + 10, base_mid[1] - 10:base_mid[1] + 10] = 0

        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.imshow(lv_edge)
        #
        #
        #
        # plt.figure()
        # plt.imshow(lv_edge)
        # plt.scatter(base_mid[1], base_mid[0])
        # plt.show()


        path = ContourMeasure.get_path(lv_edge, tuple(base[0]), tuple(base[1]))

        path_points_idx = np.linspace(0, len(path) - 1, nb_points).astype(int)

        points = path[path_points_idx]

    return np.flip(points, axis=0)


def myo_contour(segmentation, nb_points, identify_apex=True):
    myo = np.isin(segmentation, Label.MYO)

    myo_convex = convex_hull_image(myo)

    myo_points = ContourMeasure._extract_landmarks_from_polar_contour(
        myo_convex, 1, polar_smoothing_factor=5e-3, debug_plots=False
    )

    myo_edge = ContourMeasure.structure_edge(segmentation=myo_convex, label=1)  # Mask only contains filled MYO

    myo_points = myo_points.round().astype(int)

    if identify_apex:

        path1 = ContourMeasure.get_path(myo_edge, tuple(myo_points[0]), tuple(myo_points[1]))
        path2 = ContourMeasure.get_path(myo_edge, tuple(myo_points[0]), tuple(myo_points[2]))

        points_per_side = (nb_points + 1) // 2

        path1_points_idx = np.linspace(0, len(path1) - 1, points_per_side).astype(int)
        path2_points_idx = np.linspace(0, len(path2) - 1, points_per_side).astype(int)

        myo_points = np.concatenate(
            (
                myo_points[1][None],
                path1[path1_points_idx[1:-1]],
                myo_points[0][None],
                path2[-path2_points_idx[1:-1]],
                myo_points[2][None],
            ),
            axis=0,
        )
    else:
        base_mid = np.array([myo_points[1], myo_points[2]]).mean(axis=0).round().astype(int)
        myo_edge[base_mid[0] - 10:base_mid[0] + 10, base_mid[1] - 10:base_mid[1] + 10] = 0

        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.imshow(myo_edge)
        #
        #
        #
        # plt.figure()
        # plt.imshow(myo_edge)
        # plt.scatter(base_mid[1], base_mid[0])
        # plt.show()

        path = ContourMeasure.get_path(myo_edge, tuple(myo_points[1]), tuple(myo_points[2]))

        path_points_idx = np.linspace(0, len(path) - 1, nb_points).astype(int)
        myo_points = path[path_points_idx]

    return np.flip(myo_points, axis=0)

def get_contour_points(segmentation, points_dict, identify_apex=True):
    lv_points = lv_contour(segmentation, points_dict[Label.LV], identify_apex)
    myo_points = myo_contour(segmentation, points_dict[Label.MYO], identify_apex)

    lv_points = np.flip(lv_points, axis=-1)
    myo_points = np.flip(myo_points, axis=-1)

    return lv_points, myo_points

def compute_mesh_from_pts(endo_pts: np.ndarray, epi_pts: np.ndarray, nb_rad: int, flip=False):
    """

    Args:
        endo_pts: endocardium points (x,z) (n_frames, nb_pts, 2)
        epi_pts: epicardium points (x, z) (n_frames, nb_pts, 2)
        nb_rad: Number of layers in the mesh (including endo and epi)
        flip: If True, the

    Returns:
        mesh (n_frames, nb_points*nb_rad, 2)
    """

    nb_frames = endo_pts.shape[0]
    assert nb_frames == epi_pts.shape[0]

    nb_pts = endo_pts.shape[1]
    assert nb_pts == epi_pts.shape[1]

    # if flip:
    #     endo_pts = endo_pts[:, ::-1, :]
    #     epi_pts = epi_pts[:, ::-1, :]

    alpha = np.linspace(0, 1, nb_rad)[None, None]

    x = endo_pts[:, :, 0][..., None] * (1 - alpha) + alpha * epi_pts[:, :, 0][..., None]
    z = endo_pts[:, :, 1][..., None] * (1 - alpha) + alpha * epi_pts[:, :, 1][..., None]

    # print(endo_pts[0, 0, 0], epi_pts[0, 0, 0], x[0, 0])
    # print(endo_pts[0, 0, 1], epi_pts[0, 0, 1], z[0, 0])
    # print(x.shape)

    x = np.reshape(x, (nb_frames, -1, 1), order='F')
    z = np.reshape(z, (nb_frames, -1, 1), order='F')

    return np.concatenate([x, z], axis=-1)


def get_mesh(seg: np.ndarray, nb_points: int = 36, nb_rad: int = 5):
    points_dict = {1: nb_points,
                   2: nb_points}

    endo_points = []
    epi_points = []
    for i in range(len(seg)):
        endo, epi = get_contour_points(seg[i], points_dict, identify_apex=False)
        endo_points.append(endo)
        epi_points.append(epi)

    endo_points = np.array(endo_points)
    epi_points = np.array(epi_points)
    mesh = compute_mesh_from_pts(endo_points, epi_points, nb_rad)

    return endo_points, epi_points, mesh