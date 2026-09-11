import numpy as np
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, Point
from skimage.morphology import convex_hull_image

from echotk.utils.config import Label
from echotk.utils.measure import ContourMeasure


def shift_down(mask, k):
    out = np.zeros_like(mask)
    out[k:, :] = mask[:-k, :]
    return out

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

    # shift down to avoid error from segmentation being cut off
    myo = shift_down(myo, 1) # only one pixel needed

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

    # shift back up 1 pixel for all points
    myo_points[:, 0] -= 1
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


MESH_METHODS = ("linear", "perpendicular")


def get_mesh(seg: np.ndarray, nb_points: int = 36, nb_rad: int = 5, method: str = "linear"):
    """Contours of each frame's segmentation, and the mesh spanning them.

    Args:
        seg: (n_frames, H, W) segmentation.
        nb_points: points along each of the two contours.
        nb_rad: layers in the mesh, including both contours.
        method: how the layers between the contours are placed. "linear" joins point i of the
            endocardium to point i of the epicardium; "perpendicular" runs each spoke normal to
            the wall instead (see `mesh_between_curves`), at the cost of a per-point
            intersection.

    Returns:
        endo_points (n_frames, nb_points, 2), epi_points (n_frames, nb_points, 2),
        mesh (n_frames, nb_points * nb_rad, 2)
    """
    if method not in MESH_METHODS:
        raise ValueError(f"unknown mesh method {method!r}, expected one of {MESH_METHODS}")

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
    if method == "perpendicular":
        mesh = compute_mesh_from_pts_perpendicular(endo_points, epi_points, nb_rad)
    else:
        mesh = compute_mesh_from_pts(endo_points, epi_points, nb_rad)

    return endo_points, epi_points, mesh



# Blend weights for the first (and, mirrored, the last) points of a perpendicular mesh: 0 keeps
# the straight endo[i]->epi[i] segment, 1 keeps the perpendicular one. The base is where the two
# contours end, so a perpendicular there leans on the extrapolated part of the polylines and can
# land off the real contour; ramping in over a few points anchors the mesh on the true base
# points while the rest of the wall gets the orthogonal spokes.
_END_BLEND_WEIGHTS = (0, 0.125, 0.25, 0.5, 0.625, 0.75, 0.875)


def mesh_between_curves(curve1: np.ndarray, curve2: np.ndarray, nb_rad: int = 5,
                        u_extension: float = 0.05,
                        min_spacing: float = 0.35) -> np.ndarray:
    """Mesh two curves by spokes perpendicular to the wall rather than by index pairing.

    `compute_mesh_from_pts` joins point i of one curve to point i of the other, which assumes
    the two are in correspondence; they are not, since each is sampled by its own `linspace`
    over its own arc length, so those spokes skew wherever the curves run at different rates.
    Here each spoke is instead the normal to the midline: the perpendicular at the midpoint of
    pair i, intersected with each curve, and sampled between the two intersections.

    A normal field converges on the concave side, so around the apex -- where the endocardium
    curves tighter than the wall is thick -- neighbouring normals meet inside the wall and the
    mesh folds over itself. `min_spacing` is what stops that: the feet of the spokes are held
    in order along each curve, and no closer together than that fraction of the spacing an
    even sampling would give. A spoke that would have crossed its neighbour is slid along the
    curve instead, which costs it its right angle and keeps the mesh single-valued.

    Args:
        curve1: (nb_pts, 2) points of the first curve, e.g. the endocardium.
        curve2: (nb_pts, 2) points of the second curve, e.g. the epicardium. Same length.
        nb_rad: number of layers in the mesh, including both curves.
        u_extension: fraction of the parameter range each curve is extrapolated by at both
            ends, so that perpendiculars near the base still meet them.
        min_spacing: floor on the gap between consecutive feet along a curve, as a fraction of
            that curve's length over nb_pts - 1. 0 lets the spokes fall where the normals put
            them, folds and all.

    Returns:
        mesh (nb_pts, nb_rad, 2), going from curve1 to curve2 along each spoke.
    """
    assert curve1.shape == curve2.shape, \
        f"curves must correspond point to point, got {curve1.shape} and {curve2.shape}"
    nb_pts = curve1.shape[0]

    mids = 0.5 * (curve1 + curve2)

    # Tangent of the midline by central differences, one-sided at the ends, and its normal.
    tang = np.zeros_like(mids)
    tang[1:-1] = mids[2:] - mids[:-2]
    tang[0] = mids[1] - mids[0]
    tang[-1] = mids[-1] - mids[-2]
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-8)
    perps = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

    # k=1: the curves are contours traced pixel by pixel, so interpolate them linearly rather
    # than smoothing them into something that no longer passes through the traced points. The
    # parameter is sampled past [0, 1] so the extrapolated ends are part of the polyline.
    tck1, _ = splprep(curve1.T, s=0, k=1)
    tck2, _ = splprep(curve2.T, s=0, k=1)
    u_ext = np.linspace(-u_extension, 1 + u_extension, 1000)
    line1 = LineString(np.column_stack(splev(u_ext, tck1, ext=0)))
    line2 = LineString(np.column_stack(splev(u_ext, tck2, ext=0)))

    # Half-length of the perpendicular probe: enough to cross the wall from any midpoint, in
    # the pixel units the contours are in.
    reach = 100.0

    feet1 = np.zeros((nb_pts, 2))
    feet2 = np.zeros((nb_pts, 2))
    for i in range(nb_pts):
        perp_line = LineString([tuple(mids[i] + perps[i] * reach),
                                tuple(mids[i] - perps[i] * reach)])
        feet1[i] = _closest_intersection(perp_line, line1, mids[i])
        feet2[i] = _closest_intersection(perp_line, line2, mids[i])

    if min_spacing > 0:
        feet1 = _order_feet(line1, feet1, min_spacing * _length(curve1) / (nb_pts - 1))
        feet2 = _order_feet(line2, feet2, min_spacing * _length(curve2) / (nb_pts - 1))

    alpha = np.linspace(0, 1, nb_rad)[:, None]
    mesh = (1 - alpha) * feet1[:, None, :] + alpha * feet2[:, None, :]

    # Ends: fade back to the index-paired segment, which is anchored on the real end points.
    straight = (1 - alpha[None]) * curve1[:, None, :] + alpha[None] * curve2[:, None, :]
    n_blend = min(len(_END_BLEND_WEIGHTS), nb_pts // 2)
    for i in range(n_blend):
        w = _END_BLEND_WEIGHTS[i]
        for j in (i, nb_pts - 1 - i):
            mesh[j] = w * mesh[j] + (1 - w) * straight[j]

    return mesh


def _length(curve: np.ndarray) -> float:
    """Polyline length of a curve given by its points."""
    return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())


def _order_feet(curve_line: LineString, feet: np.ndarray, min_gap: float) -> np.ndarray:
    """Slide the spokes' feet along `curve_line` until they are in order and `min_gap` apart.

    Works in arc length, so a foot only ever moves along the curve it already sits on. The
    corrected positions are the closest ones (least squares) that satisfy s[i+1] >= s[i] +
    min_gap, which is the isotonic regression of s[i] - i * min_gap -- so feet the normals put
    somewhere sensible stay put, and only a run that piles up gets spread out.
    """
    s = np.array([curve_line.project(Point(f)) for f in feet])

    # The curve may be traced against the order of the points; correct in the rising direction.
    descending = np.median(np.diff(s)) < 0
    if descending:
        s = curve_line.length - s

    s = _isotonic(s - min_gap * np.arange(len(s))) + min_gap * np.arange(len(s))
    s = np.clip(s, 0.0, curve_line.length)

    if descending:
        s = curve_line.length - s
    return np.array([curve_line.interpolate(v).coords[0] for v in s])


def _isotonic(y: np.ndarray) -> np.ndarray:
    """Closest non-decreasing sequence to `y` in least squares (pool adjacent violators)."""
    values, weights = [], []
    for v in y:
        values.append(float(v))
        weights.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2 = values.pop(), weights.pop()
            v1, w1 = values.pop(), weights.pop()
            values.append((v1 * w1 + v2 * w2) / (w1 + w2))
            weights.append(w1 + w2)
    return np.repeat(values, weights)


def _closest_intersection(perp_line: LineString, curve_line: LineString, origin: np.ndarray):
    """Where `perp_line` crosses `curve_line`, nearest to `origin`.

    A perpendicular can cross a contour more than once (the opposite wall is on the same line),
    so the nearest crossing is the one on the wall being meshed. With no crossing at all -- the
    perpendicular running off the end of an already extrapolated curve -- the nearer of the
    curve's two ends stands in, which keeps the spoke on the contour instead of at infinity.
    """
    inter = perp_line.intersection(curve_line)
    if isinstance(inter, Point):
        pts = [inter]
    else:
        pts = [geom for geom in getattr(inter, "geoms", []) if isinstance(geom, Point)]
    if not pts:
        pts = [Point(curve_line.coords[0]), Point(curve_line.coords[-1])]

    origin_geom = Point(origin)
    return np.array(min(pts, key=lambda p: p.distance(origin_geom)).coords[0])


def compute_mesh_from_pts_perpendicular(endo_pts: np.ndarray, epi_pts: np.ndarray, nb_rad: int,
                                        min_spacing: float = 0.35):
    """`compute_mesh_from_pts` with `mesh_between_curves`' spokes.

    Same arguments and same (n_frames, nb_pts * nb_rad, 2) layout -- layer by layer, so the
    first nb_pts points are the endocardium and the last nb_pts the epicardium, which is what
    `mesh2mask` and the tracker expect. `min_spacing` is `mesh_between_curves`' fold guard.
    """
    nb_frames = endo_pts.shape[0]
    assert nb_frames == epi_pts.shape[0]

    meshes = [mesh_between_curves(endo_pts[i], epi_pts[i], nb_rad=nb_rad, min_spacing=min_spacing)
              for i in range(nb_frames)]
    # (frames, nb_pts, nb_rad, 2) -> (frames, nb_rad, nb_pts, 2) -> flat, layer-major
    return np.swapaxes(np.array(meshes), 1, 2).reshape(nb_frames, -1, 2)
