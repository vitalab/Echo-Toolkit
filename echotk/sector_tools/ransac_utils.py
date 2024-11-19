import math
import numpy as np
from matplotlib import pyplot as plt

from scipy.ndimage import binary_fill_holes
from scipy import signal
from skimage.measure import LineModelND, CircleModel, ransac
from skimage import draw
from skspatial.objects import Line

import warnings
warnings.filterwarnings(action='ignore', module='skimage')


def run_ransac_line(img, angle, constrain_angles=True, plot=False):
    points = np.nonzero(img)
    data = np.column_stack([points[1], points[0]])

    def is_line_valid(model, X, y=None):
        if constrain_angles:
            absolute_tol = 15
        else:
            absolute_tol = 30
        return math.isclose(np.rad2deg(np.arctan2(model.params[1][1], model.params[1][0])),
                            angle, abs_tol=absolute_tol)

    model_robust, _ = ransac(data, LineModelND, min_samples=2, is_model_valid=is_line_valid,
                             residual_threshold=1, max_trials=data.shape[0]*6, rng=0)

    line_x = np.arange(-int(img.shape[1]*0.25), int(img.shape[1]*1.25))
    line_y_robust = model_robust.predict_y(line_x)

    if plot:
        print(f"Angle of line: {np.rad2deg(np.arctan2(model_robust.params[1][1], model_robust.params[1][0]))}")
        fig, ax = plt.subplots()
        ax.set_title("Line and subset of points used to find it")
        ax.plot(points[1], points[0], '.g', alpha=0.6, label='all')
        ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
        ax.legend(loc='lower left')

    return model_robust, line_x, line_y_robust


def run_ransac_circle(img, expected_center, plot=False):
    points = np.nonzero(img)
    data = np.column_stack([points[1], points[0]])

    def is_circle_valid(model, X, y=None):
        # check if close enough to where center should be according to lines (tolerance of 20 pixels)
        # check if radius is at least 3/4 the height of image
        return img.shape[0] * 0.75 <= model.params[2] \
               and math.isclose(model.params[1], expected_center[1], abs_tol=20) \
               and math.isclose(model.params[0], expected_center[0], abs_tol=20) \

    model_robust, _ = ransac(data, CircleModel, min_samples=3, residual_threshold=1,
                             max_trials=data.shape[0]*6, is_model_valid=is_circle_valid, rng=0)
    if plot:
        plt.figure()
        plt.title("Points used to find circle")
        plt.plot(points[1], points[0], '.b')
        if not model_robust:
            plt.show()
    return model_robust


def ransac_sector_extraction(noisy_mask, slim_factor=0, use_convolutions=True, constrain_angles=True, circle_center_tol=0.5, plot=False):
    # remove pixels in center of shape, use only the edges
    noisy_mask = binary_fill_holes(noisy_mask.T)
    k = [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]]
    edges = signal.convolve2d(noisy_mask.astype(np.int32), k, mode='same')
    edges[edges == 0] = 255
    edges = edges < 8

    if use_convolutions:
        # find line on right side (angle is approx. 45 in coordinate space)
        k_45 = [[2, -1, -1],
                [-1, 2, -1],
                [-1, -1, 2]]
        edges_45 = signal.convolve2d(edges, k_45, mode='same')
        edges_45[edges_45 == edges_45.max()] = 1
        edges_45[edges_45 < edges_45.max()] = 0
    else:
        edges_45 = edges

    r_half = edges_45[:, edges_45.shape[1] // 2:]
    r_model, r_line_x, r_line_y = run_ransac_line(r_half, angle=55, constrain_angles=constrain_angles, plot=plot)
    r_line_x += (edges.shape[1] // 2)  # remove offset on r line

    if use_convolutions:
        # find line on left side (angle is approx. -45 in coordinate space)
        k_m45 = [[-1, -1, 2],
                 [-1, 2, -1],
                 [2, -1, -1]]
        edges_m45 = signal.convolve2d(edges, k_m45, mode='same')
        edges_m45[edges_m45 == edges_m45.max()] = 1
        edges_m45[edges_m45 < edges_m45.max()] = 0
    else:
        edges_m45 = edges

    l_half = edges_m45[:, :edges_m45.shape[1] // 2]
    l_model, l_line_x, l_line_y = run_ransac_line(l_half, angle=125, constrain_angles=constrain_angles, plot=plot)

    # use lines to find expected center of circle
    r_line = Line.from_points([r_line_x[0], r_line_y[0]], [r_line_x[-1], r_line_y[-1]])
    l_line = Line.from_points([l_line_x[0], l_line_y[0]], [l_line_x[-1], l_line_y[-1]])
    center = r_line.intersect_line(l_line)

    # filter horizontal/slight arc shapes in edges using "U" shaped filter
    # use only top half of points to find arc (reduces number of ransac iterations)
    k_arc = [[-1, -1, -1, -1, -1],
             [2,  -1, -1, -1,  2],
             [2,   2, -1,  2,  2],
             [-1,  2,  2,  2, -1],
             [-1, -1, -1, -1, -1]]
    edges_arc = signal.convolve2d(edges, k_arc, mode='same')
    edges_arc[:int(edges_arc.shape[0]*0.5), :] = 0
    edges_arc[edges_arc < (edges_arc.max() // 2)] = 0

    # find circle and plot onto canvas
    circle_model = run_ransac_circle(edges_arc, center, plot=plot)

    if plot:
        points = np.where(edges)
        _, ax = plt.subplots()
        ax.set_title("Calculated lines and circle arc")
        r_line.plot_2d(ax)
        l_line.plot_2d(ax)
        ax.plot(points[1], points[0], '.g')
        if circle_model:
            rr_c, cc_c = draw.circle_perimeter(int(circle_model.params[0]),
                                               int(circle_model.params[1]),
                                               int(circle_model.params[2]),
                                               shape=edges.T.shape)
            ax.scatter(circle_model.params[0], circle_model.params[1], s=50, c='red')
        else:
            rr_c, cc_c = draw.circle_perimeter(int(center[0]),
                                               int(center[1]),
                                               int(np.sqrt(edges.shape[1]**2 + edges.shape[0]**2)),
                                               shape=edges.T.shape)
        ax.plot(rr_c, cc_c, '.b')
        ax.scatter(center[0], center[1], s=50, c='yellow')

    mask = np.zeros_like(edges)
    center = center.astype(int)
    if circle_model and \
            np.isclose(circle_model.params[0], center[0], atol=mask.shape[0]*circle_center_tol) and \
            np.isclose(circle_model.params[1], center[1], atol=mask.shape[1]*circle_center_tol):
        rad = (circle_model.params[2] +
               np.sqrt((circle_model.params[1] - center[1])**2 + (circle_model.params[0] - center[0])**2)).astype(int)
    else:
        rad = np.sqrt((mask.shape[1]*1.5)**2 + (mask.shape[0]*1.5)**2)

    r_line_angle = np.arctan2(r_model.params[1][1], r_model.params[1][0])
    l_line_angle = np.arctan2(l_model.params[1][1], l_model.params[1][0])

    if center[1] < 0:
        l_start_x = (center[0] + center[1] / np.tan(np.pi - l_line_angle)).astype(int)
        l_start_y = 0
        l_offset = (-center[1] / np.cos(l_line_angle - np.pi/2)).astype(int)
        r_start_x = (center[0] - np.tan(np.pi/2 - r_line_angle) * center[1]).astype(int)
        r_start_y = 0
        r_offset = (-center[1] / np.cos(np.pi/2 - r_line_angle)).astype(int)

        # draw line on row 0 between both diagonal lines
        rr, cc = draw.line(l_start_y, l_start_x, r_start_y, r_start_x)
        mask[rr, cc] = 1
    else:
        r_offset = 0
        l_offset = 0
        l_start_x = center[0]
        l_start_y = center[1]
        r_start_x = center[0]
        r_start_y = center[1]

    l_end_y = (l_start_y + (rad - l_offset) * l_model.params[1][1]).astype(int)
    l_end_x = (l_start_x + (rad - l_offset) * l_model.params[1][0]).astype(int)
    if l_end_x <= 0:
        l_end_x = 0
        l_end_y = (l_start_x * np.tan(np.pi - l_line_angle)).astype(int)
    l_end_x += int(mask.shape[1] * slim_factor)
    rr, cc = draw.line(l_start_y,
                       l_start_x,
                       l_end_y,
                       l_end_x)
    ll_idx = np.asarray([cc, rr])
    ll_idx = ll_idx[:, (ll_idx[0] < mask.shape[1]) & (ll_idx[0] >= 0)]
    ll_idx = ll_idx[:, (ll_idx[1] < mask.shape[0]) & (ll_idx[1] >= 0)]

    r_end_y = (r_start_y + (rad - r_offset) * r_model.params[1][1]).astype(int)
    r_end_x = (r_start_x + (rad - r_offset) * r_model.params[1][0]).astype(int)
    # special case when right corner not in frame
    if r_end_x >= mask.shape[1]:
        r_end_x = mask.shape[1]-1
        r_end_y = ((mask.shape[1] - r_start_x) * np.tan(r_line_angle)).astype(int)
    r_end_x -= int(mask.shape[1] * slim_factor)
    rr, cc = draw.line(r_start_y,
                       r_start_x,
                       r_end_y,
                       r_end_x)
    rl_idx = np.asarray([cc, rr])
    rl_idx = rl_idx[:, (rl_idx[0] < mask.shape[1]) & (rl_idx[0] >= 0)]
    rl_idx = rl_idx[:, (rl_idx[1] < mask.shape[0]) & (rl_idx[1] >= 0)]

    if len(rl_idx[0]) == 0 or len(ll_idx[0]) == 0:
        raise ValueError('Invalid line(s)')

    # draw circle arc
    if circle_model and \
            np.isclose(circle_model.params[0], center[0], atol=mask.shape[0]*circle_center_tol) and \
            np.isclose(circle_model.params[1], center[1], atol=mask.shape[1]*circle_center_tol):
        rr_c, cc_c = draw.circle_perimeter(int(circle_model.params[0]),
                                           int(circle_model.params[1]),
                                           int(circle_model.params[2]),
                                           shape=(mask.shape[1], int(mask.shape[0]*1.5)))
    else:
        # print("Circle center and line intersect not close at all")
        rr_c, cc_c = draw.circle_perimeter(int(center[0]),
                                           int(center[1]),
                                           int(np.sqrt(mask.shape[1] ** 2 + mask.shape[0] ** 2)),
                                           shape=(mask.shape[1], int(mask.shape[0]*2)))
    cc_c[cc_c >= mask.shape[0]] = mask.shape[0] - 1
    arc_idx = np.asarray([rr_c, cc_c])
    arc_idx = arc_idx[:, arc_idx[0, :].argsort()]
    # remove top half of circle
    arc_idx = arc_idx[:, arc_idx[1] > circle_model.params[1]]
    # remove circle idx if out of bounds of line
    arc_idx = arc_idx[:, (arc_idx[0] <= r_end_x) & (arc_idx[0] > l_end_x)]

    # find and cut off crossing lines
    l_crossing = np.where(np.isclose(arc_idx.T[:, None], ll_idx.T, atol=0.99).all(-1).any(-1))[0]
    if len(l_crossing) > 0:
        ll_idx = ll_idx[:, :np.where(np.isclose(ll_idx[1, :], arc_idx[1, l_crossing[0]], atol=0.99))[0][0]]
        arc_idx = arc_idx[:, l_crossing[0]:]

    r_crossing = np.where(np.isclose(arc_idx.T[:, None], rl_idx.T, atol=0.99).all(-1).any(-1))[0]
    if len(r_crossing) > 0:
        rl_idx = rl_idx[:, :np.where(np.isclose(rl_idx[1, :], arc_idx[1, r_crossing[-1]], atol=0.99))[0][0]]
        arc_idx = arc_idx[:, :r_crossing[-1]+1]

    # draw the lines on the mask image
    mask[arc_idx[1], arc_idx[0]] = 1
    mask[ll_idx[1], ll_idx[0]] = 1
    mask[rl_idx[1], rl_idx[0]] = 1

    # fill in missing small bits
    if arc_idx[1, 0] != ll_idx[1, -1] or arc_idx[0, 0] != ll_idx[0, -1]:
        rr, cc = draw.line(arc_idx[1, 0],
                           arc_idx[0, 0],
                           ll_idx[1, -1],
                           ll_idx[0, -1])
        mask[rr, cc] = 1
    if arc_idx[1, -1] != rl_idx[1, -1] or arc_idx[0, -1] != rl_idx[0, -1]:
        rr, cc = draw.line(arc_idx[1, -1],
                           arc_idx[0, -1],
                           rl_idx[1, -1],
                           rl_idx[0, -1])
        mask[rr, cc] = 1

    mask = binary_fill_holes(mask)

    if plot:
        plt.figure()
        plt.title('Edges used for computation of mask, with output mask')
        plt.imshow(edges)
        plt.imshow(mask, alpha=0.5)
        plt.show()

    # return mask in same format as input
    return mask.T, {'circle_model': circle_model.params,
                    'left_model': [p.tolist() for p in l_model.params],
                    'right_model': [p.tolist() for p in r_model.params],
                    'intercept': [int(c) for c in center]}
