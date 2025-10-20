import warnings

import numpy as np

from echotk.metrics.cardiac_cycle import estimate_num_cycles
from echotk.metrics.utils.config import Label
from echotk.metrics.utils.measure import EchoMeasure

warnings.simplefilter(action='ignore', category=FutureWarning)


def mitral_valve_distance(batch_segmentation, gt, spacing, mistake_distances=[5, 7.5], return_mean=True):
    """
    Compute mitral valve localization errors between candidate and ground-truth segmentations.

    This function evaluates how accurately the mitral valve base points are localized in the
    predicted segmentation relative to the ground truth. It computes mean absolute error (MAE),
    mean squared error (MSE), and counts of “mistake” frames — cases where the valve base is
    displaced by more than a given threshold (in mm). These mistake counts are normalized by
    the estimated number of cardiac cycles.

    Parameters
    ----------
    batch_segmentation : np.ndarray
        Predicted segmentation sequence of shape `(T, H, W)` or `(T, W, H)`,
        where `T` is the number of frames.
    gt : np.ndarray
        Ground-truth segmentation sequence of the same shape as `segmentation`.
    spacing : tuple of float
        Physical pixel spacing (in mm) along the two spatial dimensions.
        Must be approximately isometric (i.e., spacing[0] ≈ spacing[1]).
    mistake_distances : list of float, optional
        Distance thresholds (in mm) used to define “mistakes” per cycle.
        Defaults to `[5, 7.5]`.
    return_mean : bool, optional
        If `True`, returns mean values for MAE and MSE.
        If `False`, returns per-frame values. Defaults to `True`.

    Returns
    -------
    dict
        Dictionary containing the following metrics:

        - **mse** : float or np.ndarray
          Mean (or per-frame) mean squared error between mitral valve base points.
        - **mae_L**, **mae_R**, **mae** : float or np.ndarray
          Mean absolute error for the left base point, right base point, and their average.
        - **mistake_per_cycle_{d}mm**, **mistake_per_cycle_{d}mm_L**, **mistake_per_cycle_{d}mm_R** : float
          Fraction of cardiac cycles with base-point displacement greater than `d` mm
          (computed for each threshold in `mistake_distances`).

    Notes
    -----
    - The function assumes that the segmentation contains left ventricle (LV) and myocardium (MYO) labels.
    - Uses `EchoMeasure._endo_base` to extract the mitral valve base points.
    - If spacing is not isometric (difference > 0.001 mm), a `ValueError` is raised.
    - Frames where extraction fails are penalized with maximal errors and counted as mistakes.

    See Also
    --------
    EchoMeasure._endo_base : Extracts endocardial base points for the LV.
    estimate_num_cycles : Estimates the number of cardiac cycles based on LV area variation.

    """
    mae = []
    mse = []
    mistakes = dict((f"mistake_per_cycle_{d}mm", 0) for d in mistake_distances)
    mistakes.update(dict((f"mistake_per_cycle_{d}mm_L", 0) for d in mistake_distances))
    mistakes.update(dict((f"mistake_per_cycle_{d}mm_R", 0) for d in mistake_distances))

    lv_area = EchoMeasure.structure_area(gt, labels=1)
    n_cardiac_cycles, _, _ = estimate_num_cycles(lv_area)

    for i in range(len(gt)):
        try:
            lv_points = np.asarray(
                EchoMeasure._endo_base(gt[i].T, lv_labels=Label.LV, myo_labels=Label.MYO))
            p_points = np.asarray(
                EchoMeasure._endo_base(batch_segmentation[i].T, lv_labels=Label.LV, myo_labels=Label.MYO))
            mae_values = [np.linalg.norm(lv_points[0] - p_points[0]), np.linalg.norm(lv_points[1] - p_points[1])]
            mae += [mae_values]
            mse += [((lv_points - p_points) ** 2).mean()]

            for dist in mistake_distances:
                if abs(spacing[0] - spacing[1]) > 0.001:  # account for very close but not exactly same
                    raise ValueError("Spacing not isometric, not currently handled")
                num_pixels = dist / spacing[1]

                # left
                if mae_values[0] > num_pixels:
                    mistakes[f"mistake_per_cycle_{dist}mm_L"] += 1
                # right
                if mae_values[1] > num_pixels:
                    mistakes[f"mistake_per_cycle_{dist}mm_R"] += 1
                # either
                if (mae_values > num_pixels).any():
                    mistakes[f"mistake_per_cycle_{dist}mm"] += 1

        except Exception as e:
            print(f"LM exception: {e}")
            mae += [[batch_segmentation.shape[-1], batch_segmentation.shape[-1]]]
            mse += [batch_segmentation.shape[-1] ** 2]
            for k in mistakes.keys():
                mistakes[k] += 1

    # normalize by number of cycles
    for k in mistakes.keys():
        mistakes[k] /= n_cardiac_cycles

    mae = np.asarray(mae)
    if return_mean:
        metrics = {"mse": np.asarray(mse).mean(), "mae_L": mae[..., 0].mean(), "mae_R": mae[..., 1].mean(),
                   "mae": mae.mean()}
    else:
        metrics = {"mse": np.asarray(mse), "mae_L": mae[..., 0], "mae_R": mae[..., 1], "mae": mae}
    metrics.update(mistakes)
    return metrics
