import numpy as np
from typing import Tuple, TypeVar, Union

from torch import Tensor

T = TypeVar("T", np.ndarray, Tensor)


def minmax_scaling(data: T, bounds: Tuple[Union[float, T], Union[float, T]] = None) -> T:
    """Standardizes data w.r.t. predefined min/max bounds, if provided, or its own min/max otherwise.

    Args:
        data: Data to scale.
        bounds: Prior min and max bounds to use to scale the data.

    Returns:
        Data scaled w.r.t. the predefined or computed bounds.
    """
    # If no prior min/max bounds for the data are provided, compute them from its values
    if bounds is None:
        min, max = data.min(), data.max()
    else:
        min, max = bounds
    return (data - min) / (max - min)


def check_temporal_consistency_errors(threshold: float, *args, **kwargs) -> np.ndarray:
    """Identifies instants where the temporal consistency metric exceed a certain threshold value.

    Args:
        threshold: The maximum value above which the absolute value of the temporal consistency metric flags the instant
            as temporally inconsistent.
        *args: Positional arguments to pass to the temporal consistency metric computation.
        **kwargs: Keyword arguments to pass to the temporal consistency metric computation.

    Returns:
        temporal_errors: (n_samples,) Whether each instant is temporally inconsistent (`True`) or not (`False`).
    """
    return np.abs(compute_temporal_consistency_metric(*args, **kwargs)) > threshold


def compute_temporal_consistency_metric(attribute: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Computes the error between attribute values and the interpolation between their previous/next neighbors.

    Args:
        attribute: (n_samples, [1]), The 1D signal to analyze for temporal inconsistencies between instants.
        *args: Positional arguments to pass to the scaling function.
        **kwargs: Keyword arguments to pass to the scaling function.

    Returns:
        metric: (n_samples,) Error between attribute values and the interpolation between their previous/next neighbors.
    """
    attribute = minmax_scaling(attribute, *args, **kwargs)

    # Compute the temporal consistency metric
    prev_neigh = attribute[:-2]  # Previous neighbors of non-edge instants
    next_neigh = attribute[2:]  # Next neighbors of non-edge instants
    neigh_inter_diff = attribute[1:-1] - ((prev_neigh + next_neigh) / 2)
    # Pad edges with 0; since edges are missing a neighbor for the interpolation,
    # they are considered temporally consistent by default
    pad_width = [(1, 1)] + [(0, 0)] * (attribute.ndim - 1)
    neigh_inter_diff = np.pad(neigh_inter_diff, pad_width, constant_values=0)

    return neigh_inter_diff
