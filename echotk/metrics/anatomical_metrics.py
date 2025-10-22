import os
import warnings
from itertools import repeat
from multiprocessing import Pool

import torch

from echotk.metrics.anatomical.utils import check_segmentation_validity
from echotk.metrics.utils.config import Label

warnings.simplefilter(action='ignore', category=FutureWarning)

def check_frame_anatomical_validity(frame, voxelspacing, labels=[Label.BG, Label.LV, Label.MYO]):
    """
    Check whether a single segmentation frame is anatomically valid.

    This function wraps `check_segmentation_validity` and handles potential
    exceptions gracefully, returning `0` if validity cannot be determined.

    Parameters
    ----------
    frame : np.ndarray or torch.Tensor
        Segmentation mask for a single frame.
    voxelspacing : tuple of float
        Physical spacing (in mm) between pixels along each spatial dimension.
    labels : list of int
        List of label IDs to consider in the anatomical validity check.

    Returns
    -------
    int
        `1` if the segmentation is anatomically valid, `0` otherwise.
    """
    try:
        return int(check_segmentation_validity(frame, voxelspacing, labels))
    except Exception as e:
        print(e)
        return 0


def is_anatomically_valid(output, voxelspacing=(1.0, 1.0), labels=[Label.BG, Label.LV, Label.MYO]):
    """
    Evaluate anatomical validity for each frame in a segmentation sequence.

    Parameters
    ----------
    output : list or torch.Tensor
        Sequence of segmentation masks (e.g., cardiac cycle frames).
    voxelspacing : tuple of float, optional
        Physical spacing (in mm) between pixels along each spatial dimension.
        Default is `(1.0, 1.0)`.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(len(output),)` containing `1` for valid frames and `0` for invalid ones.
    """
    out = []
    for i in range(len(output)):
        out += [check_frame_anatomical_validity(output[i], voxelspacing, labels)]
    return out


def is_anatomically_valid_multiproc(output,
                                    voxel_spacing=(1.0, 1.0),
                                    num_proc=os.cpu_count() - 1,
                                    labels=[Label.BG, Label.LV, Label.MYO]):
    """
    Evaluate anatomical validity for each frame using multiprocessing.

    This version parallelizes the per-frame validity checks to speed up
    processing of large segmentation sequences.

    Parameters
    ----------
    output : list or torch.Tensor
        Sequence of segmentation masks to validate.
    voxel_spacing : tuple of float, optional
        Physical spacing (in mm) between pixels along each spatial dimension.
        Default is `(1.0, 1.0)`.
    num_proc : int, optional
        Number of processes to use. Defaults to `os.cpu_count() - 1`.

    Returns
    -------
    list of int
        List containing `1` for anatomically valid frames and `0` otherwise.
    """
    segmentations = [output[i].T for i in range(len(output))]
    with Pool(processes=num_proc) as pool:
        out = list(
            pool.starmap(
                check_frame_anatomical_validity,
                zip(
                    segmentations,
                    repeat(voxel_spacing),
                    repeat([0, 1, 2])
                )
            )
        )
    return out

