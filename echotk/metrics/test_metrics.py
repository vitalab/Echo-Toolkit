import time
from tabnanny import verbose
from typing import Tuple

import numpy as np
import torch
from humanize import metric
from medpy.metric import dc, hd

from echotk.metrics.anatomical_metrics import is_anatomically_valid
from echotk.metrics.landmark_metrics import mitral_valve_distance
from echotk.metrics.temporal_metrics import check_temporal_validity
from echotk.metrics.utils.config import LabelEnum, Label


def dice(pred: np.ndarray, target: np.ndarray, labels: Tuple[LabelEnum], exclude_bg: bool = True,
         all_classes: bool = False):
    """
    Compute dice for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)
        labels: List of labels for which to compute the dice.
        exclude_bg: If true, background dice is not considered.

    Returns:
        mean dice,
        per-class dice
    """
    dices = []
    if len(labels) > 2:
        for label in labels:
            if exclude_bg and label == 0:
                pass
            else:
                pred_mask, gt_mask = np.isin(pred, label), np.isin(target, label)
                dices.append(dc(pred_mask, gt_mask))
        if all_classes:
            dice_dict = {f"dice/{label.name}": dice for label, dice in zip(labels[1:], dices)}
            dice_dict['Dice'] = np.array(dices).mean()
            return dice_dict
        else:
            return np.array(dices).mean()
    else:
        if all_classes:
            return {'Dice': dc(pred.squeeze(), target.squeeze())}
        else:
            return dc(pred.squeeze(), target.squeeze())


def hausdorff(pred: np.ndarray, target: np.ndarray, labels: Tuple[LabelEnum], exclude_bg: bool = True,
         all_classes: bool = False, voxel_spacing: Tuple[float] = None):
    """Compute hausdorff for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)
        labels: List of labels for which to compute the metric.
        exclude_bg: If true, background dice is not considered.

    Returns:
        hausdorff
        per-class hausdorff
    """
    hd_dict = {}
    for i in range(len(pred)):
        hausdorffs = []
        for label in labels:
            if exclude_bg and label == 0:
                pass
            else:
                pred_mask, gt_mask = np.isin(pred[i], label), np.isin(target[i], label)
                if pred_mask.sum() == 0:
                    pred_mask[pred_mask.shape[0] // 2, pred_mask.shape[1] // 2] = True
                    print('empty mask')
                hausdorffs.append(hd(pred_mask, gt_mask, voxel_spacing if voxel_spacing is not None else None))
        if all_classes:
            for label, haus in zip(labels[1:], hausdorffs):
                hd_dict[f"hd/{label.name}"] = hd_dict.get(f"hd/{label.name}", 0) + haus / len(pred)
            hd_dict['Hausdorff'] = hd_dict.get('Hausdorff', 0) + np.array(hausdorffs).mean() / len(pred)
        else:
            hd_dict['Hausdorff'] = hd_dict.get('Hausdorff', 0) + np.array(hausdorffs).mean() / len(pred)
    return hd_dict


def full_test_metrics(batchwise_3d_segmentation, batchwise_gt, voxel_spacing, device, prefix='', verbose=True):
    """
    Compute all evaluation metrics for a predicted segmentation sequence.

    Parameters
    ----------
    batchwise_3d_segmentation : np.ndarray
        Predicted segmentation sequence (T, H, W). T Can be 1 for 2D segmentation
    batchwise_gt : np.ndarray
        Ground-truth segmentation sequence (T, H, W).
    voxel_spacing : tuple of float
        Pixel spacing (mm) for spatial metrics.
    device : torch.device
        Target device for returned tensors.
    prefix : str, optional
        Prefix for metric keys (e.g., 'val' or 'test').
    verbose : bool, optional
        If True, print timing for each metric group.

    Returns
    -------
    dict
        Dictionary of evaluation metrics including Dice, Hausdorff,
        anatomical and temporal validity, and mitral valve distances.
    """
    if len(prefix) > 0:
        prefix = f"{prefix}/"
    logs = {} # To be filled in...

    start_time = time.time()
    test_dice = dice(batchwise_3d_segmentation, batchwise_gt, labels=(Label.BG, Label.LV, Label.MYO),
                     exclude_bg=True, all_classes=True)
    test_dice_epi = dice((batchwise_3d_segmentation != 0).astype(np.uint8), (batchwise_gt != 0).astype(np.uint8),
                         labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False)
    if verbose:
        print(f"Dice took {round(time.time() - start_time, 4)} (s).")
    logs.update({f"{prefix}{k}": v for k, v in test_dice.items()})
    logs.update({f"{prefix}dice/epi": test_dice_epi})

    start_time = time.time()
    test_hd = hausdorff(batchwise_3d_segmentation, batchwise_gt, labels=(Label.BG, Label.LV, Label.MYO),
                        exclude_bg=True, all_classes=True, voxel_spacing=voxel_spacing)
    test_hd_epi = hausdorff((batchwise_3d_segmentation != 0).astype(np.uint8), (batchwise_gt != 0).astype(np.uint8),
                            labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False,
                            voxel_spacing=voxel_spacing)['Hausdorff']
    if verbose:
        print(f"HD took {round(time.time() - start_time, 4)} (s).")
    logs.update({f"{prefix}{k}": v for k, v in test_hd.items()})
    logs.update({f"{prefix}hd/epi": test_hd_epi})

    start_time = time.time()
    anat_errors = is_anatomically_valid(batchwise_3d_segmentation)
    if verbose:
        print(f"AV took {round(time.time() - start_time, 4)} (s).")
    logs.update({
        f"{prefix}anat_valid": int(all(anat_errors)),
        f"{prefix}anat_valid_frames": np.mean(anat_errors)
    })

    # temporal metrics have no meaning below 3 consecutive frames
    # ignore otherwise
    # Ideally more are needed
    if len(batchwise_3d_segmentation) >= 3:
        start_time = time.time()
        temporal_valid, num_temporal_errors = check_temporal_validity(batchwise_3d_segmentation.transpose((0, 2, 1)),
                                                 voxel_spacing)
        if verbose:
            print(f"TV took {round(time.time() - start_time, 4)} (s).")
        logs.update({
            f"{prefix}temporal_valid": temporal_valid,
            f"{prefix}temporal_errors": num_temporal_errors
        })

    start_time = time.time()
    lm_metrics = mitral_valve_distance(batchwise_3d_segmentation, batchwise_gt, voxel_spacing)
    if verbose:
        print(f"LM dist took {round(time.time() - start_time, 4)} (s).")
    logs.update({f"{prefix}LM/{k}": v for k, v in lm_metrics.items()})

    return logs


if __name__ == "__main__":
    import nibabel as nib

    reference_nii = nib.load('./../../data/examples/segmentation_reference.nii.gz')
    reference = reference_nii.get_fdata().transpose((2, 0, 1)) # metrics expect T H W (time as batch)

    # sanity check, should all be perfect
    metrics_dict = full_test_metrics(reference, reference, voxel_spacing=reference_nii.header['pixdim'][1:3], device='cpu', verbose=False)
    print(metrics_dict)

    # good segmentation
    candidate_nii = nib.load('./../../data/examples/segmentation_candidate1.nii.gz') # good one
    candidate = candidate_nii.get_fdata().transpose((2, 0, 1)) # metrics expect T H W (time as batch)
    metrics_dict = full_test_metrics(candidate, reference, voxel_spacing=reference_nii.header['pixdim'][1:3], device='cpu', verbose=False)
    print(metrics_dict)

    # not so good
    candidate_nii = nib.load('./../../data/examples/segmentation_candidate2.nii.gz') # not so good
    candidate = candidate_nii.get_fdata().transpose((2, 0, 1)) # metrics expect T H W (time as batch)
    metrics_dict = full_test_metrics(candidate, reference, voxel_spacing=reference_nii.header['pixdim'][1:3], device='cpu', verbose=False)
    print(metrics_dict)
