import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.morphology import binary_erosion

from echotk.sector_tools.extract_sector import calculate_variance
from echotk.sector_tools.ransac_utils import ransac_sector_extraction


def total_mask_area_ratio(mask):
    """
    Measure ratio of mask area over size of image
    """
    if len(mask.shape) == 3:
        return mask.sum() / (mask.shape[0] * mask.shape[1] * mask.shape[2])
    elif len(mask.shape) == 2:
        return mask.sum() / (mask.shape[0] * mask.shape[1])
    else:
        raise ValueError('Invalid mask array shape')


def measure_signal_lost(img, mask, plot=False):
    """
    Measure amount of signal lost from applying mask
    """

    var = calculate_variance(img.astype(np.int32))

    # make mask 3d if needed and apply
    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]
        mask = np.repeat(mask, img.shape[2], axis=-1)
    img[mask] = 0

    # remove pixels with negligible variance and unimportant location
    var_3d = var[:, :, np.newaxis]
    var_3d = np.repeat(var_3d, img.shape[2], axis=-1)
    img[(var_3d <= np.quantile(var, 0.65))] = 0  # 65% seems to remove all annotations and keep most signal pixels
    img[:, int(img.shape[1]*0.85):, :] = 0  # ignore diff in bottom 10%

    if plot:
        plt.figure()
        plt.title("Signal lost: masked image areas with high variance")
        plt.imshow(img[:, :, 0].T)

        plt.figure()
        plt.title("Signal lost: high variance areas")
        plt.imshow(var.T)
        plt.show()

    # percent of pixels containing signal that were lost with mask application
    return np.count_nonzero(img) / (np.count_nonzero(var_3d > np.quantile(var, 0.65)) * 100 + 1e-9)


def compute_mask_diff(mask1, mask2, plot=False):
    """
    Compare difference in mask coverage
    """
    # where are masks different
    d = np.zeros_like(mask1)
    d[mask1 != mask2] = 1

    # erosion to remove thin lines that are essentially the same mask
    d = binary_erosion(d)

    if plot:
        plt.figure()
        plt.title("Mask diff: difference in mask coverage")
        plt.imshow(d.T)
        plt.show()

    # percentage of pixels different in both masks
    return d.sum() / mask2.sum() * 100


def measure_remaining_annotations(img, mask, plot=False):
    """
    Use variance calculation to measure remaining annotations within mask
    """
    var = calculate_variance(img.astype(np.int32))
    var /= (var.max() + 1e-9)
    e = np.zeros_like(var)
    e[(var <= np.quantile(var, 0.05)) & (img.mean(axis=2) >= (0.5 * img.max()))] = 1
    e[~mask] = 0

    if plot:
        plt.figure()
        plt.title("Remaining annotations: input image")
        plt.imshow(img.mean(axis=2).T)

        plt.figure()
        plt.title("Remaining annotations: low variance areas within mask")
        plt.imshow(img[:, :, 0].T)
        plt.imshow(e.T, alpha=0.7, cmap='jet')
        plt.show()

    return (e == 1).sum() / mask.sum() * 100


def ransac_sector_w_metrics(mask, img, plot=False):
    # get 2d version of mask
    flat_mask = mask.sum(axis=2)
    flat_mask[flat_mask != 0] = 1

    # Use only the largest blob, makes sure that edges are not inflated by useless extra blob
    # Find each blob in the image
    lbl, num = ndimage.label(flat_mask)
    # Count the number of elements per label
    count = np.bincount(lbl.flat)
    if not np.any(count[1:]):
        return mask
    # Select the largest blob
    maxi = np.argmax(count[1:]) + 1
    # Remove the other blobs
    lbl[lbl != maxi] = 0

    try:
        # ransac mask from flattened nn mask
        r_mask, param_dict = ransac_sector_extraction(lbl, slim_factor=0.0075, use_convolutions=False, constrain_angles=False, plot=plot)
    except Exception as e:
        print(f"EXCEPTION creating ransac mask: {e}")
        r_mask = np.zeros_like(flat_mask)
        param_dict = None

    diff = compute_mask_diff(flat_mask, r_mask, plot=plot)
    ratio = total_mask_area_ratio(r_mask)
    annot = measure_remaining_annotations(img, r_mask, plot=plot)
    sig = measure_signal_lost(img, r_mask, plot=plot)

    r_mask = r_mask[:, :, np.newaxis]
    r_mask = np.repeat(r_mask, img.shape[2], axis=-1)
    return r_mask, diff, ratio, annot, sig, param_dict
