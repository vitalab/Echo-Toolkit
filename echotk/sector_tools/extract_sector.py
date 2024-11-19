import numpy as np
from matplotlib import pyplot as plt

from echotk.sector_tools.ransac_utils import ransac_sector_extraction


def calculate_variance(seq):
    """
    Return variance of each pixel in the sequence
    """
    seq_squared = np.square(seq, dtype=np.int32)
    var = seq_squared.sum(axis=2) - (1 / seq.shape[2] * np.square(seq.sum(axis=2), dtype=np.int32))
    return var


def apply_sector_extraction(im, plot=False):
    """Function to extract important region (triangle) of echo image"""

    var = calculate_variance(im)

    center = var[var.shape[0]//2 - var.shape[0]//8:var.shape[0]//2 + var.shape[0]//8,
                 var.shape[1]//2 - var.shape[1]//8:var.shape[1]//2 + var.shape[1]//8]
    thresh = np.quantile(center, 0.01)
    mask = (var > thresh)

    mask = ransac_sector_extraction(mask, plot=plot)

    validate_sector_area(mask)

    if plot:
        plt.figure()
        plt.imshow(im[:, :, 1].T)
        plt.imshow(mask.T, cmap='jet', alpha=0.5)
        plt.show()

    mask = mask[:, :, np.newaxis]
    mask = np.repeat(mask, im.shape[2], axis=-1)
    im[~mask] = 0

    return im, mask


def validate_sector_area(mask, fill_amount=0.4):
    total_area = mask.shape[1] * mask.shape[0]
    if (mask.sum() / total_area) < fill_amount:
        raise Exception(f"Mask is invalid, does not cover sufficient area in image, "
                        f"only covers {(mask.sum() / total_area)}%")


if __name__ == "__main__":

    var = np.load('var.npy')

    center = var[var.shape[0] // 2 - var.shape[0] // 8:var.shape[0] // 2 + var.shape[0] // 8,
             var.shape[1] // 2 - var.shape[1] // 8:var.shape[1] // 2 + var.shape[1] // 8]

    thresh = np.quantile(center, 0.01)
    print(thresh)
    mask = (var > thresh)

    ransac_sector_extraction(mask, plot=True)

