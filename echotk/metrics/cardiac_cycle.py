import numpy as np
from scipy.signal import find_peaks


def extract_cycle_points(lv_area, pad_width=10, peaks_prominence=0.15, peaks_height=0.5, peaks_distance=10):
    """
    Detect systolic peaks and diastolic valleys in the LV area signal.

    inspired by code from
    https://github.com/nathanpainchaud/vital/blob/feature/data/cardinal/vital/data/cardinal/clip_cycle.py

    Parameters
    ----------
    lv_area : np.ndarray
        LV cavity area over time.
    pad_width : int, optional
        Padding applied to handle edge peaks. Default is 10.
    peaks_prominence, peaks_height : float, optional
        Relative prominence and height thresholds for peak detection.
    peaks_distance : int, optional
        Minimum distance between peaks. Default is 10.

    Returns
    -------
    peaks, valleys : list of int
        Indices of detected peaks and valleys.
    """

    # pad to consider peaks at ends
    lv_area_pad = np.pad(lv_area, pad_width, mode="reflect")
    lv_area_pad[:pad_width] -= 1
    lv_area_pad[-pad_width:] -= 1

    min_max_diff = np.max(lv_area) - np.min(lv_area)
    peaks_height = np.min(lv_area) + (peaks_height * min_max_diff)
    peaks_prominence = min_max_diff * peaks_prominence

    peaks, _ = find_peaks(lv_area_pad, distance=peaks_distance, height=peaks_height, prominence=peaks_prominence)
    valleys, _ = find_peaks(-lv_area_pad, distance=peaks_distance, height=-peaks_height, prominence=-peaks_prominence)

    # remove padding
    peaks -= pad_width
    peaks = [p for p in peaks if len(lv_area) > p >= 0]

    valleys -= pad_width
    valleys = [v for v in valleys if len(lv_area) > v >= 0]

    return peaks, valleys


def estimate_num_cycles(lv_area, pad_width=10, peaks_prominence=0.15, peaks_height=0.5, peaks_distance=10, plot=False):
    """
    Estimate the number of cardiac cycles from the LV area curve.

    Parameters
    ----------
    lv_area : np.ndarray
        LV cavity area over time.
    pad_width, peaks_prominence, peaks_height, peaks_distance : optional
        Parameters controlling peak and valley detection.
    plot : bool, optional
        If True, plot the detected peaks and valleys.

    Returns
    -------
    num_cycles : int
        Estimated number of cardiac cycles.
    peaks, valleys : list of int
        Indices of detected peaks and valleys.
    """
    peaks, valleys = extract_cycle_points(lv_area, pad_width, peaks_prominence, peaks_height, peaks_distance)
    num_cycles = max((len(peaks) - (len(peaks) - len(valleys))), 1)  # max to make sure at least one cycle

    if plot:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(lv_area)
        for p in peaks:
            plt.scatter(p, lv_area[p], marker='x', c='red')

        for v in valleys:
            plt.scatter(v, lv_area[v], marker='o', c='green')
        plt.title(f"{num_cycles}")
        plt.show()

    return num_cycles, peaks, valleys


