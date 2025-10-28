import functools
import itertools
import sys
from collections import deque
from numbers import Real
from typing import Callable, List, Literal, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image, erosion

from echotk.utils.config import SemanticStructureId, Label
from echotk.utils.decorators import auto_cast_data, batch_function
from echotk.utils.image import cart2pol, T


class Measure:
    """Generic implementations of various measures on images represented as numpy arrays or torch tensors."""

    @staticmethod
    @auto_cast_data
    def structure_area(segmentation: T, labels: SemanticStructureId = None, voxelarea: float = None) -> T:
        """Computes the number of pixels, in a segmentation map, associated to a structure.

        Args:
            segmentation: ([N], H, W), Segmentation in which to identify the number of pixels of the structure.
            labels: Labels of the classes that are part of the structure for which to count the number of pixels. If
                `None`, all truthy values will be considered part of the structure.
            voxelarea: Size of the mask's voxels along each (height, width) dimension (in mm).

        Returns:
            ([N]), Number of pixels associated to the structure, in each segmentation of the batch.
        """
        if labels:
            mask = np.isin(segmentation, labels)
        else:
            mask = segmentation.astype(bool)

        if voxelarea is None:
            voxelarea = 1
        return mask.sum((-2, -1)) * voxelarea

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def structure_center(segmentation: T, labels: SemanticStructureId = None, axis: int = None) -> T:
        """Computes the center of mass of a structure in a segmentation map.

        Args:
            segmentation: ([N], H, W), Segmentation in which to identify the center of mass of the structure.
            labels: Labels of the classes that are part of the structure for which to measure the center of mass. If
                `None`, all truthy values will be considered part of the structure.
            axis: Index of a dimension of interest, for which to get the center of mass. If provided, the value of the
                center of mass will only be returned for this axis. If `None`, the center of mass along all axes will be
                returned.

        Returns:
            ([N], [2]), Center of mass of the structure, for a specified axis or across all axes, in each segmentation
            of the batch.
        """
        if labels:
            mask = np.isin(segmentation, labels)
        else:
            mask = segmentation.astype(bool)

        center = ndimage.center_of_mass(mask)
        if any(np.isnan(center)):  # Default to the center of the image if the center of mass can't be found
            center = np.array(segmentation.shape) // 2
        if axis is not None:
            center = center[axis]
        return center

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def structure_orientation(segmentation: T, labels: SemanticStructureId = None, reference_orientation: int = 0) -> T:
        """Computes the angle w.r.t. a reference orientation of a structure in a segmentation map.

        Args:
            segmentation: ([N], H, W), Segmentation in which to identify the orientation of the structure.
            labels: Labels of the classes that are part of the structure for which to measure orientation. If `None`,
                all truthy values will be considered part of the structure.
            reference_orientation: Reference orientation, that would correspond to a returned orientation of `0` if the
                structure where aligned on it perfectly. By default, this orientation corresponds to the positive x
                axis.

        Returns:
            ([N]), Orientation of the structure w.r.t. the reference orientation, in each segmentation of the batch.
        """
        if labels:
            structure_mask = np.isin(segmentation, labels)
        else:
            structure_mask = segmentation.astype(bool)

        if np.any(structure_mask):  # If the structure is present in the segmentation
            # Get the right eigenvectors of the structure's mass
            structure_inertia_tensors = measure.inertia_tensor(structure_mask)
            _, evecs = np.linalg.eigh(structure_inertia_tensors)

            # Find the 1st eigenvector, that corresponds to the orientation of the structure's longest axis
            evec1 = evecs[-1]

            # Compute the rotation necessary to align it with the x-axis (horizontal)
            orientation = math.degrees(np.arctan2(evec1[1], evec1[0]))
            orientation -= reference_orientation  # Get angle with reference orientation from angle with x-axis
        else:  # If the structure is not present in the segmentation, consider it aligned to the reference by default
            orientation = 0
        return orientation

    @staticmethod
    @auto_cast_data
    def bbox(
        segmentation: T, labels: SemanticStructureId = None, bbox_margin: Real = 0.05, normalize: bool = False
    ) -> T:
        """Computes the coordinates of a bounding box (bbox) around a region of interest (ROI).

        Args:
            segmentation: ([N], H, W), Segmentation in which to identify the coordinates of the bbox.
            labels: Labels of the classes that are part of the ROI. If `None`, all truthy values will be considered part
                of the ROI.
            bbox_margin: Ratio by which to enlarge the bbox from the closest possible fit, so as to leave a slight
                margin at the edges of the bbox.
            normalize: If ``True``, normalizes the bbox coordinates from between 0 and H or W to between 0 and 1.

        Returns:
            ([N], 4), Coordinates of the bbox, in (y1, x1, y2, x2) format.
        """
        if labels:
            roi_mask = np.isin(segmentation, labels)
        else:
            roi_mask = segmentation.astype(bool)

        # Find the coordinates of the bounding box around the ROI
        rows = roi_mask.any(1)
        cols = roi_mask.any(0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Compute the size of the margin between the ROI and its bounding box
        dx = int(bbox_margin * (x2 - x1))
        dy = int(bbox_margin * (y2 - y1))

        # Apply margin to bbox coordinates
        y1, y2 = y1 - dy, y2 + dy + 1
        x1, x2 = x1 - dx, x2 + dx + 1

        # Check limits
        y1, y2 = max(0, y1), min(y2, roi_mask.shape[0])
        x1, x2 = max(0, x1), min(x2, roi_mask.shape[1])

        roi_bbox = np.array([y1, x1, y2, x2])

        if normalize:
            roi_bbox = roi_bbox.astype(float)
            roi_bbox[[0, 2]] = roi_bbox[[0, 2]] / segmentation.shape[0]  # Normalize height
            roi_bbox[[1, 3]] = roi_bbox[[1, 3]] / segmentation.shape[1]  # Normalize width

        return roi_bbox

    @staticmethod
    @auto_cast_data
    def denormalize_bbox(roi_bbox: T, output_size: Tuple[int, int], check_bounds: bool = False) -> T:
        """Gives the pixel-indices of a bounding box (bbox) w.r.t an output size based on the bbox's normalized coord.

        Args:
            roi_bbox: ([N], 4), Normalized coordinates of the bbox, in (y1, x1, y2, x2) format.
            output_size: (X, Y), Size for which to compute pixel-indices based on the normalized coordinates.
            check_bounds: If ``True``, perform various checks on the denormalized coordinates:
                - ensure they fit between 0 and X or Y
                - ensure that the min bounds are smaller than the max bounds
                - ensure that the bbox is at least one pixel wide in each dimension

        Returns:
            ([N], 4), Coordinates of the bbox, in (y1, x1, y2, x2) format.
        """
        # Copy input data to ensure we don't write over user data
        roi_bbox = np.copy(roi_bbox)

        if check_bounds:
            # Clamp predicted RoI bbox to ensure it won't end up out of range of the image
            roi_bbox = np.clip(roi_bbox, 0, 1)

        # Change ROI bbox from normalized between 0 and 1 to absolute pixel coordinates
        roi_bbox[:, (0, 2)] = (roi_bbox[:, (0, 2)] * output_size[0]).round()  # Y
        roi_bbox[:, (1, 3)] = (roi_bbox[:, (1, 3)] * output_size[1]).round()  # X

        if check_bounds:
            # Clamp predicted min bounds are at least two pixels smaller than image bounds
            # to allow for inclusive upper bounds
            roi_bbox[:, 0] = np.minimum(roi_bbox[:, 0], output_size[0] - 1)  # Y
            roi_bbox[:, 1] = np.minimum(roi_bbox[:, 1], output_size[1] - 1)  # X

            # Clamp predicted max bounds are at least one pixel bigger than min bounds
            roi_bbox[:, 2] = np.maximum(roi_bbox[:, 2], roi_bbox[:, 0] + 1)  # Y
            roi_bbox[:, 3] = np.maximum(roi_bbox[:, 3], roi_bbox[:, 1] + 1)  # X

        return roi_bbox


class EchoMeasure(Measure):
    """Implementation of various echocardiography-specific measures on images."""

    @staticmethod
    def _extract_landmarks_from_polar_contour(
        segmentation: np.ndarray,
        labels: SemanticStructureId,
        polar_smoothing_factor: float = 0,
        debug_plots: bool = False,
        apex: bool = True,
        base: bool = True,
    ) -> List[np.ndarray]:
        """Extracts a structure's landmarks that produce characteristic peaks in the polar projection of the contour.

        Args:
            segmentation: (H, W), Segmentation map.
            labels: Labels of the classes that are part of the structure for which to extract landmarks.
            polar_smoothing_factor: Multiplicative factor (for the number of points along the contour), to determine the
                standard deviation of a Gaussian kernel to smooth the projection of the contour points in polar
                coordinates.
            debug_plots: Whether to plot the peaks found in the projection of the contour points in polar coordinates +
                where the selected peaks map back on the segmentation. These plots should only be used for debugging the
                identification of the landmarks.
            apex: Whether to try and extract the apex of the structure.
            base: Whether to try and extract the left and right corners at the base of the structure.

        Returns:
            The coordinates of the structure's apex (if `apex==True`) and left and right corners at the base
            (if `base==True`).
        """
        structure_mask = np.isin(segmentation, labels)

        # Extract all the points on the contour of the structure of interest
        # Use `level=0.9` to force the contour to be closer to the structure of interest than the background
        contour = find_contours(structure_mask, level=0.9)[0]

        # Shift the contour, so it's centered around the center of mass of the structure
        contour_centered = contour - ndimage.center_of_mass(structure_mask)

        # Obtain the projection of the contour in polar coordinates
        theta, rho, sort_indices = cart2pol(contour_centered[:, 1], contour_centered[:, 0], sort_by_theta=True)

        if polar_smoothing_factor:
            # Smooth the signal to avoid finding peaks for small localities
            rho = gaussian_filter1d(rho, len(contour) * polar_smoothing_factor)

        # Detect peaks that correspond to endo/epi base and apex
        peaks, properties = find_peaks(rho, height=0)
        peak_heights = properties["peak_heights"]

        landmarks_polar_indices = []

        if apex:
            # Discard base peaks by only keeping peaks found in the upper half of the mask
            # (by discarding peaks found where theta < 0)
            apex_peaks_mask = theta[peaks] < 0
            apex_peaks = peaks[apex_peaks_mask]
            apex_peak_heights = peak_heights[apex_peaks_mask]

            if not len(apex_peaks):
                raise RuntimeError("Unable to identify the apex of the endo/epi.")

            # Keep only the highest peak as the peak corresponding to the apex
            landmarks_polar_indices.append(apex_peaks[apex_peak_heights.argmax()])

        if base:
            # Discard apex peak by only keeping peaks found in the lower half of the mask
            # (by discarding peaks found where theta > 0)
            base_peaks_mask = theta[peaks] > 0
            base_peaks = peaks[base_peaks_mask]
            base_peak_heights = peak_heights[base_peaks_mask]

            if (num_peaks := len(base_peaks)) < 2:
                raise RuntimeError(
                    f"Identified {num_peaks} corner(s) for the endo/epi base. We needed to find at least 2 corners to "
                    f"identify the corners at the base of the endo/epi."
                )

            # Identify the indices of the 2 highest peaks in the list of peaks
            base_highest_peaks = base_peak_heights.argsort()[-2:]
            # Sort the indices of the 2 highest peaks to make sure they stay ordered by descending theta
            # (so that the peak of the left corner comes first) regardless of their heights
            base_highest_peaks = sorted(base_highest_peaks, reverse=True)

            landmarks_polar_indices.extend(base_peaks[base_highest_peaks])

        if debug_plots:
            # Display contour curve in polar coordinates
            import seaborn as sns
            from matplotlib import pyplot as plt

            with sns.axes_style("darkgrid"):
                plot = sns.lineplot(data=pd.DataFrame({"theta": theta, "rho": rho}), x="theta", y="rho")

            # Annotate the peaks with their respective index
            for peak_idx, peak in enumerate(peaks):
                plot.annotate(f"{peak_idx}", (theta[peak], rho[peak]), xytext=(1, 4), textcoords="offset points")

            # Plot lines pointing to the peaks to make them more visible
            plot.vlines(x=theta[peaks], ymin=rho.min(), ymax=rho[peaks], linestyles="dashed")

            plt.show()

        # Map the indices of the peaks in polar coordinates back to the indices in the list of contour points
        contour_indices = sort_indices[landmarks_polar_indices]
        landmarks = contour[contour_indices]

        if debug_plots:
            plt.imshow(structure_mask)
            for landmark in landmarks:
                plt.scatter(landmark[1], landmark[0], c="r", marker="o", s=3)
            plt.show()

        return landmarks

    @staticmethod
    def _endo_epi_contour(
        segmentation: np.ndarray,
        labels: SemanticStructureId,
        base_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Lists points on the contour of the endo/epi (excluding the base), from the left of the base to its right.

        Args:
            segmentation: (H, W), Segmentation map.
            labels: Labels of the classes that are part of the endocardium/epicardium.
            base_fn: Function that identifies the left and right corners at the base of the endocardium/epicardium in a
                segmentation mask.

        Returns:
            Coordinates of points on the contour of the endo/epi (excluding the base), from the left of the base to its
            right.
        """
        structure_mask = np.isin(segmentation, labels)

        # Identify the left/right markers at the base of the endo/epi
        left_corner, right_corner = base_fn(segmentation)

        # Extract all the points on the contour of the structure of interest
        # Use `level=0.9` to force the contour to be closer to the structure of interest than the background
        contour = find_contours(structure_mask, level=0.9)[0]

        # Shift the contour so that they start at the left corner
        # To detect the contour coordinates that match the corner, we use the closest match since skimage's
        # `find_contours` coordinates are interpolated between pixels, so they won't match exactly corner coordinates
        dist_to_left_corner = np.linalg.norm(left_corner - contour, axis=1)
        left_corner_contour_idx = np.argmin(dist_to_left_corner)
        contour = np.roll(contour, -left_corner_contour_idx, axis=0)

        # Filter the full contour to discard points along the base
        # We implement this by slicing the contours from the left corner to the right corner, since the contour returned
        # by skimage's `find_contours` is oriented clockwise
        dist_to_right_corner = np.linalg.norm(right_corner - contour, axis=1)
        right_corner_contour_idx = np.argmin(dist_to_right_corner)
        contour_without_base = contour[: right_corner_contour_idx + 1]

        return contour_without_base

    @staticmethod
    def _endo_base(
        segmentation: np.ndarray, lv_labels: SemanticStructureId, myo_labels: SemanticStructureId
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Finds the left/right markers at the base of the endocardium.

        Notes:
            - This implementation exists because it is more reliable for the endo than the more general algorithm that
              tries to identify peaks in the polar projection of a contour. As such, in cases where the latter gives
              acceptable results, it should be preferred.

        Args:
            segmentation: (H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the left ventricle.

        Returns:
            Coordinates of the left/right markers at the base of the endocardium.
        """
        struct = ndimage.generate_binary_structure(2, 2)
        left_ventricle = np.isin(segmentation, lv_labels)
        myocardium = np.isin(segmentation, myo_labels)
        others = ~(left_ventricle + myocardium)
        dilated_myocardium = ndimage.binary_dilation(myocardium, structure=struct)
        dilated_others = ndimage.binary_dilation(others, structure=struct)
        y_coords, x_coords = np.nonzero(left_ventricle * dilated_myocardium * dilated_others)

        if (num_markers := len(y_coords)) < 2:
            raise RuntimeError(
                f"Identified {num_markers} marker(s) at the edges of the left ventricle/myocardium frontier. We need "
                f"to identify at least 2 such markers to determine the base of the left ventricle."
            )

        if np.all(x_coords == x_coords.mean()):
            # Edge case where the base points are aligned vertically
            # Divide frontier into bottom and top halves.
            coord_mask = y_coords > y_coords.mean()
            left_point_idx = y_coords[coord_mask].argmin()
            right_point_idx = y_coords[~coord_mask].argmax()
        else:
            # Normal case where there is a clear divide between left and right markers at the base
            # Divide frontier into left and right halves.
            coord_mask = x_coords < x_coords.mean()
            left_point_idx = y_coords[coord_mask].argmax()
            right_point_idx = y_coords[~coord_mask].argmax()
        return (
            np.array([y_coords[coord_mask][left_point_idx], x_coords[coord_mask][left_point_idx]]),
            np.array([y_coords[~coord_mask][right_point_idx], x_coords[~coord_mask][right_point_idx]]),
        )

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def endo_epi_control_points(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        structure: Literal["endo", "epi"],
        num_control_points: int,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Lists uniformly distributed control points along the contour of the endocardium/epicardium.

        Args:
            segmentation: (H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            structure: Structure for which to identify the control points.
            num_control_points: Number of control points to sample along the contour of the endocardium/epicardium. The
                number of control points should be odd to be divisible evenly between the base -> apex and apex -> base
                segments.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            Coordinates of the control points along the contour of the endocardium/epicardium.
        """
        voxelspacing = np.array(voxelspacing)

        # "Backend" function used to find the corners at the base of the structure depends on the structure
        match structure:
            case "endo":
                struct_labels = lv_labels
                base_fn = functools.partial(EchoMeasure._endo_base, lv_labels=lv_labels, myo_labels=myo_labels)
            case "epi":
                struct_labels = [lv_labels, myo_labels]
                base_fn = functools.partial(
                    EchoMeasure._extract_landmarks_from_polar_contour,
                    labels=struct_labels,
                    polar_smoothing_factor=5e-3,  # 5e-3 was determined empirically
                    apex=False,
                )
            case _:
                raise ValueError(f"Unexpected value for 'mode': {structure}. Use either 'endo' or 'epi'.")

        # Find the points along the contour of the endo/epi excluding the base
        contour = EchoMeasure._endo_epi_contour(segmentation, struct_labels, base_fn)

        # Identify the apex from the points within the contour
        apex = EchoMeasure._extract_landmarks_from_polar_contour(
            segmentation, struct_labels, polar_smoothing_factor=5e-2, base=False  # 5e-2 was determined empirically
        )[0]

        # Round the contour's coordinates, so they don't fall between pixels anymore
        contour = contour.round().astype(int)

        # Break the contour down into independent segments (base -> apex, apex -> base) along which to uniformly
        # distribute control points
        apex_idx_in_contour = np.linalg.norm((contour - apex) * voxelspacing, axis=1).argmin()
        segments = [0, apex_idx_in_contour, len(contour) - 1]

        if (num_control_points - 1) % (num_segments := len(segments) - 1):
            raise ValueError(
                f"The number of requested control points: {num_control_points}, cannot be divided evenly across the "
                f"{num_segments} contour segments. Please set a number of control points that, when subtracted by 1, "
                f"is divisible by {num_segments}."
            )
        num_control_points_per_segment = (num_control_points - 1) // num_segments

        # Simplify the general case for handling th
        control_points_indices = [0]
        for segment_start, segment_stop in itertools.pairwise(segments):
            # Slice segment so that both the start and stop points are included in the segment
            segment = contour[segment_start : segment_stop + 1]

            # Compute the geometric distances between each point along the segment and the previous point.
            # This allows to then simply compute the cumulative distance from the left corner to each segment point
            segment_dist_to_prev = [0.0] + [
                np.linalg.norm((p1 - p0) * voxelspacing) for p0, p1 in itertools.pairwise(segment)
            ]
            segment_cum_dist = np.cumsum(segment_dist_to_prev)

            # Select points along the segment that are equidistant (by selecting points that are closest to where
            # steps of `perimeter / num_control_points` would expect to find points)
            control_points_step = np.linspace(0, segment_cum_dist[-1], num=num_control_points_per_segment + 1)
            segment_control_points = [
                segment_start + np.argmin(np.abs(point_cum_dist - segment_cum_dist))
                for point_cum_dist in control_points_step
            ]
            # Skip the first control point in the current segment, because its already included as the last control
            # point of the previous segment
            control_points_indices += segment_control_points[1:]

        return contour[control_points_indices]

    @staticmethod
    @auto_cast_data
    def gls(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Global Longitudinal Strain (GLS) for each frame in the sequence, compared to the first frame.

        Args:
            segmentation: (N, H, W), Segmentation map for a whole sequence, where the first frame is assumed to be an
                ED instant.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            (N,), Global Longitudinal Strain (GLS) curve, where the value (in percentage) is the relative difference
            in length of the endocardium compared to the first frame.
        """
        voxelspacing = np.array(voxelspacing)

        def _lv_longitudinal_length(frame: np.ndarray) -> float:
            # Find the points along the contour of the LV excluding the base
            contour = EchoMeasure._endo_epi_contour(
                frame, lv_labels, functools.partial(EchoMeasure._endo_base, lv_labels=lv_labels, myo_labels=myo_labels)
            )

            # Compute the perimeter as the sum of distances between each point along the contour and the previous one
            return sum(np.linalg.norm((p1 - p0) * voxelspacing) for p0, p1 in itertools.pairwise(contour))

        # Compute the longitudinal length of the LV for each frame in the sequence
        lv_longitudinal_lengths = np.array([_lv_longitudinal_length(frame) for frame in segmentation])

        # Compute the GLS for each frame in the sequence
        ed_lv_longitudinal_length = lv_longitudinal_lengths[0]
        gls = ((lv_longitudinal_lengths - ed_lv_longitudinal_length) / ed_lv_longitudinal_length) * 100

        return gls

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def lv_base_width(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Measures the distance between the left and right markers at the base of the left ventricle.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            ([N]), Distance between the left and right markers at the base of the left ventricle, or NaNs for the
            images where those 2 points cannot be reliably estimated.
        """
        voxelspacing = np.array(voxelspacing)

        # Identify the base of the left ventricle
        left_corner, right_corner = EchoMeasure._endo_base(segmentation, lv_labels, myo_labels)

        # Compute the distance between the points at the base
        return np.linalg.norm((left_corner - right_corner) * voxelspacing)

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def lv_length(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Measures the LV length as the distance between the base's midpoint and the apex.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            ([N]), Length of the left ventricle.
        """
        voxelspacing = np.array(voxelspacing)

        # Identify major landmarks of the left ventricle (i.e. base corners, base's midpoint and apex)
        left_corner, right_corner = EchoMeasure._endo_base(segmentation, lv_labels, myo_labels)
        base_mid = (left_corner + right_corner) / 2
        apex = EchoMeasure._extract_landmarks_from_polar_contour(
            segmentation, lv_labels, polar_smoothing_factor=5e-2, base=False
        )[0]

        # Compute the distance between the apex and the base's midpoint
        return np.linalg.norm((apex - base_mid) * voxelspacing)

class ContourMeasure:

    @staticmethod
    def _extract_landmarks_from_polar_contour(
        segmentation: np.ndarray,
        labels: SemanticStructureId,
        polar_smoothing_factor: float = 0,
        debug_plots: bool = False,
        apex: bool = True,
        base: bool = True,
    ) -> List[np.ndarray]:
        """Extracts a structure's landmarks that produce characteristic peaks in the polar projection of the contour.

        Args:
            segmentation: (H, W), Segmentation map.
            labels: Labels of the classes that are part of the structure for which to extract landmarks.
            polar_smoothing_factor: Multiplicative factor (for the number of points along the contour), to determine the
                standard deviation of a Gaussian kernel to smooth the projection of the contour points in polar
                coordinates.
            debug_plots: Whether to plot the peaks found in the projection of the contour points in polar coordinates +
                where the selected peaks map back on the segmentation. These plots should only be used for debugging the
                identification of the landmarks.
            apex: Whether to try and extract the apex of the structure.
            base: Whether to try and extract the left and right corners at the base of the structure.

        Returns:
            The coordinates of the structure's apex (if `apex==True`) and left and right corners at the base
            (if `base==True`).
        """
        structure_mask = np.isin(segmentation, labels)

        # Extract all the points on the contour of the structure of interest
        # Use `level=0.9` to force the contour to be closer to the structure of interest than the background
        contour = find_contours(structure_mask, level=0.9)[0]

        # Shift the contour, so it's centered around the center of mass of the structure
        contour_centered = contour - ndimage.center_of_mass(structure_mask)

        # Obtain the projection of the contour in polar coordinates
        theta, rho, sort_indices = cart2pol(contour_centered[:, 1], contour_centered[:, 0], sort_by_theta=True)

        if polar_smoothing_factor:
            # Smooth the signal to avoid finding peaks for small localities
            rho = gaussian_filter1d(rho, len(contour) * polar_smoothing_factor)

        # Detect peaks that correspond to endo/epi base and apex
        peaks, properties = find_peaks(rho, height=0)
        peak_heights = properties["peak_heights"]

        landmarks_polar_indices = []

        if apex:
            # Discard base peaks by only keeping peaks found in the upper half of the mask
            # (by discarding peaks found where theta < 0)
            apex_peaks_mask = theta[peaks] < 0
            apex_peaks = peaks[apex_peaks_mask]
            apex_peak_heights = peak_heights[apex_peaks_mask]

            if not len(apex_peaks):
                raise RuntimeError("Unable to identify the apex of the endo/epi.")

            # Keep only the highest peak as the peak corresponding to the apex
            landmarks_polar_indices.append(apex_peaks[apex_peak_heights.argmax()])

        if base:
            # Discard apex peak by only keeping peaks found in the lower half of the mask
            # (by discarding peaks found where theta > 0)
            base_peaks_mask = theta[peaks] > 0
            base_peaks = peaks[base_peaks_mask]
            base_peak_heights = peak_heights[base_peaks_mask]

            if (num_peaks := len(base_peaks)) < 2:
                raise RuntimeError(
                    f"Identified {num_peaks} corner(s) for the endo/epi base. We needed to find at least 2 corners to "
                    f"identify the corners at the base of the endo/epi."
                )

            # Identify the indices of the 2 highest peaks in the list of peaks
            base_highest_peaks = base_peak_heights.argsort()[-2:]
            # Sort the indices of the 2 highest peaks to make sure they stay ordered by descending theta
            # (so that the peak of the left corner comes first) regardless of their heights
            base_highest_peaks = sorted(base_highest_peaks, reverse=True)

            landmarks_polar_indices.extend(base_peaks[base_highest_peaks])

        if debug_plots:
            # Display contour curve in polar coordinates
            import seaborn as sns
            from matplotlib import pyplot as plt

            with sns.axes_style("darkgrid"):
                plot = sns.lineplot(data=pd.DataFrame({"theta": theta, "rho": rho}), x="theta", y="rho")

            # Annotate the peaks with their respective index
            for peak_idx, peak in enumerate(peaks):
                plot.annotate(f"{peak_idx}", (theta[peak], rho[peak]), xytext=(1, 4), textcoords="offset points")

            # Plot lines pointing to the peaks to make them more visible
            plot.vlines(x=theta[peaks], ymin=rho.min(), ymax=rho[peaks], linestyles="dashed")

            plt.show()

        # Map the indices of the peaks in polar coordinates back to the indices in the list of contour points
        contour_indices = sort_indices[landmarks_polar_indices]
        landmarks = contour[contour_indices]

        if debug_plots:
            plt.imshow(structure_mask)
            for landmark in landmarks:
                plt.scatter(landmark[1], landmark[0], c="r", marker="o", s=3)
            plt.show()

        return landmarks


    @staticmethod
    def _endo_base(
            segmentation: np.ndarray, lv_labels: SemanticStructureId, myo_labels: SemanticStructureId
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Finds the left/right markers at the base of the endocardium.

        Notes:
            - This implementation exists because it is more reliable for the endo than the more general algorithm that
              tries to identify peaks in the polar projection of a contour. As such, in cases where the latter gives
              acceptable results, it should be preferred.

        Args:
            segmentation: (H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the left ventricle.

        Returns:
            Coordinates of the left/right markers at the base of the endocardium.TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

        """
        struct = ndimage.generate_binary_structure(2, 2)
        left_ventricle = np.isin(segmentation, lv_labels)
        myocardium = np.isin(segmentation, myo_labels)
        others = ~(left_ventricle + myocardium)
        dilated_myocardium = ndimage.binary_dilation(myocardium, structure=struct)
        dilated_others = ndimage.binary_dilation(others, structure=struct)
        y_coords, x_coords = np.nonzero(left_ventricle * dilated_myocardium * dilated_others)

        if (num_markers := len(y_coords)) < 2:
            raise RuntimeError(
                f"Identified {num_markers} marker(s) at the edges of the left ventricle/myocardium frontier. We need "
                f"to identify at least 2 such markers to determine the base of the left ventricle."
            )

        if np.all(x_coords == x_coords.mean()):
            # Edge case where the base points are aligned vertically
            # Divide frontier into bottom and top halves.
            coord_mask = y_coords > y_coords.mean()
            left_point_idx = y_coords[coord_mask].argmin()
            right_point_idx = y_coords[~coord_mask].argmax()
        else:
            # Normal case where there is a clear divide between left and right markers at the base
            # Divide frontier into left and right halves.
            coord_mask = x_coords < x_coords.mean()
            left_point_idx = y_coords[coord_mask].argmax()
            right_point_idx = y_coords[~coord_mask].argmax()
        return (
            np.array([y_coords[coord_mask][left_point_idx], x_coords[coord_mask][left_point_idx]]),
            np.array([y_coords[~coord_mask][right_point_idx], x_coords[~coord_mask][right_point_idx]]),
        )

    @staticmethod
    def structure_apex(
            segmentation: T,
            label: SemanticStructureId,
            base_coords: Tuple,
    ) -> T:
        if np.isnan(base_coords).any():
            # Early return if we couldn't reliably estimate the landmarks at the base of the left ventricle
            return np.nan

        # Identify the midpoint at the base of the left ventricle
        base_mid = np.array(base_coords).mean(axis=0)

        # Compute the distance from all pixels in the image to `lv_base_midpoint`
        mask = np.ones_like(segmentation, dtype=bool)
        mask[tuple(base_mid.round().astype(int))] = False
        dist_to_base_mid = ndimage.distance_transform_edt(mask)

        # Find the point within the left ventricle mask with maximum distance
        strucure = np.isin(segmentation, label)
        apex_coords = np.unravel_index(np.argmax(dist_to_base_mid * strucure), segmentation.shape)

        return apex_coords

    @staticmethod
    def lv_apex(
            segmentation: T,
            lv_labels: SemanticStructureId = Label.LV.value,
            myo_labels: SemanticStructureId = Label.MYO.value,
    ) -> T:
        """Measures the LV length as the distance between the LV's base midpoint and its furthest point at the apex.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium. The location of the myocardium is
                necessary to identify the markers at the base of the left ventricle.

        Returns:
            ([N], 1), Length of the left ventricle, or NaNs for the images where the LV base's midpoint cannot be
            reliably estimated.
        """
        # Identify the base of the left ventricle
        lv_base_coords = ContourMeasure._endo_base(segmentation, lv_labels=lv_labels, myo_labels=myo_labels)
        lv_apex_coords = ContourMeasure.structure_apex(segmentation, lv_labels, lv_base_coords)
        return lv_apex_coords

    @staticmethod
    def myo_apex(
            segmentation: T,
            myo_base_coords,
            myo_labels: SemanticStructureId = Label.MYO.value,
    ) -> T:
        """Measures the LV length as the distance between the LV's base midpoint and its furthest point at the apex.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium. The location of the myocardium is
                necessary to identify the markers at the base of the left ventricle.

        Returns:
            ([N], 1), Length of the left ventricle, or NaNs for the images where the LV base's midpoint cannot be
            reliably estimated.
        """
        myo_apex_coords = ContourMeasure.structure_apex(segmentation, myo_labels, myo_base_coords)
        return myo_apex_coords

    @staticmethod
    def structure_edge(
            segmentation: np.ndarray,
            label: SemanticStructureId,
    ) -> np.ndarray:
        mask = np.isin(segmentation, label).astype(int)
        edge = mask ^ erosion(mask, selem=np.ones((3, 3)))
        return edge

    @staticmethod
    def myo_edge(segmentation: np.ndarray, myo_labels: SemanticStructureId = Label.MYO) -> np.ndarray:
        myo_mask = np.isin(segmentation, myo_labels).astype(int)
        myo_mask = convex_hull_image(myo_mask)
        myo_edge = myo_mask ^ erosion(myo_mask, selem=np.ones((3, 3)))
        return myo_edge

    @staticmethod
    def get_path(img, start, end):
        height, width = img.shape

        # All 8 directions
        delta = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

        # Store the results of the BFS as the shortest distance to start
        grid = [[sys.maxsize for _ in range(width)] for _ in range(height)]
        grid[start[0]][start[1]] = 0

        # The actual BFS algorithm
        bfs = deque([start])
        found = False
        while len(bfs) > 0:
            y, x = bfs.popleft()
            # We've reached the end!
            if (y, x) == end:
                found = True
                break

            # Look all 8 directions for a good path
            for dy, dx in delta:
                yy, xx = y + dy, x + dx
                # If the next position hasn't already been looked at and it's white
                if 0 <= yy < height and 0 <= xx < width and grid[y][x] + 1 < grid[yy][xx] and img[yy][xx] != 0:
                    grid[yy][xx] = grid[y][x] + 1
                    bfs.append((yy, xx))

        if found:
            # Now rebuild the path from the end to beginning
            path = []
            y, x = end
            while grid[y][x] != 0:
                for dy, dx in delta:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width and grid[yy][xx] == grid[y][x] - 1:
                        path.append((yy, xx))
                        y, x = yy, xx
            # Get rid of the starting point from the final path
            path.pop()

            return np.array(path)
        else:
            plt.figure()
            plt.imshow(img)
            plt.scatter(start[0], start[1], label='start')
            plt.scatter(end[0], end[1], label='end')
            plt.legend()
            plt.show()