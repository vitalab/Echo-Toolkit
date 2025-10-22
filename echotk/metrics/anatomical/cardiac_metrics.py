from echotk.metrics.utils.config import Label
from echotk.metrics.anatomical.anatomical_structure_metrics import Anatomical2DStructureMetrics
from echotk.metrics.anatomical.segmentation_metrics import Segmentation2DMetrics


class LeftAtriumMetrics(Anatomical2DStructureMetrics):
    """Class to compute metrics on the segmentation of the left atrium."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        super().__init__(segmentation_metrics, Label.ATRIUM)


class EpicardiumMetrics(Anatomical2DStructureMetrics):
    """Class to compute metrics on the segmentation of the left ventricle epicardium (LV + MYO)."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        # For myocardium we want to calculate anatomical metrics for the entire epicardium
        # Therefore we concatenate label 1 (lumen) and 2 (myocardium)
        super().__init__(segmentation_metrics, (Label.LV, Label.MYO))


class FrontierMetrics:
    """Class to compute metrics on the frontiers between segmentations of multiple anatomical structures."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        self.segmentation_metrics = segmentation_metrics

    def count_holes_between_lv_and_myo(self) -> int:
        """Counts the pixels in the gap between the left ventricle (LV) and myocardium (MYO).

        Returns:
            Count of pixels in the gap between the two areas segmented as left ventricle and myocardium.
        """
        return self.segmentation_metrics.count_holes_between_regions(Label.LV, Label.MYO)

    def count_holes_between_lv_and_atrium(self) -> int:
        """Counts the pixels in the gap between the left ventricle (LV) and left atrium.

        Returns:
            Count of pixels in the gap between the two areas segmented as left ventricle and left atrium.
        """
        return self.segmentation_metrics.count_holes_between_regions(Label.LV, Label.MYO)

    def measure_frontier_ratio_between_lv_and_bg(self) -> float:
        """Measures the ratio between the length of the frontier between the LV and BG and the width of the LV.

        Returns:
            Ratio between the length of the frontier between the LV and BG and the width of the LV.
        """
        return self.segmentation_metrics.measure_frontier_ratio_between_regions(Label.LV, Label.BG)

    def measure_frontier_ratio_between_myo_and_atrium(self) -> float:
        """Measures the ratio between the length of the frontier between the MYO and atrium and the width of the MYO.

        Returns:
            Ratio between the length of the frontier between the MYO and atrium and the width of the MYO.
        """
        return self.segmentation_metrics.measure_frontier_ratio_between_regions(Label.MYO, Label.ATRIUM)


class LeftVentricleMetrics(Anatomical2DStructureMetrics):
    """Class to compute metrics on the segmentation of the left ventricle."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        super().__init__(segmentation_metrics, Label.LV)


class MyocardiumMetrics(Anatomical2DStructureMetrics):
    """Class to compute metrics on the segmentation of the myocardium."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        super().__init__(segmentation_metrics, Label.MYO)


class SizeMetrics:
    """Class to compute metrics comparing sizes between segmentations of multiple anatomical structures."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        self.segmentation_metrics = segmentation_metrics
        self.no_structure_flag = float("nan")

    def measure_width_ratio_between_lv_and_myo(self) -> float:
        """Measures the relative width of the left ventricle and the myocardium at their joined center of mass.

        The width comparison is measured using the ratio between the width the width of the left ventricle and the total
        width of both myocardium walls along an horizontal line anchored at their joined center of mass.

        Returns:
            Ratio between the width the width of the left ventricle and the total width of both myocardium walls along
            an horizontal line anchored at their joined center of mass.
        """
        return self.segmentation_metrics.measure_width_ratio_between_regions(
            Label.LV, Label.MYO, no_structure_flag=self.no_structure_flag
        )
