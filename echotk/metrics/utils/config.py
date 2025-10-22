from enum import IntEnum, unique
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

SemanticStructureId = Union[int, Sequence[int]]
ProtoLabel = Union[int, str, "LabelEnum"]


class LabelEnum(IntEnum):
    """Abstract base class for enumerated labels, providing more functionalities on top of the core `IntEnum`."""

    def __str__(self):  # noqa: D105
        return self.name.lower()

    @classmethod
    def from_proto_label(cls, proto_label: ProtoLabel) -> "LabelEnum":
        """Creates a label from a protobuf label.

        Args:
            proto_label: Either the integer value of a label, the (case-insensitive) name of a label, or directly a
                `LabelEnum`.

        Returns:
            A `LabelEnum` instance.
        """
        if isinstance(proto_label, int):
            label = cls(proto_label)
        elif isinstance(proto_label, str):
            label = cls[proto_label.upper()]
        elif isinstance(proto_label, cls):
            label = proto_label
        else:
            raise ValueError(
                f"Unsupported label type: {type(proto_label)} for proto-label. Should be one of: {ProtoLabel}."
            )
        return label

    @classmethod
    def from_proto_labels(cls, proto_labels: Sequence[ProtoLabel]) -> List["LabelEnum"]:
        """Creates a list of labels from a sequence of protobuf labels.

        Args:
            proto_labels: Sequence of either integer values of labels, (case-insensitive) names of labels, or directly
                `LabelEnum`s.

        Returns:
            List of `LabelEnum` instances
        """
        return [cls.from_proto_label(proto_label) for proto_label in proto_labels]

@unique
class Label(LabelEnum):
    """Identifiers of the different anatomical structures available in the dataset's segmentation mask."""

    BG = 0
    """BackGround"""
    LV = 1
    """Left Ventricle"""
    MYO = 2
    """MYOcardium"""
    ATRIUM = 3
    """Atrium"""
