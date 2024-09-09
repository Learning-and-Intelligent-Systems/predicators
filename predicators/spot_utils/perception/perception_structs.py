"""Structs for perception."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from bosdyn.api.geometry_pb2 import FrameTreeSnapshot
from bosdyn.client import math_helpers
from numpy.typing import NDArray
from scipy import ndimage


@dataclass
class RGBDImageWithContext:
    """An RGBD image with context including the pose and intrinsics of the
    camera."""
    rgb: NDArray[np.uint8]
    depth: NDArray[np.uint16]
    image_rot: float
    camera_name: str
    world_tform_camera: math_helpers.SE3Pose
    depth_scale: float
    transforms_snapshot: FrameTreeSnapshot
    frame_name_image_sensor: str
    camera_model: Any  # bosdyn.api.image_pb2.PinholeModel, but not available

    @property
    def rotated_rgb(self) -> NDArray[np.uint8]:
        """The image rotated to be upright."""
        return ndimage.rotate(self.rgb, self.image_rot, reshape=False)

@dataclass
class RGBDImage:
    """An RGBD image"""
    rgb: NDArray[np.uint8]
    depth: NDArray[np.uint16]
    image_rot: float
    camera_name: str
    depth_scale: float
    camera_model: Any  # bosdyn.api.image_pb2.PinholeModel, but not available

    @property
    def rotated_rgb(self) -> NDArray[np.uint8]:
        """The image rotated to be upright."""
        return ndimage.rotate(self.rgb, self.image_rot, reshape=False)


@dataclass(frozen=True)
class ObjectDetectionID:
    """A unique identifier for an object that is to be detected."""


@dataclass(frozen=True)
class AprilTagObjectDetectionID(ObjectDetectionID):
    """An ID for an object to be detected from an april tag.

    The object center is defined to be the center of the april tag plus
    offset.
    """
    april_tag_number: int

    def __str__(self) -> str:
        return f"AprilTag({self.april_tag_number})"

    def __repr__(self) -> str:
        return f"AprilTag({self.april_tag_number})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, AprilTagObjectDetectionID)
        return self.april_tag_number == other.april_tag_number


@dataclass(frozen=True)
class LanguageObjectDetectionID(ObjectDetectionID):
    """An ID for an object to be detected with a vision-language model."""
    language_id: str

    def __str__(self) -> str:
        return f"LanguageID({self.language_id})"

    def __repr__(self) -> str:
        return f"LanguageID({self.language_id})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, LanguageObjectDetectionID)
        return self.language_id == other.language_id


@dataclass(frozen=True)
class PythonicObjectDetectionID(ObjectDetectionID):
    """An ID for an object to be detected with an arbitrary python function."""
    name: str
    fn: Callable[[Dict[str, RGBDImageWithContext]],
                 Optional[math_helpers.SE3Pose]]

    def __str__(self) -> str:
        return f"PythonicID({self.name})"

    def __repr__(self) -> str:
        return f"PythonicID({self.name})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, PythonicObjectDetectionID)
        return self.name == other.name


@dataclass(frozen=True)
class KnownStaticObjectDetectionID(ObjectDetectionID):
    """An ID for an object with a known, static pose."""
    obj_name: str
    pose: math_helpers.SE3Pose

    def __str__(self) -> str:
        return f"KnownStaticObject({self.obj_name})"

    def __repr__(self) -> str:
        return f"KnownStaticObject({self.obj_name})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, KnownStaticObjectDetectionID)
        return self.obj_name == other.obj_name


@dataclass(frozen=True)
class SegmentedBoundingBox:
    """Intermediate return value from vision-language models."""
    bounding_box: Tuple[float, float, float, float]
    mask: NDArray[np.uint8]
    score: float
