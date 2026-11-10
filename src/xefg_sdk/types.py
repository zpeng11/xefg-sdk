"""Normalized model inputs, independent of engine and graphics API conventions."""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class UiMode(IntEnum):
    AUTO = 0
    NONE = 1
    BACKBUFFER_UITEXTURE = 2
    HUDLESS_UITEXTURE = 3
    BACKBUFFER_HUDLESS = 4
    BACKBUFFER_HUDLESS_UITEXTURE = 5


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class ImageView:
    """HWC (colour/MV) or HW (depth); strides come from the NumPy array.

    Formats: BGRA8_UNORM, RGBA8_UNORM, RG16_FLOAT, RG32_FLOAT, R32_FLOAT.
    Version 1 supports full-resource rectangles only.
    """
    data: np.ndarray
    format: str
    rect: Rect | None = None

    @property
    def size(self):
        if self.data.ndim < 2:
            raise ValueError("image data must have height and width dimensions")
        return int(self.data.shape[1]), int(self.data.shape[0])

    def copy(self):
        return ImageView(self.data.copy(), self.format, self.rect)


@dataclass(frozen=True)
class FrameInputs:
    """One frame at the model boundary, after engine-side adaptation.

    Motion includes camera/object motion, in current-to-previous MV-grid pixel
    units with jitter already compensated. Depth follows the model package's
    raw, non-inverted convention; the SDK does not linearize or rescale it.
    Mode 3 UI is premultiplied. Engine matrices/scales/flags are not inputs.
    """
    frame_id: int
    backbuffer_color: ImageView | None = None
    hudless_color: ImageView | None = None
    depth: ImageView | None = None
    motion_vectors: ImageView | None = None
    ui: ImageView | None = None
    reset_history: bool = False

    def copy(self):
        return FrameInputs(
            frame_id=self.frame_id,
            **{name: None if getattr(self, name) is None else getattr(self, name).copy()
               for name in ("backbuffer_color", "hudless_color", "depth", "motion_vectors", "ui")},
            reset_history=self.reset_history)


@dataclass(frozen=True)
class Config:
    """Model selection only; input conventions are fixed by the package."""
    ui_mode: UiMode = UiMode.AUTO
    profile_id: str | None = None


@dataclass(frozen=True)
class InterpolationResult:
    status: str
    generated_frame: ImageView | None
    previous_frame_id: int | None
    current_frame_id: int
    interpolation_fraction: float = 0.5
    ui_mode: UiMode | None = None
    profile_id: str | None = None
