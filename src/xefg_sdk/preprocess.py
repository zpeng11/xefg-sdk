"""Pack normalized model inputs; engine-coordinate adaptation happens upstream."""
import numpy as np

from .types import Config, FrameInputs, ImageView, Rect, UiMode


FORMATS = {
    "BGRA8_UNORM": (np.dtype("uint8"), 4),
    "RGBA8_UNORM": (np.dtype("uint8"), 4),
    "RG16_FLOAT": (np.dtype("float16"), 2),
    "RG32_FLOAT": (np.dtype("float32"), 2),
    "R32_FLOAT": (np.dtype("float32"), None),
}


def image_data(view: ImageView, allowed):
    if not isinstance(view, ImageView):
        raise ValueError("required image is missing; supply an ImageView")
    if view.format not in allowed:
        raise ValueError("unsupported format %s; expected %s" % (view.format, ", ".join(allowed)))
    data = np.asarray(view.data)
    dtype, channels = FORMATS[view.format]
    if data.dtype != dtype or data.ndim != (2 if channels is None else 3):
        raise ValueError("%s requires %s %s data" % (
            view.format, dtype, "HW" if channels is None else "HWC"))
    if channels is not None and data.shape[2] != channels:
        raise ValueError("%s requires %d channels" % (view.format, channels))
    if min(data.shape[:2]) <= 0:
        raise ValueError("empty images are not supported")
    width, height = view.size
    if view.rect is not None and view.rect != Rect(0, 0, width, height):
        raise ValueError("version 1 profiles support full-resource rectangles only")
    return np.ascontiguousarray(data)


def color_data(view):
    data = image_data(view, ("BGRA8_UNORM", "RGBA8_UNORM"))
    if view.format == "RGBA8_UNORM":
        data = np.ascontiguousarray(data[..., [2, 1, 0, 3]])
    return data


def resolve_mode(previous: FrameInputs, current: FrameInputs, config: Config):
    mode = UiMode(config.ui_mode)
    if mode == UiMode.AUTO:
        if previous.hudless_color is not None and current.hudless_color is not None and current.ui is not None:
            return UiMode.HUDLESS_UITEXTURE
        if previous.backbuffer_color is not None and current.backbuffer_color is not None:
            return UiMode.NONE
        raise ValueError("AUTO needs a hudless pair plus UI, or a backbuffer pair")
    if mode not in (UiMode.NONE, UiMode.HUDLESS_UITEXTURE):
        raise ValueError("only NONE and HUDLESS_UITEXTURE models are supported")
    return mode


def prepare_inputs(previous: FrameInputs, current: FrameInputs, config: Config):
    if current.reset_history:
        raise ValueError("reset_history invalidates a pair; submit through Session or start a new pair")
    if current.frame_id <= previous.frame_id:
        raise ValueError("current frame_id must follow previous frame_id")
    mode = resolve_mode(previous, current, config)
    field = "hudless_color" if mode == UiMode.HUDLESS_UITEXTURE else "backbuffer_color"
    old_color = color_data(getattr(previous, field))
    new_color = color_data(getattr(current, field))
    if old_color.shape != new_color.shape:
        raise ValueError("the frame pair must have identical colour dimensions")
    motion = image_data(current.motion_vectors, ("RG16_FLOAT", "RG32_FLOAT"))
    # The graph's input texture is RG16F even when the application computes
    # vectors in float32. Round once before either graph consumer sees it.
    motion = np.ascontiguousarray(motion, dtype=np.float16)
    depth = image_data(current.depth, ("R32_FLOAT",))
    if motion.shape[:2] != depth.shape:
        raise ValueError("motion and depth must have the same dimensions")
    values = {"previous_color": old_color, "current_color": new_color,
              "motion_vectors": motion, "depth": depth}
    if mode == UiMode.HUDLESS_UITEXTURE:
        ui = color_data(current.ui)
        if ui.shape != new_color.shape:
            raise ValueError("UI must match colour dimensions")
        values["ui"] = ui
    return mode, values
