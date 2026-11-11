"""Numeric BGRA8 comparison; no image-quality pass/fail threshold."""
import math

import numpy as np


def compare(reference, actual, mask=None, tolerance=2):
    if reference.dtype != np.uint8 or actual.dtype != np.uint8 or reference.shape != actual.shape:
        raise ValueError("Comparison requires matching uint8 images")
    if reference.ndim != 3 or reference.shape[2] != 4:
        raise ValueError("Comparison requires BGRA images")
    difference = np.abs(reference.astype(np.int16) - actual.astype(np.int16))
    if mask is not None:
        if mask.shape != reference.shape[:2] or mask.dtype != bool:
            raise ValueError("Mask must be a boolean image of matching size")
        difference = difference[mask]
    else:
        difference = difference.reshape(-1, 4)
    if not difference.size:
        raise ValueError("Comparison region is empty")
    rgb = difference[:, :3].astype(np.float64)
    mse = float(np.mean(rgb ** 2))
    return {"pixels": len(rgb), "rgb_mae": float(rgb.mean()), "rgb_max": int(rgb.max()),
            "rgb_rmse": math.sqrt(mse), "rgb_psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "rgb_identical": mse == 0, "different_pixel_fraction": float(np.any(rgb != 0, axis=1).mean()),
            "over_tolerance_pixel_fraction": float(np.any(rgb > tolerance, axis=1).mean()),
            "tolerance": tolerance, "alpha_mae": float(difference[:, 3].mean()),
            "alpha_max": int(difference[:, 3].max())}


def rgb_image(bgra):
    return bgra[:, :, [2, 1, 0]]


def difference_image(reference, actual, gain=8):
    difference = np.abs(reference.astype(np.int16) - actual.astype(np.int16))
    return np.clip(difference[:, :, [2, 1, 0]] * gain, 0, 255).astype(np.uint8)
