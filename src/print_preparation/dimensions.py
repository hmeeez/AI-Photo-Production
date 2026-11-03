from __future__ import annotations

import math

A1_MM = (594, 841)
A1_LANDSCAPE_MM = (841, 594)


def target_pixels(width_mm: float, height_mm: float, ppi: int) -> tuple[int, int]:
    if (
        not math.isfinite(width_mm)
        or not math.isfinite(height_mm)
        or width_mm <= 0
        or height_mm <= 0
    ):
        raise ValueError("Paper dimensions must be finite positive values.")
    if not 72 <= ppi <= 1200:
        raise ValueError("PPI must be between 72 and 1200.")
    return (
        round(width_mm / 25.4 * ppi),
        round(height_mm / 25.4 * ppi),
    )
