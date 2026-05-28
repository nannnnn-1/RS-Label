from typing import List

import cv2
import numpy as np


def mask_to_polygons(
    mask: np.ndarray, epsilon_factor: float = 0.001
) -> List[List[List[float]]]:
    """Convert binary mask to list of polygons."""
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        poly = approx.reshape(-1, 2).tolist()
        if len(poly) >= 3:
            polygons.append(poly)

    return polygons


def largest_polygon(mask: np.ndarray, epsilon_factor: float = 0.001) -> List[List[float]]:
    """Get the largest polygon from a mask."""
    polygons = mask_to_polygons(mask, epsilon_factor)
    if not polygons:
        return []
    return max(polygons, key=lambda p: cv2.contourArea(
        np.array(p, dtype=np.int32).reshape(-1, 1, 2)
    ))
