"""Distances along a band path for the plots of band structures."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def get_bands_data(
    kpoint_coords: NDArray,
    reclat: NDArray,
    nk_per_seg: int,
    labels: list[str],
) -> tuple[NDArray, list, list]:
    """Measure distances of points from origin along paths and get special points."""
    k_cart = kpoint_coords @ reclat
    nk_total = len(kpoint_coords)
    n_segments = nk_total // nk_per_seg
    assert n_segments * nk_per_seg == nk_total
    distances = _get_distances_along_BZ_path(nk_total, n_segments, nk_per_seg, k_cart)
    points, labels_at_points = _get_special_points(
        labels, distances, n_segments, nk_per_seg, nk_total
    )

    return distances, points, labels_at_points


def _get_distances_along_BZ_path(
    nk_total: int, n_segments: int, nk_per_seg: int, k_cart: NDArray
) -> NDArray:
    """Measure distances of points from origin along paths.

    Returns
    -------
    distances : np.ndarray
        Distances of points from origin along BZ-paths.
        shape=(nk_total,)

    """
    distances = np.zeros(nk_total)
    count = 0
    for _ in range(n_segments):
        for i_pts in range(nk_per_seg):
            # Treatment of jump between equivalent points on BZ boundary
            if i_pts == 0:
                delta_dist = 0
            else:
                delta_dist = np.linalg.norm(k_cart[count, :] - k_cart[count - 1, :])
            distances[count] = distances[count - 1] + delta_dist
            count += 1
    return distances


def _get_special_points(
    labels: list[str],
    distances: NDArray,
    n_segments: int,
    nk_per_seg: int,
    nk_total: int,
) -> tuple[list, list]:
    """Plot special points at vertical lines and labels."""
    # Left most
    points = []
    labels_at_points = []

    labels_at_points.append(labels[0][0])
    points.append(distances[0])

    count = 0
    for i_seg in range(n_segments):
        for _ in range(nk_per_seg):
            count += 1
        if count != nk_total:
            points.append(distances[count])
            if labels[i_seg * 2 + 1] == labels[i_seg * 2 + 2]:
                labels_at_points.append(labels[i_seg * 2 + 1][0])
            else:
                labels_at_points.append(
                    f"{labels[i_seg * 2 + 1][0]}|{labels[i_seg * 2 + 2][0]}"
                )

    labels_at_points.append(labels[-1][0])
    points.append(distances[-1])

    return points, labels_at_points
