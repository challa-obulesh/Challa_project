"""
occupancy_grid.py
=================
Builds a coarse occupancy grid from a traversability score map.

Each grid cell is assigned a cost value (0 = free, 1 = occupied)
based on the *mean traversability score* of its pixels.  Cells with
a mean score below `free_threshold` are treated as obstacles.

Compared to the previous binary version this gives the A* planner
richer cost information so it can prefer higher-score corridors.
"""

import cv2
import numpy as np


def build_grid(
    score_map: np.ndarray,
    grid_size: int = 10,
    free_threshold: float = 0.40,
) -> np.ndarray:
    """
    Convert a floating-point traversability score map to a binary
    occupancy grid.

    Parameters
    ----------
    score_map       : np.ndarray (H, W) float32  – per-pixel scores 0-1
    grid_size       : int  – side length (pixels) of each grid cell
    free_threshold  : float  – cells with mean score >= this are free (0)

    Returns
    -------
    grid : np.ndarray (rows, cols) uint8  – 0 = free, 1 = obstacle
    """
    h, w = score_map.shape
    rows = h // grid_size
    cols = w // grid_size

    grid = np.ones((rows, cols), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            cell = score_map[
                r * grid_size:(r + 1) * grid_size,
                c * grid_size:(c + 1) * grid_size,
            ]
            mean_score = float(np.mean(cell))
            grid[r, c] = 0 if mean_score >= free_threshold else 1

    return grid


def build_cost_grid(
    score_map: np.ndarray,
    grid_size: int = 10,
) -> np.ndarray:
    """
    Build a floating-point cost grid (0.0 = fully safe, 1.0 = blocked).
    Used by the cost-aware A* planner.

    Parameters
    ----------
    score_map : np.ndarray (H, W) float32
    grid_size : int

    Returns
    -------
    cost_grid : np.ndarray (rows, cols) float32
    """
    h, w = score_map.shape
    rows = h // grid_size
    cols = w // grid_size

    cost_grid = np.ones((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            cell = score_map[
                r * grid_size:(r + 1) * grid_size,
                c * grid_size:(c + 1) * grid_size,
            ]
            mean_score = float(np.mean(cell))
            # cost = 1 - score  (high score → low cost → preferred path)
            cost_grid[r, c] = 1.0 - mean_score

    return cost_grid


def draw_grid(frame: np.ndarray, grid_size: int = 10) -> np.ndarray:
    """Overlay a white grid on *frame* (in-place)."""
    h, w = frame.shape[:2]

    for y in range(0, h, grid_size):
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 1)

    for x in range(0, w, grid_size):
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)

    return frame
