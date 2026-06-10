"""
semantic_mapping.py
===================
Week 2 + IEEE Extension  –  2D Global Semantic Map
----------------------------------------------------
Maintains an 800×800 top-down map of the environment, updated each
frame with:
  • Green  – traversable (safe) pixels
  • Red    – obstacle pixels
  • Heatmap gradient – traversability score overlay (IEEE extension)
  • White  – robot position

The map is used for navigation context and for the paper figures.
"""

import cv2
import numpy as np

MAP_H = 800
MAP_W = 800

global_map       = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
traversal_scores = np.zeros((MAP_H, MAP_W),    dtype=np.float32)

robot_x = MAP_W // 2
robot_y = MAP_H - 100
scale   = 4


def update_map(
    traversable:  np.ndarray,
    obstacle:     np.ndarray,
    score_map:    np.ndarray | None = None,
) -> np.ndarray:
    """
    Update and return the 2-D global map.

    Parameters
    ----------
    traversable : np.ndarray (H, W) uint8  – traversable mask (0/255)
    obstacle    : np.ndarray (H, W) uint8  – obstacle mask    (0/255)
    score_map   : np.ndarray (H, W) float32 – traversability scores
                  (optional, IEEE extension – adds colour gradient)

    Returns
    -------
    global_map : np.ndarray (MAP_H, MAP_W, 3) BGR
    """
    global global_map, traversal_scores, robot_x, robot_y

    h, w = traversable.shape

    # ── Traversable pixels ─────────────────────────────────
    free_pts = np.column_stack(np.where(traversable == 255))
    for y, x in free_pts:
        mx = robot_x + (x - w // 2) // scale
        my = robot_y - (h - y)      // scale
        if 0 <= mx < MAP_W and 0 <= my < MAP_H:
            if score_map is not None:
                score = float(score_map[y, x])
                # Map score to green intensity: darker green = lower score
                g_val = int(score * 255)
                global_map[my, mx] = (0, g_val, 0)
                traversal_scores[my, mx] = score
            else:
                global_map[my, mx] = (0, 200, 0)

    # ── Obstacle pixels ────────────────────────────────────
    obs_pts = np.column_stack(np.where(obstacle == 255))
    for y, x in obs_pts:
        mx = robot_x + (x - w // 2) // scale
        my = robot_y - (h - y)      // scale
        if 0 <= mx < MAP_W and 0 <= my < MAP_H:
            global_map[my, mx]       = (0, 0, 200)
            traversal_scores[my, mx] = 0.0

    # ── Robot marker ───────────────────────────────────────
    cv2.circle(global_map, (robot_x, robot_y), 8, (255, 255, 255), -1)
    cv2.putText(
        global_map, "ROBOT",
        (robot_x + 12, robot_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
    )

    return global_map


def reset_map() -> None:
    """Clear the global map (call between scenes/environments)."""
    global global_map, traversal_scores
    global_map[:]       = 0
    traversal_scores[:] = 0.0
