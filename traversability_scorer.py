"""
traversability_scorer.py
========================
Novel IEEE Contribution Module
-------------------------------
Title : Adaptive Semantic Traversability Estimation Using SegFormer
        for Real-Time Autonomous Robot Navigation

Instead of a binary safe / unsafe mask this module assigns every
pixel a continuous traversability score in [0.0, 1.0] based on the
semantic class predicted by SegFormer-B0 (ADE20K-512 label set).

The resulting *Semantic Traversability Heatmap* is the core research
contribution that distinguishes this work from standard segmentation
pipelines.

ADE20K label reference (selected):
  0  wall        4  floor        6  road
  1  building    5  tree         9  grass
  2  sky         7  ceiling      10 sidewalk
  3  floor       8  bed          11 person
 12  car        14  person      21  water
 22  rock       26  shelf       84  door
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────
#  ADE20K-512  Traversability Score Table
#  1.0 = completely safe   0.0 = hard obstacle
# ─────────────────────────────────────────────
ADE_TRAVERSABILITY_SCORES = {
    # ── Fully traversable surfaces ──────────────
    3:   1.0,   # floor
    4:   1.0,   # floor (alias)
    1:   1.0,   # road / pavement (ADE idx 1)
    6:   0.95,  # road
    10:  0.90,  # sidewalk / path
    11:  0.85,  # path
    29:  0.85,  # field
    91:  0.80,  # dirt / earth track

    # ── Conditionally traversable ────────────────
    9:   0.70,  # grass
    5:   0.60,  # tree (shadow/canopy on ground)
    22:  0.50,  # rock / gravel
    26:  0.40,  # shelf / low obstacle
    84:  0.35,  # door

    # ── Uncertain / dynamic ──────────────────────
    8:   0.20,  # bed (indoor clutter)
    14:  0.15,  # person
    11:  0.15,  # person (alias)
    12:  0.10,  # car
    13:  0.10,  # table

    # ── Hard obstacles / non-traversable ─────────
    0:   0.00,  # wall
    2:   0.00,  # sky
    15:  0.00,  # cabinet
    18:  0.00,  # curtain / pillar
    25:  0.00,  # bookcase
    55:  0.00,  # stairs (unsafe for wheeled robot)
    59:  0.00,  # stairs (alt)
}

# Default score for classes not in the table.
# 0.3 = uncertain / treat with caution.
DEFAULT_SCORE = 0.3

# Score thresholds for binary mask generation
TRAVERSABLE_THRESHOLD = 0.50   # >= this → traversable
OBSTACLE_THRESHOLD    = 0.25   # <  this → obstacle


def get_traversability_score(class_id: int) -> float:
    """Return the traversability score for a single ADE20K class id."""
    return ADE_TRAVERSABILITY_SCORES.get(int(class_id), DEFAULT_SCORE)


def build_score_map(seg_map: np.ndarray) -> np.ndarray:
    """
    Convert a 2-D integer segmentation label map (H×W) into a
    floating-point traversability score map (H×W, float32, range 0–1).

    Parameters
    ----------
    seg_map : np.ndarray  shape (H, W), dtype int/uint8
        Per-pixel ADE20K class labels from SegFormer.

    Returns
    -------
    score_map : np.ndarray  shape (H, W), dtype float32
        Per-pixel traversability scores in [0.0, 1.0].
    """
    score_map = np.full(seg_map.shape, DEFAULT_SCORE, dtype=np.float32)

    for cls_id, score in ADE_TRAVERSABILITY_SCORES.items():
        score_map[seg_map == cls_id] = score

    return score_map


def build_traversable_mask(score_map: np.ndarray) -> np.ndarray:
    """
    Threshold the score map into a binary traversable mask (uint8 0/255).
    """
    mask = np.zeros(score_map.shape, dtype=np.uint8)
    mask[score_map >= TRAVERSABLE_THRESHOLD] = 255
    return mask


def build_obstacle_mask(score_map: np.ndarray) -> np.ndarray:
    """
    Threshold the score map into a binary obstacle mask (uint8 0/255).
    """
    mask = np.zeros(score_map.shape, dtype=np.uint8)
    mask[score_map < OBSTACLE_THRESHOLD] = 255
    return mask


def build_heatmap(score_map: np.ndarray) -> np.ndarray:
    """
    Render the traversability score map as a colour heatmap (H×W×3, BGR).

    Colour encoding (JET-based, custom remapped):
      Blue  → score 0.0  (hard obstacle)
      Cyan  → score 0.2  (dangerous)
      Green → score 0.5  (uncertain)
      Yellow→ score 0.8  (mostly safe)
      Red   → score 1.0  (fully traversable)

    Parameters
    ----------
    score_map : np.ndarray  shape (H, W), dtype float32

    Returns
    -------
    heatmap : np.ndarray  shape (H, W, 3), dtype uint8  (BGR)
    """
    # Normalise to 0–255 uint8
    score_uint8 = (score_map * 255).clip(0, 255).astype(np.uint8)

    # Apply OpenCV JET colormap (blue=low, red=high)
    # Then invert so that red = safe (high score) which is intuitive
    heatmap = cv2.applyColorMap(score_uint8, cv2.COLORMAP_JET)

    return heatmap


def overlay_heatmap(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.50
) -> np.ndarray:
    """
    Alpha-blend the traversability heatmap on top of the raw camera frame.

    Parameters
    ----------
    frame   : np.ndarray  (H, W, 3) BGR camera frame
    heatmap : np.ndarray  (H, W, 3) BGR heatmap from build_heatmap()
    alpha   : float  blending weight for the heatmap  (0=invisible, 1=full)

    Returns
    -------
    blended : np.ndarray  (H, W, 3)
    """
    return cv2.addWeighted(frame, 1.0 - alpha, heatmap, alpha, 0)


def get_risk_class(score: float) -> str:
    """
    Classify a continuous traversability score into a discrete risk category.
    This enables high-level decision making beyond just visualization.
    """
    if score >= 0.75:
        return "Safe"
    elif score >= 0.35:
        return "Moderate Risk"
    else:
        return "Dangerous"

def score_map_stats(score_map: np.ndarray) -> dict:
    """
    Return summary statistics for the current score map.
    Useful for benchmarking and the FPS/accuracy report.
    """
    safe_pct = float(np.mean(score_map >= 0.75) * 100)
    moderate_pct = float(np.mean((score_map >= 0.35) & (score_map < 0.75)) * 100)
    dangerous_pct = float(np.mean(score_map < 0.35) * 100)
    
    return {
        "mean_score":        float(np.mean(score_map)),
        "safe_pixel_pct":    safe_pct,
        "moderate_risk_pct": moderate_pct,
        "dangerous_pct":     dangerous_pct,
        "obstacle_pixel_pct":float(np.mean(score_map < OBSTACLE_THRESHOLD) * 100),
        "uncertain_pixel_pct": float(
            np.mean(
                (score_map >= OBSTACLE_THRESHOLD) &
                (score_map <  TRAVERSABLE_THRESHOLD)
            ) * 100
        ),
    }
