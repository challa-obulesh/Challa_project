"""
Traversability Scoring & A* Path Planning for Cityscapes Segmentation.

Converts semantic segmentation maps (Cityscapes 19-class) into traversability
heatmaps and plans safe navigation paths using A* search.
"""

import numpy as np
import heapq
import cv2


# ── Cityscapes class → traversability score ──────────────────────────────────
# 0 = impassable, 1.0 = fully traversable
CITYSCAPES_TRAVERSABILITY = {
    0: 1.0,    # road
    1: 0.8,    # sidewalk
    2: 0.0,    # building
    3: 0.0,    # wall
    4: 0.0,    # fence
    5: 0.0,    # pole
    6: 0.0,    # traffic light
    7: 0.0,    # traffic sign
    8: 0.1,    # vegetation
    9: 0.6,    # terrain
    10: 0.0,   # sky
    11: 0.0,   # person
    12: 0.0,   # rider
    13: 0.0,   # car
    14: 0.0,   # truck
    15: 0.0,   # bus
    16: 0.0,   # train
    17: 0.0,   # motorcycle
    18: 0.0,   # bicycle
}

# Precompute lookup table as numpy array for fast vectorized mapping
_TRAV_LUT = np.zeros(256, dtype=np.float32)
for _cls_id, _score in CITYSCAPES_TRAVERSABILITY.items():
    _TRAV_LUT[_cls_id] = _score

# Cityscapes colour palette (BGR) for visualisation
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],   # road
    [232, 35, 244],   # sidewalk
    [70, 70, 70],     # building
    [156, 102, 102],  # wall
    [153, 153, 190],  # fence
    [153, 153, 153],  # pole
    [30, 170, 250],   # traffic light
    [0, 220, 220],    # traffic sign
    [35, 142, 107],   # vegetation
    [152, 251, 152],  # terrain
    [180, 130, 70],   # sky
    [60, 20, 220],    # person
    [0, 0, 255],      # rider
    [142, 0, 0],      # car
    [70, 0, 0],       # truck
    [100, 60, 0],     # bus
    [100, 80, 0],     # train
    [230, 0, 0],      # motorcycle
    [32, 11, 119],    # bicycle
], dtype=np.uint8)


def seg_to_traversability(seg_map: np.ndarray) -> np.ndarray:
    """
    Convert a Cityscapes segmentation map to a traversability heatmap.

    Args:
        seg_map: (H, W) uint8 array with Cityscapes class IDs (0-18).

    Returns:
        (H, W) float32 traversability map in [0, 1].
    """
    return _TRAV_LUT[seg_map]


def apply_distance_weighting(trav_map: np.ndarray, decay: float = 2.0) -> np.ndarray:
    """
    Apply distance-based weighting: areas closer to the bottom (ego vehicle)
    are weighted higher for navigation relevance.

    Args:
        trav_map: (H, W) float32 traversability map.
        decay:    Exponential decay factor. Higher = steeper falloff.

    Returns:
        (H, W) float32 weighted traversability map.
    """
    h = trav_map.shape[0]
    # Weights go from 1.0 at bottom to near-0 at top
    weights = np.linspace(0, 1, h, dtype=np.float32) ** decay
    # Flip so bottom = 1.0
    weights = weights[::-1]
    return trav_map * weights[:, np.newaxis]


def seg_to_colourmap(seg_map: np.ndarray) -> np.ndarray:
    """
    Convert segmentation class IDs to a colour overlay (BGR).

    Args:
        seg_map: (H, W) uint8 array with Cityscapes class IDs.

    Returns:
        (H, W, 3) uint8 BGR colour image.
    """
    # Clip to valid range
    clipped = np.clip(seg_map, 0, len(CITYSCAPES_PALETTE) - 1)
    return CITYSCAPES_PALETTE[clipped]


def traversability_to_heatmap(trav_map: np.ndarray) -> np.ndarray:
    """
    Convert traversability map to a colourful heatmap (BGR).

    Green = traversable, Red = obstacle, Yellow = partial.

    Args:
        trav_map: (H, W) float32 traversability map in [0, 1].

    Returns:
        (H, W, 3) uint8 BGR heatmap image.
    """
    # Normalise to 0-255 and apply colourmap
    norm = np.clip(trav_map * 255, 0, 255).astype(np.uint8)
    # COLORMAP_JET: blue(low) → green(mid) → red(high)
    # We want green=safe, red=danger, so invert
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return heatmap


def inject_yolo_obstacles(
    trav_map: np.ndarray,
    detections: list,
    frame_shape: tuple,
    padding: int = 10
) -> np.ndarray:
    """
    Zero out traversability in regions where YOLO detected obstacles.

    Args:
        trav_map:    (H, W) float32 traversability map.
        detections:  List of YOLO detection dicts with 'bbox' (x1,y1,x2,y2)
                     in original frame coordinates.
        frame_shape: (orig_H, orig_W) of the original frame.
        padding:     Pixels to expand each detection box.

    Returns:
        Modified traversability map with obstacle regions zeroed.
    """
    if not detections:
        return trav_map

    trav_h, trav_w = trav_map.shape
    orig_h, orig_w = frame_shape[:2]

    scale_y = trav_h / orig_h
    scale_x = trav_w / orig_w

    result = trav_map.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        # Scale to traversability map coordinates
        ty1 = max(0, int((y1 - padding) * scale_y))
        ty2 = min(trav_h, int((y2 + padding) * scale_y))
        tx1 = max(0, int((x1 - padding) * scale_x))
        tx2 = min(trav_w, int((x2 + padding) * scale_x))
        result[ty1:ty2, tx1:tx2] = 0.0

    return result


# ── A* Path Planner ──────────────────────────────────────────────────────────

class AStarPlanner:
    """
    A* path planner on a downsampled traversability grid.
    Plans from bottom-center (ego position) to top-center (forward goal).
    """

    def __init__(self, grid_size: tuple = (40, 24)):
        """
        Args:
            grid_size: (grid_H, grid_W) for the planning grid.
        """
        self.grid_h, self.grid_w = grid_size

    def plan(
        self,
        trav_map: np.ndarray,
        start: tuple = None,
        goal: tuple = None
    ) -> list:
        """
        Plan a path on the traversability map.

        Args:
            trav_map: (H, W) float32 traversability map.
            start:    (row, col) in grid coords. Default: bottom-center.
            goal:     (row, col) in grid coords. Default: top-center.

        Returns:
            List of (row, col) grid coordinates forming the path.
            Empty list if no path found.
        """
        # Downsample to planning grid
        grid = cv2.resize(trav_map, (self.grid_w, self.grid_h),
                          interpolation=cv2.INTER_AREA)

        if start is None:
            start = (self.grid_h - 1, self.grid_w // 2)
        if goal is None:
            goal = (0, self.grid_w // 2)

        # A* search
        path = self._astar(grid, start, goal)
        return path

    def path_to_frame_coords(
        self,
        path: list,
        frame_shape: tuple
    ) -> list:
        """
        Convert grid path coordinates to original frame pixel coordinates.

        Args:
            path:        List of (row, col) grid coordinates.
            frame_shape: (H, W) of the original frame.

        Returns:
            List of (x, y) pixel coordinates in the original frame.
        """
        if not path:
            return []

        h, w = frame_shape[:2]
        scale_y = h / self.grid_h
        scale_x = w / self.grid_w

        return [
            (int((col + 0.5) * scale_x), int((row + 0.5) * scale_y))
            for row, col in path
        ]

    def _astar(
        self,
        grid: np.ndarray,
        start: tuple,
        goal: tuple
    ) -> list:
        """Core A* implementation."""
        rows, cols = grid.shape

        # Validate start/goal
        if (start[0] < 0 or start[0] >= rows or
                start[1] < 0 or start[1] >= cols):
            return []
        if (goal[0] < 0 or goal[0] >= rows or
                goal[1] < 0 or goal[1] >= cols):
            return []

        # 8-connected neighbours
        neighbours = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        closed = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            if current in closed:
                continue
            closed.add(current)

            for dr, dc in neighbours:
                nr, nc = current[0] + dr, current[1] + dc
                neighbour = (nr, nc)

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if neighbour in closed:
                    continue

                # Cost: inverse of traversability (higher traversability = lower cost)
                trav = grid[nr, nc]
                if trav < 0.05:
                    continue  # Impassable

                # Diagonal moves cost more
                move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                edge_cost = move_cost / (trav + 1e-6)

                tentative_g = g_score[current] + edge_cost

                if tentative_g < g_score.get(neighbour, float('inf')):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    f_score = tentative_g + heuristic(neighbour, goal)
                    heapq.heappush(open_set, (f_score, neighbour))

        return []  # No path found


def draw_path_on_frame(
    frame: np.ndarray,
    path_pixels: list,
    colour: tuple = (0, 255, 0),
    thickness: int = 3,
    dot_radius: int = 4
) -> np.ndarray:
    """
    Draw a navigation path overlay on a frame.

    Args:
        frame:       BGR image to draw on (will be modified in-place).
        path_pixels: List of (x, y) pixel coordinates.
        colour:      BGR colour tuple.
        thickness:   Line thickness.
        dot_radius:  Radius of path point dots.

    Returns:
        Frame with path overlay drawn.
    """
    if len(path_pixels) < 2:
        return frame

    # Draw path line
    pts = np.array(path_pixels, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=False, color=colour,
                  thickness=thickness, lineType=cv2.LINE_AA)

    # Draw start (green circle) and goal (red circle)
    cv2.circle(frame, path_pixels[0], dot_radius + 2,
               (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, path_pixels[-1], dot_radius + 2,
               (0, 0, 255), -1, cv2.LINE_AA)

    return frame
