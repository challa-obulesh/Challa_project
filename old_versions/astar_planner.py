"""
astar_planner.py
================
Cost-Aware A* Path Planner
---------------------------
Extended from the standard binary-grid version to support a
floating-point *cost grid* produced by `occupancy_grid.build_cost_grid`.

Each grid cell carries a traversal cost in [0.0, 1.0]:
  0.0 = fully safe (traversability score 1.0)
  1.0 = blocked   (traversability score 0.0)

Cells with cost >= OBSTACLE_COST_THRESHOLD are treated as impassable.
For passable cells the actual cost is added to the g-score so the
planner naturally prefers high-traversability corridors.
"""

import heapq
import numpy as np

# Cells with cost above this value are treated as hard obstacles.
OBSTACLE_COST_THRESHOLD = 0.75


def heuristic(a: tuple, b: tuple) -> float:
    """Euclidean distance heuristic."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def astar(
    grid,
    start: tuple,
    goal: tuple,
) -> list | None:
    """
    Standard binary A* (grid values: 0 = free, 1 = obstacle).

    Kept for backward-compatibility with code that still passes a
    binary occupancy grid.

    Parameters
    ----------
    grid  : 2-D array-like  – binary occupancy (0/1)
    start : (row, col)
    goal  : (row, col)

    Returns
    -------
    path  : list of (row, col) tuples from start to goal, or None
    """
    rows = len(grid)
    cols = len(grid[0])

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score   = {start: 0.0}
    closed    = set()

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        r, c = current

        neighbors = [
            (r - 1, c), (r + 1, c),
            (r, c - 1), (r, c + 1),
            (r - 1, c - 1), (r - 1, c + 1),
            (r + 1, c - 1), (r + 1, c + 1),
        ]

        for nr, nc in neighbors:
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:
                continue

            move_cost = 1.0 if nr == r or nc == c else 1.414

            tentative_g = g_score[current] + move_cost

            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                came_from[(nr, nc)] = current
                g_score[(nr, nc)]   = tentative_g
                f = tentative_g + heuristic((nr, nc), goal)
                heapq.heappush(open_set, (f, (nr, nc)))

    return None


def astar_cost(
    cost_grid: np.ndarray,
    start: tuple,
    goal: tuple,
) -> list | None:
    """
    Cost-Aware A* that uses the floating-point cost grid from
    `occupancy_grid.build_cost_grid`.

    Cells with cost >= OBSTACLE_COST_THRESHOLD are impassable.
    For passable cells the per-cell cost is added to g, so the
    planner prefers low-cost (high-traversability) corridors.

    Parameters
    ----------
    cost_grid : np.ndarray (rows, cols) float32  values in [0, 1]
    start     : (row, col)
    goal      : (row, col)

    Returns
    -------
    path : list of (row, col) tuples, or None if no path found
    """
    rows, cols = cost_grid.shape

    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from = {}
    g_score   = {start: 0.0}
    closed    = set()

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        r, c = current

        neighbors = [
            (r - 1, c), (r + 1, c),
            (r, c - 1), (r, c + 1),
            (r - 1, c - 1), (r - 1, c + 1),
            (r + 1, c - 1), (r + 1, c + 1),
        ]

        for nr, nc in neighbors:
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            cell_cost = float(cost_grid[nr, nc])

            if cell_cost >= OBSTACLE_COST_THRESHOLD:
                continue

            diag_factor = 1.414 if (nr != r and nc != c) else 1.0
            move_cost   = diag_factor * (1.0 + cell_cost)   # penalise risky cells

            tentative_g = g_score[current] + move_cost

            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                came_from[(nr, nc)] = current
                g_score[(nr, nc)]   = tentative_g
                f = tentative_g + heuristic((nr, nc), goal)
                heapq.heappush(open_set, (f, (nr, nc)))

    return None
