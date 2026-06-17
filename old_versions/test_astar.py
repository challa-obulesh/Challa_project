import numpy as np

from astar_planner import astar

grid = np.zeros((10,10), dtype=np.uint8)

grid[4,2:8] = 1

start = (9,5)

goal = (0,5)

path = astar(
    grid,
    start,
    goal
)

print(path)
