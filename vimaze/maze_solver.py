from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vimaze.graph import Graph
    from vimaze.maze_animator import MazeAnimator


class MazeSolver:
    def __init__(self, graph: 'Graph', animator: 'MazeAnimator'):
        self.graph = graph
        self.animator = animator

    def solve_maze(self):
        pass
