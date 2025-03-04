from typing import TYPE_CHECKING, Optional
import logging

from vimaze.solvers.dfs_solver import DfsSolver

if TYPE_CHECKING:
    from vimaze.graph import Graph
    from vimaze.graph import Node
    from vimaze.maze_animator import MazeAnimator


class MazeSolver:
    def __init__(self, graph: 'Graph', animator: 'MazeAnimator'):
        self.graph = graph
        self.animator = animator
        
        self.solved_path: Optional[list['Node']] = None

    def solve_maze(self, start_pos: tuple[int, int], end_pos: tuple[int, int], algorithm: str):
        """Solve a maze using the specified algorithm."""
        logging.debug(f"Solving maze with {algorithm} algorithm, {start_pos} {end_pos}")

        if algorithm == "DFS":
            solver = DfsSolver()
            solver.solve()
            
