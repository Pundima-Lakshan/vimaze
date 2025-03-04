import logging
from typing import TYPE_CHECKING, Optional

from vimaze.solvers.bfs_solver import BfsSolver
from vimaze.solvers.dfs_solver import DfsSolver

if TYPE_CHECKING:
    from vimaze.graph import Graph
    from vimaze.graph import Node
    from vimaze.maze_animator import MazeAnimator
    from vimaze.timer import Timer


class MazeSolver:
    def __init__(self, animator: 'MazeAnimator', timer: 'Timer'):
        self.animator = animator
        self.timer = timer

        self.graph: Optional['Graph'] = None

        self.solved_path: Optional[list['Node']] = None

    def solve_maze(self, start_pos: tuple[int, int], end_pos: tuple[int, int], algorithm: str, graph: 'Graph'):
        """Solve a maze using the specified algorithm."""
        logging.debug(f"Solving maze with {algorithm} algorithm, {start_pos} {end_pos}")

        self.graph = graph
        self.solved_path = None

        if algorithm == "DFS":
            solver = DfsSolver(self.graph, self.animator, self.timer)
            path_names_array = solver.solve(start_pos, end_pos)
            
            self.solved_path = list(map(lambda path_name: self.graph.nodes[path_name], path_names_array))

        elif algorithm == "BFS":
            solver = BfsSolver(self.graph, self.animator, self.timer)
            path_names_array = solver.solve(start_pos, end_pos)

            self.solved_path = list(map(lambda path_name: self.graph.nodes[path_name], path_names_array))
