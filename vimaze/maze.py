from typing import Optional, TYPE_CHECKING

from customtkinter import CTkCanvas

from vimaze.maze_animator import MazeAnimator
from vimaze.maze_display import MazeDisplay
from vimaze.maze_generator import MazeGenerator
from vimaze.maze_solver import MazeSolver

if TYPE_CHECKING:
    from vimaze.graph import Graph


class Maze:
    def __init__(self, maze_canvas: CTkCanvas):
        self.maze_canvas = maze_canvas

        self.graph: Optional['Graph'] = None
        self.gen_algorithm: Optional[str] = None
        self.cols: Optional[int] = None
        self.rows: Optional[int] = None

        self.animator = MazeAnimator(self.maze_canvas)
        self.generator = MazeGenerator(self.animator)
        self.displayer = MazeDisplay()
        self.solver = MazeSolver(self.graph, self.animator)

    def gen_algo_maze(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.graph = self.generator.generate_maze(rows, cols, self.gen_algorithm)

    def set_maze_gen_algorithm(self, value: str):
        self.gen_algorithm = value

    def display_maze(self):
        self.displayer.display_maze(self, self.maze_canvas)

    def animate_last_operation(self):
        self.animator.animate()
