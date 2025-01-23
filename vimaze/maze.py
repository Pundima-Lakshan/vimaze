from customtkinter import CTkCanvas

from vimaze.maze_display import MazeDisplay
from vimaze.maze_generator import MazeGenerator


class Maze:
    def __init__(self, maze_canvas: CTkCanvas):
        self.maze_canvas = maze_canvas
        self.graph = None
        self.gen_algorithm = None
        self.cols = None
        self.rows = None

        self.displayer = MazeDisplay(self)

    def gen_algo_maze(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.graph = MazeGenerator.generate_maze(rows, cols, self.gen_algorithm)

    def set_maze_gen_algorithm(self, value):
        self.gen_algorithm = value

    def display_maze(self):
        self.displayer.display_maze(self.maze_canvas)
