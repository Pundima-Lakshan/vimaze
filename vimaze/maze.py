from typing import Optional, TYPE_CHECKING

from customtkinter import CTkCanvas

from vimaze.configs import maze_animator_options
from vimaze.maze_animator import MazeAnimator
from vimaze.maze_display import MazeDisplay
from vimaze.maze_graph_generator import MazeGraphGenerator
from vimaze.maze_solver import MazeSolver
from vimaze.timer import Timer

if TYPE_CHECKING:
    from vimaze.graph import Graph
    from vimaze.graph import Node
    from vimaze.app import SolverApp


class Maze:
    def __init__(self, maze_canvas: CTkCanvas, app: 'SolverApp'):
        self.maze_canvas = maze_canvas

        self.app = app

        self.graph: Optional['Graph'] = None

        self.gen_algorithm: Optional[str] = None
        self.solving_algorithm: Optional[str] = None

        self.cols: Optional[int] = None
        self.rows: Optional[int] = None

        self.timer = Timer(self)
        self.displayer = MazeDisplay(self.maze_canvas)
        self.animator = MazeAnimator(self.maze_canvas, self.displayer, self)
        self.generator = MazeGraphGenerator(self.animator, self.timer)
        self.solver = MazeSolver(self.animator, self.timer)

    def gen_algo_maze(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.graph = self.generator.generate_maze_graph(rows, cols, self.gen_algorithm)
        
    def init_from_image(self, image_path: str):
        """
        Initialize maze from an image.
        
        Args:
            image_path: Path to the maze image file
        """
        from vimaze.maze_image_processor import MazeImageProcessor
        # Create a processor and process the image
        processor = MazeImageProcessor(self.timer)
        self.graph, self.rows, self.cols = processor.process_image(image_path)
        # Display the maze
        self.display_maze()
        
    def init_from_image_with_params(self, image_path: str, invert_binary: bool = False, 
                            wall_threshold: int = 127, debug_mode: bool = False):
        """
        Initialize maze from an image with custom parameters.
        
        Args:
            image_path: Path to the maze image file
            invert_binary: Whether to invert the binary image
            wall_threshold: Threshold for wall detection (0-255)
            debug_mode: Whether to save debug visualizations
        """
        from vimaze.maze_image_processor import MazeImageProcessor
        
        # Create a processor with custom parameters
        processor = MazeImageProcessor(self.timer)
        
        # Set parameters
        processor.invert_binary = invert_binary
        processor.wall_threshold = wall_threshold
        processor.debug_mode = debug_mode
        
        # Process the image
        self.graph, self.rows, self.cols = processor.process_image(image_path)
        
        # Display the maze
        self.maze_canvas.delete("all")  # Clear existing content
        self.display_maze()

    def set_maze_gen_algorithm(self, value: str):
        self.gen_algorithm = value

    def set_maze_solving_algorithm(self, value: str):
        self.solving_algorithm = value

    def display_maze(self):
        self.displayer.display_maze(self)

    def display_path(self, nodes: list['Node']):
        self.displayer.display_path(nodes, maze_animator_options['solving']['defaults']['path_color'],
                                    maze_animator_options['solving']['defaults']['start_color'],
                                    maze_animator_options['solving']['defaults']['end_color'])

    def solve_maze(self, start_pos: tuple[int, int], end_pos: tuple[int, int]):
        self.solver.solve_maze(start_pos, end_pos, self.solving_algorithm, self.graph)
