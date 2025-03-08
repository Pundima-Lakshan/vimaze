import logging
from typing import Optional, TYPE_CHECKING

from customtkinter import CTkCanvas

from vimaze.animator import MazeAnimator
from vimaze.configs import maze_animator_options
from vimaze.display import MazeDisplay
from vimaze.generator import MazeGraphGenerator
from vimaze.image_processor import MazeImageProcessor
from vimaze.solver import MazeSolver
from vimaze.timer import Timer

if TYPE_CHECKING:
    from vimaze.ds.graph import Graph
    from vimaze.ds.graph import Node
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
        
        self.start_pos: Optional[tuple[int, int]] = None
        self.end_pos: Optional[tuple[int, int]] = None

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
        from vimaze.image_processor import MazeImageProcessor
        # Create a processor and process the image
        processor = MazeImageProcessor(self.timer)
        self.graph, self.rows, self.cols = processor.process_image(image_path)
        # Display the maze
        self.display_maze()

    def init_from_image_with_params(self, image_path: str, invert_binary: bool = False, wall_threshold: int = 127,
                                    debug_mode: bool = False, cell_size: int = 20):
        """
        Initialize maze from an image with custom parameters.
        
        Args:
            image_path: Path to the maze image file
            invert_binary: Whether to invert the binary image
            wall_threshold: Threshold for wall detection (0-255)
            debug_mode: Whether to save debug visualizations
            cell_size: Size of cells (for simple processor)
        """
        # Log the parameters
        print(f"Processing image: {image_path}")
        print(f"Parameters: invert_binary={invert_binary}, wall_threshold={wall_threshold}, debug_mode={debug_mode}")

        processor = MazeImageProcessor(self.timer)
        
        processor.wall_threshold = wall_threshold
        processor.debug_mode = debug_mode
        processor.adaptive_threshold = True

        # Process the image
        self.graph, self.rows, self.cols, self.start, self.end = processor.process_image(image_path)

        # Display the maze
        self.maze_canvas.delete("all")  # Clear existing content
        self.display_maze()

    def set_maze_gen_algorithm(self, value: str):
        self.gen_algorithm = value

    def set_maze_solving_algorithm(self, value: str):
        self.solving_algorithm = value

    def display_maze(self, start_pos: Optional[tuple[int, int]] = None, end_pos: Optional[tuple[int, int]] = None):
        self.displayer.display_maze(self)
        if start_pos is not None and end_pos is not None:
            self.displayer.display_start_end(start_pos, end_pos)

    def display_path(self, nodes: list['Node']):
        self.displayer.display_path(nodes, maze_animator_options['solving']['defaults']['path_color'],
                                    maze_animator_options['solving']['defaults']['start_color'],
                                    maze_animator_options['solving']['defaults']['end_color'])

    def solve_maze(self, start_pos: tuple[int, int], end_pos: tuple[int, int]):
        start_x, start_y = start_pos
        end_x, end_y = end_pos

        if not (0 <= start_x < self.rows and 0 <= end_x < self.rows and
                0 <= start_y < self.rows and 0 <= end_y < self.rows):
            logging.debug('Start and end positions are out of bound')
            return False

        self.start_pos = start_pos
        self.end_pos = end_pos

        self.solver.solve_maze(start_pos, end_pos, self.solving_algorithm, self.graph)
        return True
