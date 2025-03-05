from typing import TYPE_CHECKING

from vimaze.configs import maze_display_options
from vimaze.ds.graph import Graph

if TYPE_CHECKING:
    from vimaze.maze import Maze
    from vimaze.ds.graph import Node
    from customtkinter import CTkCanvas


class MazeDisplay:
    def __init__(self, canvas: 'CTkCanvas'):
        self.speed = maze_display_options["speed"]
        self.cell_size = maze_display_options["cell_size"]
        self.offset = maze_display_options["offset"]
        self.outline_width = maze_display_options["outline_width"]
        self.cell_color = maze_display_options["cell_color"]
        self.cell_outline = maze_display_options["cell_outline"]
        self.wall_color = maze_display_options["wall_color"]
        self.starting_coords = None

        self.canvas = canvas

    def adjust_cell_size(self, size: tuple[int, int]):
        canvas_width = self.canvas.winfo_width() - 2 * self.offset
        canvas_height = self.canvas.winfo_height() - 2 * self.offset

        self.cell_size = min(canvas_width // size[1], canvas_height // size[0])

    def adjust_starting_coords(self, size: tuple[int, int]):
        maze_width = size[1] * self.cell_size
        maze_height = size[0] * self.cell_size

        self.starting_coords = (
            (self.canvas.winfo_width() - maze_width) // 2,
            (self.canvas.winfo_height() - maze_height) // 2
        )

    def display_maze(self, maze: 'Maze', cell_fill: str = None):
        self.display_graph(maze.rows, maze.cols, maze.graph, cell_fill)

    def reset_maze_display(self, maze: 'Maze', cell_fill=None):
        rows, cols = maze.rows, maze.cols
        empty_graph = Graph()
        for row in range(rows):
            for col in range(cols):
                empty_graph.add_node((row, col))

        self.display_graph(maze.rows, maze.cols, empty_graph, cell_fill)

    def display_graph(self, rows: int, cols: int, graph: 'Graph', cell_fill: str = None):
        self.adjust_cell_size((rows, cols))
        self.adjust_starting_coords((rows, cols))

        if cell_fill is None:
            cell_fill = self.cell_color

        for node_value, node in graph.nodes.items():
            row, col = node.position
            x = (col * self.cell_size) + self.starting_coords[0]
            y = (row * self.cell_size) + self.starting_coords[1]

            # Draw the cell
            self.canvas.create_rectangle(x, y, x + self.cell_size, y + self.cell_size, outline=self.cell_outline,
                                         width=self.outline_width,
                                         fill=cell_fill)

            # Draw walls (no edge = wall)
            if row - 1 >= 0 and not node.is_neighbour(graph.get_node((row - 1, col))):
                self.canvas.create_line(x, y, x + self.cell_size, y, fill=self.wall_color,
                                        width=self.outline_width)  # North wall
            if col - 1 >= 0 and not node.is_neighbour(graph.get_node((row, col - 1))):
                self.canvas.create_line(x, y, x, y + self.cell_size, fill=self.wall_color,
                                        width=self.outline_width)  # West wall
            if row + 1 <= rows - 1 and not node.is_neighbour(graph.get_node((row + 1, col))):
                self.canvas.create_line(x, y + self.cell_size, x + self.cell_size, y + self.cell_size,
                                        fill=self.wall_color,
                                        width=self.outline_width)  # South wall
            if col + 1 <= cols - 1 and not node.is_neighbour(graph.get_node((row, col + 1))):
                self.canvas.create_line(x + self.cell_size, y, x + self.cell_size, y + self.cell_size,
                                        fill=self.wall_color,
                                        width=self.outline_width)  # East wall

        # Draw borders
        x = self.starting_coords[0]
        y = self.starting_coords[1]

        self.canvas.create_line(x, y, x + self.cell_size * rows, y, fill=self.wall_color,
                                width=self.outline_width)  # North wall
        self.canvas.create_line(x, y, x, y + self.cell_size * cols, fill=self.wall_color,
                                width=self.outline_width)  # West wall
        self.canvas.create_line(x, y + self.cell_size * cols, x + self.cell_size * rows, y + self.cell_size * cols,
                                fill=self.wall_color,
                                width=self.outline_width)  # South wall
        self.canvas.create_line(x + self.cell_size * rows, y, x + self.cell_size * rows, y + self.cell_size * cols,
                                fill=self.wall_color,
                                width=self.outline_width)

    def display_cell(self, row: int, col: int, color: str):
        x = (col * self.cell_size) + self.starting_coords[0]
        y = (row * self.cell_size) + self.starting_coords[1]

        offset = self.outline_width

        self.canvas.create_rectangle(x + offset, y + offset, x + self.cell_size - offset, y + self.cell_size - offset,
                                     outline=color,
                                     width=self.outline_width,
                                     fill=color)

    def display_path(self, nodes: list['Node'], path_color: str, start_color: str, end_color: str):
        for index, node in enumerate(nodes):
            row, col = node.position

            color = path_color
            if index == 0:
                color = start_color
            elif index == len(nodes) - 1:
                color = end_color

            self.display_cell(row, col, color)

    def draw_walls(self, x: int, y: int, directions: list[str], remove: bool):
        color = self.cell_outline if remove else self.wall_color

        for direction in directions:
            if direction == 'n':
                self.canvas.create_line(x, y, x + self.cell_size, y, fill=color,
                                        width=self.outline_width)  # North wall
            if direction == 'w':
                self.canvas.create_line(x, y, x, y + self.cell_size, fill=color,
                                        width=self.outline_width)  # West wall
            if direction == 's':
                self.canvas.create_line(x, y + self.cell_size, x + self.cell_size, y + self.cell_size, fill=color,
                                        width=self.outline_width)  # South wall
            if direction == 'e':
                self.canvas.create_line(x + self.cell_size, y, x + self.cell_size, y + self.cell_size, fill=color,
                                        width=self.outline_width)  # East wall

    def display_walls(self, cell_u_pos: tuple[int, int], cell_v_pos: tuple[int, int], remove: bool):
        row_u, col_u = cell_u_pos
        row_v, col_v = cell_v_pos

        x_u = (col_u * self.cell_size) + self.starting_coords[0]
        y_u = (row_u * self.cell_size) + self.starting_coords[1]

        x_v = (col_v * self.cell_size) + self.starting_coords[0]
        y_v = (row_v * self.cell_size) + self.starting_coords[1]

        wall_remove_u = None
        wall_remove_v = None

        if row_u == row_v and col_u - col_v == 1:
            wall_remove_u = 'w'
            wall_remove_v = 'e'
        elif row_u == row_v and col_u - col_v == -1:
            wall_remove_u = 'e'
            wall_remove_v = 'w'
        elif col_u == col_v and row_u - row_v == 1:
            wall_remove_u = 'n'
            wall_remove_v = 's'
        elif col_u == col_v and row_u - row_v == -1:
            wall_remove_u = 's'
            wall_remove_v = 'n'

        self.draw_walls(x_u, y_u, [wall_remove_u], remove)
        self.draw_walls(x_v, y_v, [wall_remove_v], remove)
