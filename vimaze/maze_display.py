from customtkinter import CTkCanvas

from vimaze.configs import maze_display_options
from vimaze.graph import Graph


class MazeDisplay:
    def __init__(self, maze):
        self.maze = maze

        self.speed = maze_display_options["speed"]
        self.cell_size = maze_display_options["cell_size"]
        self.offset = maze_display_options["offset"]
        self.outline_width = maze_display_options["outline_width"]
        self.cell_color = maze_display_options["cell_color"]
        self.cell_outline = maze_display_options["cell_outline"]
        self.wall_color = maze_display_options["wall_color"]
        self.starting_coords = None

    def adjust_cell_size(self, canvas: CTkCanvas):
        canvas_width = canvas.winfo_width() - 2 * self.offset
        canvas_height = canvas.winfo_height() - 2 * self.offset

        self.cell_size = min(canvas_width // self.maze.cols, canvas_height // self.maze.rows)

    def adjust_starting_coords(self, canvas: CTkCanvas):
        maze_width = self.maze.cols * self.cell_size
        maze_height = self.maze.rows * self.cell_size

        self.starting_coords = (
            (canvas.winfo_width() - maze_width) // 2,
            (canvas.winfo_height() - maze_height) // 2
        )

    def display_maze(self, canvas: CTkCanvas):
        self.adjust_cell_size(canvas)
        self.adjust_starting_coords(canvas)

        for node_value, node in self.maze.graph.nodes.items():
            row, col = Graph.get_row_col_from_value(node_value)
            x = (col * self.cell_size) + self.starting_coords[0]
            y = (row * self.cell_size) + self.starting_coords[1]

            # Draw the cell
            canvas.create_rectangle(x, y, x + self.cell_size, y + self.cell_size, outline=self.cell_outline,
                                    width=self.outline_width,
                                    fill=self.cell_color)

            # Draw walls (no edge = wall)
            if not node.get_edge("north"):
                canvas.create_line(x, y, x + self.cell_size, y, fill=self.wall_color,
                                   width=self.outline_width)  # North wall
            if not node.get_edge("west"):
                canvas.create_line(x, y, x, y + self.cell_size, fill=self.wall_color,
                                   width=self.outline_width)  # West wall
            if not node.get_edge("south"):
                canvas.create_line(x, y + self.cell_size, x + self.cell_size, y + self.cell_size, fill=self.wall_color,
                                   width=self.outline_width)  # South wall
            if not node.get_edge("east"):
                canvas.create_line(x + self.cell_size, y, x + self.cell_size, y + self.cell_size, fill=self.wall_color,
                                   width=self.outline_width)  # East wall
