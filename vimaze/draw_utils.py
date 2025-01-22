import customtkinter as ctk

from vimaze.grid_utils import get_row_col_from_value


def draw_maze(graph, canvas: ctk.CTkCanvas, _rows, _cols, cell_size=30, offset=10, outline_width=2, cell_color="white",
              cell_outline="white smoke", wall_color="blue"):
    for node_value, node in graph.nodes.items():
        row, col = get_row_col_from_value(node_value)
        x = (col * cell_size) + offset
        y = (row * cell_size) + offset

        # Draw the cell
        canvas.create_rectangle(x, y, x + cell_size, y + cell_size, outline=cell_outline, width=outline_width,
                                fill=cell_color)

        # Draw walls (no edge = wall)
        if not node.get_edge("north"):
            canvas.create_line(x, y, x + cell_size, y, fill=wall_color, width=outline_width)  # North wall
        if not node.get_edge("west"):
            canvas.create_line(x, y, x, y + cell_size, fill=wall_color, width=outline_width)  # West wall
        if not node.get_edge("south"):
            canvas.create_line(x, y + cell_size, x + cell_size, y + cell_size, fill=wall_color,
                               width=outline_width)  # South wall
        if not node.get_edge("east"):
            canvas.create_line(x + cell_size, y, x + cell_size, y + cell_size, fill=wall_color,
                               width=outline_width)  # East wall
