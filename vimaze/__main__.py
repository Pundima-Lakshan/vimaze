import customtkinter as ctk


class Node:
    def __init__(self, value, row, col):
        self.value = value
        self.row = row
        self.col = col
        self.edges = {
            'north': None,
            'west': None,
            'south': None,
            'east': None
        }

    def set_edge(self, direction, node):
        if direction in self.edges:
            self.edges[direction] = node

    def get_edge(self, direction):
        if direction in self.edges:
            return self.edges[direction]
        return None


class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.value] = node

    def get_node(self, value):
        return self.nodes.get(value)

    def connect_nodes(self, u_node_value, v_node_value):
        u_node = self.get_node(u_node_value)
        v_node = self.get_node(v_node_value)

        if not v_node or not u_node:
            return ValueError("Both nodes must exist in the graph", u_node, v_node)

        if not Graph.valid_4_neighbors(u_node, v_node):
            return ValueError("Nodes must be valid 4 neighbors", u_node, v_node)

        if u_node.row == v_node.row:
            if u_node.col < v_node.col:
                u_node.set_edge("east", v_node)
                v_node.set_edge("west", u_node)
            else:
                u_node.set_edge("west", v_node)
                v_node.set_edge("east", u_node)
        elif u_node.col == v_node.col:
            if u_node.row < v_node.row:
                u_node.set_edge("south", v_node)
                v_node.set_edge("north", u_node)
            else:
                u_node.set_edge("north", v_node)
                v_node.set_edge("south", u_node)

    @staticmethod
    def valid_4_neighbors(u_node, v_node):
        if u_node.row == v_node.row and abs(u_node.col - v_node.col) == 1:
            return True
        if u_node.col == v_node.col and abs(u_node.row - v_node.row) == 1:
            return True
        return False

    @staticmethod
    def get_neighbors(node):
        neighbors = []
        for direction, edge in node.edges.items():
            if edge:
                neighbors.append((direction, edge))
        return neighbors


def create_grid(rows, cols):
    graph = Graph()

    # Create the nodes for the grid
    for row in range(rows):
        for col in range(cols):
            node = Node(get_cell_value(row, col), row, col)
            graph.add_node(node)

    return graph


def get_cell_value(row, col):
    return f"{row},{col}"


def get_row_col_from_value(value):
    return map(int, value.split(","))


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


def main():
    root = ctk.CTk()
    root.title("Maze Visualization")

    # Create the user input fields for rows and columns
    def on_generate_maze():
        try:
            rows = int(entry_rows.get())
            cols = int(entry_cols.get())

            offset = 20
            width = cols * 30 + (2 * offset) - 1
            height = rows * 30 + (2 * offset) - 1

            # Set up the CustomTkinter Canvas
            canvas.config(width=width, height=height, bg="red")
            canvas.delete("all")  # Clear the canvas before drawing
            print(canvas.winfo_width(), canvas.winfo_height())

            # Create a new graph based on the rows and columns
            graph = create_grid(rows, cols)

            # Generate the maze starting from node "0,0"
            # prim_maze(graph, "0,0")

            # Draw the maze on the canvas
            draw_maze(graph=graph, canvas=canvas, _rows=rows, _cols=cols, cell_size=30, offset=offset, outline_width=2)

        except ValueError:
            print("Invalid input for rows or columns.")

    # Row and column input
    entry_rows = ctk.CTkEntry(root, placeholder_text="Rows")
    entry_rows.pack(pady=5)
    entry_cols = ctk.CTkEntry(root, placeholder_text="Columns")
    entry_cols.pack(pady=5)

    # Generate maze button
    button_generate = ctk.CTkButton(root, text="Generate Maze", command=on_generate_maze)
    button_generate.pack(pady=10)

    # Set up the CustomTkinter Canvas
    canvas = ctk.CTkCanvas(root)
    canvas.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
