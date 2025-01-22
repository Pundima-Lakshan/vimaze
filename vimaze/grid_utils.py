from vimaze.graph import Graph, Node


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
