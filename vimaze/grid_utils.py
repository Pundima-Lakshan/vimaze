from vimaze.graph import Graph, Node


def create_grid(rows, cols):
    graph = Graph()

    # Create the nodes for the grid
    for row in range(rows):
        for col in range(cols):
            node = Node(get_cell_value(row, col), row, col)
            graph.add_node(node)

    return graph
