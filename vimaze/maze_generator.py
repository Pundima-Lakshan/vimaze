import random

from vimaze.graph import Graph, Node


class MazeGenerator:
    @staticmethod
    def generate_maze(rows, cols, algorithm):
        graph = Graph()
        for row in range(rows):
            for col in range(cols):
                node = Node(Graph.get_cell_value(row, col), row, col)
                graph.add_node(node)

        # if algorithm == "prim":
        if True:
            # Initialize Prim's algorithm
            visited = set()  # Tracks cells added to the maze
            walls = []       # Tracks walls between visited and unvisited cells
    
            # Start with a random cell (e.g., top-left corner)
            start_value = Graph.get_cell_value(0, 0)
            start_node = graph.get_node(start_value)
            visited.add(start_value)
    
            # Add initial walls (edges between the starting cell and its neighbors)
            for neighbor_value in MazeGenerator._get_adjacent_nodes(start_node, rows, cols):
                walls.append((start_value, neighbor_value))
    
            # Process walls until none remain
            while walls:
                # Pick a random wall
                wall_idx = random.randint(0, len(walls) - 1)
                u_value, v_value = walls.pop(wall_idx)
    
                # Check if one cell is visited and the other is not
                u_visited = u_value in visited
                v_visited = v_value in visited
    
                if u_visited != v_visited:
                    # Connect the two cells (carve a passage)
                    graph.connect_nodes(u_value, v_value)
    
                    # Determine the newly added cell
                    new_value = v_value if not v_visited else u_value
                    new_node = graph.get_node(new_value)
                    visited.add(new_value)
    
                    # Add new walls from the newly added cell
                    for neighbor_value in MazeGenerator._get_adjacent_nodes(new_node, rows, cols):
                        if neighbor_value not in visited:
                            walls.append((new_value, neighbor_value))
    
        return graph

    @staticmethod
    def _get_adjacent_nodes(node, rows, cols):
        """Returns the values of adjacent nodes within grid bounds."""
        directions = [
            (-1, 0),  # North
            (1, 0),   # South
            (0, 1),   # East
            (0, -1)   # West
        ]
        neighbors = []
        for dr, dc in directions:
            new_row = node.row + dr
            new_col = node.col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                neighbors.append(Graph.get_cell_value(new_row, new_col))
        return neighbors
