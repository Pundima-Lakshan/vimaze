import logging
import random
from typing import TYPE_CHECKING

from vimaze.graph import Graph

if TYPE_CHECKING:
    from vimaze.maze_animator import MazeAnimator
    from vimaze.graph import Node

class MazeGenerator:
    def __init__(self, animator: 'MazeAnimator'):
        self.animator = animator
    
    def generate_maze(self, rows: int, cols: int, algorithm: str):
        """Generates a maze using the specified algorithm."""
        logging.debug(f"Generating maze with {algorithm} algorithm")

        graph = Graph()
        for row in range(rows):
            for col in range(cols):
                graph.add_node((row, col))

        if algorithm == "Prim\'s":
            # Initialize Prim's algorithm
            visited = set()  # Tracks cells added to the maze
            walls: list[tuple[str, str]] = []  # Tracks walls between visited and unvisited cells

            # Start with a random cell (e.g., top-left corner)
            start_node_name = Graph.get_node_name((0, 0))
            start_node = graph.get_node((0, 0))
            visited.add(start_node_name)

            # Add initial walls (edges between the starting cell and its neighbors)
            for neighbor_node_name in MazeGenerator._get_adjacent_nodes(start_node, rows, cols):
                walls.append((start_node_name, neighbor_node_name))

            # Process walls until none remain
            while walls:
                # Pick a random wall
                wall_idx = random.randint(0, len(walls) - 1)
                u_node_name, v_node_name = walls.pop(wall_idx)

                # Check if one cell is visited and the other is not
                u_visited = u_node_name in visited
                v_visited = v_node_name in visited

                if u_visited != v_visited:
                    # Connect the two cells (carve a passage)
                    graph.connect_nodes(graph.nodes[u_node_name].position, graph.nodes[v_node_name].position)

                    # Determine the newly added cell
                    new_node_name = v_node_name if not v_visited else u_node_name
                    new_node = graph.nodes[new_node_name]
                    visited.add(new_node_name)

                    # Add new walls from the newly added cell
                    for neighbor_node_name in MazeGenerator._get_adjacent_nodes(new_node, rows, cols):
                        if neighbor_node_name not in visited:
                            walls.append((new_node_name, neighbor_node_name))

        return graph

    @staticmethod
    def _get_adjacent_nodes(node: 'Node', rows: int, cols: int):
        """Returns the values of adjacent nodes within grid bounds."""
        directions = [
            (-1, 0),  # North
            (1, 0),  # South
            (0, 1),  # East
            (0, -1)  # West
        ]
        neighbors = []
        for dr, dc in directions:
            new_row = node.position[0] + dr
            new_col = node.position[1] + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                neighbors.append(Graph.get_node_name((new_row, new_col)))
        return neighbors
