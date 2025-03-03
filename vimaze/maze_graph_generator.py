import logging
from typing import TYPE_CHECKING

from vimaze.generators.prims_generator import PrimsGenerator
from vimaze.graph import Graph

if TYPE_CHECKING:
    from vimaze.maze_animator import MazeAnimator
    from vimaze.graph import Node


class MazeGraphGenerator:
    def __init__(self, animator: 'MazeAnimator'):
        self.animator = animator

    def generate_maze_graph(self, rows: int, cols: int, algorithm: str):
        """Generates a maze using the specified algorithm."""
        logging.debug(f"Generating maze with {algorithm} algorithm")

        graph = Graph()
        for row in range(rows):
            for col in range(cols):
                graph.add_node((row, col))

        if algorithm == "Prim\'s":
            generator = PrimsGenerator(rows, cols, graph)
            graph = generator.generate_maze()
            
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