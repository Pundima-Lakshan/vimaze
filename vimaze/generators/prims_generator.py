import random
from typing import TYPE_CHECKING

from vimaze.ds.indexed_set import IndexedSet

if TYPE_CHECKING:
    from vimaze.graph import Node
    from vimaze.graph import Graph
    from vimaze.maze_animator import MazeAnimator
    from vimaze.timer import Timer


class PrimsGenerator:
    def __init__(self, rows: int, cols: int, graph: 'Graph', animator: 'MazeAnimator', timer: 'Timer'):
        self.rows = rows
        self.cols = cols

        self.graph = graph
        self.animator = animator
        self.timer = timer

        self.frontier_names: IndexedSet[str] = IndexedSet()
        self.visited_names: IndexedSet[str] = IndexedSet()

    # Pseudocode
    # Initialize all cells to have walls
    # Pick cell c at random and mark it as visited
    # Get frontiers s of c and add to set fs that contains all frontier cells
    # while fs is not empty:
    #   Pick a random cell c from fs and remove it from fs
    #   Get neighbours ns of c that are in the maze
    #   Connect c with random neighbour (nx, ny) from ns
    #   Add the frontier s of c to fs
    #   Mark c as visited

    def generate_maze(self):
        self.animator.start_recording('generation', 'prims')
        self.timer.start('generation', 'prims')

        start_node = self.graph.get_node((0, 0))

        self.visited_names.add(start_node.name)
        self.animator.add_step_cell(start_node, 'visited_update')

        self.update_frontier_cells(start_node)

        while len(self.frontier_names) != 0:
            selected_frontier_cell_name = PrimsGenerator.pop_a_random_element(self.frontier_names)
            self.animator.add_step_cell(self.graph.nodes[selected_frontier_cell_name], 'frontier_select')

            selected_maze_cell_name = PrimsGenerator.pop_a_random_element(self.get_possible_4_adjacent_maze_nodes(
                self.graph.nodes[selected_frontier_cell_name])).name
            self.animator.add_step_cell(self.graph.nodes[selected_maze_cell_name], 'maze_cell_select')
            self.animator.add_step_cell(self.graph.nodes[selected_maze_cell_name], 'maze_cell_deselect')

            self.graph.connect_nodes(self.graph.nodes[selected_frontier_cell_name].position,
                                     self.graph.nodes[selected_maze_cell_name].position)
            self.animator.add_step_edge(
                [self.graph.nodes[selected_frontier_cell_name], self.graph.nodes[selected_maze_cell_name]],
                'node_connect')
            
            self.update_frontier_cells(self.graph.nodes[selected_frontier_cell_name])

            self.visited_names.add(selected_frontier_cell_name)
            self.animator.add_step_cell(self.graph.nodes[selected_frontier_cell_name], 'visited_update')

        self.timer.stop()
        
        return self.graph

    def update_frontier_cells(self, node: 'Node'):
        for neighbour in self.get_possible_frontier_nodes(node):
            self.frontier_names.add(neighbour.name)
            self.animator.add_step_cell(neighbour, 'frontier_update')

    def get_possible_frontier_nodes(self, node: 'Node'):
        nodes: list['Node'] = []

        for neighbour in self.get_4_adjacent_neighbours(node):
            if not self.visited_names.lookup(neighbour.name):
                nodes.append(neighbour)

        return nodes

    def get_possible_4_adjacent_maze_nodes(self, node: 'Node'):
        nodes: IndexedSet['Node'] = IndexedSet()

        for neighbour in self.get_4_adjacent_neighbours(node):
            if self.visited_names.lookup(neighbour.name):
                nodes.add(neighbour)

        return nodes

    def get_4_adjacent_neighbours(self, node: 'Node'):
        row, col = node.position

        neighbours: list['Node'] = []

        if col - 1 >= 0:
            neighbour = self.graph.get_node((row, col - 1))
            neighbours.append(neighbour)
        if row + 1 <= self.rows - 1:
            neighbour = self.graph.get_node((row + 1, col))
            neighbours.append(neighbour)
        if col + 1 <= self.cols - 1:
            neighbour = self.graph.get_node((row, col + 1))
            neighbours.append(neighbour)
        if row - 1 >= 0:
            neighbour = self.graph.get_node((row - 1, col))
            neighbours.append(neighbour)

        return neighbours

    @staticmethod
    def pop_a_random_element(node_set: IndexedSet):
        random_index = random.randint(0, len(node_set) - 1)
        return node_set.pop_at(random_index)
