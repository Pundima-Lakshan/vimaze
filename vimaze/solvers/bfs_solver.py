from collections import deque
from typing import TYPE_CHECKING, Optional

from vimaze.ds.indexed_set import IndexedSet

if TYPE_CHECKING:
    from vimaze.graph import Graph
    from vimaze.maze_animator import MazeAnimator
    from vimaze.timer import Timer


class BfsSolver:
    def __init__(self, graph: 'Graph', animator: 'MazeAnimator', timer: 'Timer'):
        self.graph = graph
        self.animator = animator
        self.timer = timer

    def solve(self, start_pos: tuple[int, int], end_pos: tuple[int, int]):
        self.animator.start_recording('solving', 'bfs')
        self.timer.start('solving', 'bfs')

        visited_names: IndexedSet[str] = IndexedSet()
        names_queue: deque[str] = deque()
        path_names_map: dict[str, Optional[str]] = {}

        start_node_name = self.graph.get_node(start_pos).name
        names_queue.append(start_node_name)
        visited_names.add(start_node_name)
        self.animator.add_step_cell(self.graph.get_node(start_pos), 'search_start_node')

        path_names_map[start_node_name] = None

        while names_queue:
            curr_name = names_queue.popleft()
            self.animator.add_step_cell(self.graph.nodes[curr_name], 'visited_update')

            if curr_name == self.graph.get_node(end_pos).name:
                self.animator.add_step_cell(self.graph.nodes[curr_name], 'search_end_node')
                break

            for neighbour in self.graph.nodes[curr_name].neighbors:
                if not visited_names.lookup(neighbour.name):
                    visited_names.add(neighbour.name)
                    names_queue.append(neighbour.name)
                    path_names_map[neighbour.name] = curr_name

        path_names_array: list[str] = [self.graph.get_node(end_pos).name]

        while path_names_map[path_names_array[-1]] is not None:
            parent = path_names_map[path_names_array[-1]]
            path_names_array.append(parent)
            self.animator.add_step_cell(self.graph.nodes[parent], 'backtrack_path')

        self.animator.add_step_cell(self.graph.nodes[path_names_array[-1]], 'search_start_node')

        self.timer.stop()

        return path_names_array
