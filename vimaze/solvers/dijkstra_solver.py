import heapq
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vimaze.ds.graph import Graph
    from vimaze.animator import MazeAnimator
    from vimaze.timer import Timer


class DijkstraSolver:
    def __init__(self, graph: 'Graph', animator: 'MazeAnimator', timer: 'Timer'):
        self.graph = graph

        self.animator = animator
        self.timer = timer

    def solve(self, start_pos: tuple[int, int], end_pos: tuple[int, int]):
        self.animator.start_recording('solving', 'dijkstra')
        self.timer.start('solving', 'dijkstra')

        path_names_map: dict[str, Optional[str]] = {}
        pq_names: list[tuple[int, str]] = []
        names_dist: dict[str, int] = {node.name: sys.maxsize for node in self.graph.nodes.values()}

        path_names_map[self.graph.get_node(start_pos).name] = None
        heapq.heappush(pq_names, (0, self.graph.get_node(start_pos).name))
        names_dist[self.graph.get_node(start_pos).name] = 0
        self.animator.add_step_cell(self.graph.get_node(start_pos), 'search_start_node')

        while pq_names:
            d, x_name = heapq.heappop(pq_names)
            if x_name == self.graph.get_node(start_pos).name:
                self.animator.add_step_cell(self.graph.nodes[x_name], 'search_start_node')
            else:
                self.animator.add_step_cell(self.graph.nodes[x_name], 'pq_pop')

            for an in self.graph.nodes[x_name].neighbors:
                an_dist = names_dist[an.name]
                x_an_dist = names_dist[x_name] + 1
                if an_dist > x_an_dist:
                    names_dist[an.name] = x_an_dist
                    path_names_map[an.name] = x_name
                    heapq.heappush(pq_names, (x_an_dist, an.name))
                    self.animator.add_step_cell(an, 'pq_push')

        path_names_array: list[str] = [self.graph.get_node(end_pos).name]
        self.animator.add_step_cell(self.graph.get_node(end_pos), 'search_end_node')

        while path_names_map[path_names_array[-1]] is not None:
            parent = path_names_map[path_names_array[-1]]
            path_names_array.append(parent)
            self.animator.add_step_cell(self.graph.nodes[parent], 'backtrack_path')

        self.animator.add_step_cell(self.graph.nodes[path_names_array[-1]], 'search_start_node')

        self.timer.stop()

        return path_names_array
