import heapq
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vimaze.ds.graph import Graph
    from vimaze.animator import MazeAnimator
    from vimaze.timer import Timer


class AStarSolver:
    def __init__(self, graph: 'Graph', animator: 'MazeAnimator', timer: 'Timer'):
        self.graph = graph
        self.animator = animator
        self.timer = timer

    def _heuristic(self, a_pos: tuple[int, int], b_pos: tuple[int, int]) -> int:
        """Manhattan distance heuristic"""
        return abs(a_pos[0] - b_pos[0]) + abs(a_pos[1] - b_pos[1])

    def solve(self, start_pos: tuple[int, int], end_pos: tuple[int, int]):
        self.animator.start_recording('solving', 'astar')
        self.timer.start('solving', 'astar')

        start_node = self.graph.get_node(start_pos)
        end_node = self.graph.get_node(end_pos)

        open_heap = []
        open_set = set()
        closed_set = set()
        came_from: dict[str, Optional[str]] = {}
        g_score = {node.name: sys.maxsize for node in self.graph.nodes.values()}
        g_score[start_node.name] = 0

        f_score = g_score[start_node.name] + self._heuristic(start_node.position, end_node.position)
        heapq.heappush(open_heap, (f_score, start_node.name))
        open_set.add(start_node.name)
        self.animator.add_step_cell(start_node, 'search_start_node')
        came_from[start_node.name] = None

        while open_heap:
            current_f, current_name = heapq.heappop(open_heap)
            open_set.remove(current_name)

            if current_name in closed_set:
                continue

            closed_set.add(current_name)
            current_node = self.graph.nodes[current_name]

            if current_name == end_node.name:
                break

            self.animator.add_step_cell(current_node, 'pq_pop')

            for neighbor in current_node.neighbors:
                tentative_g = g_score[current_name] + 1  # edge weight is 1

                if tentative_g < g_score[neighbor.name]:
                    came_from[neighbor.name] = current_name
                    g_score[neighbor.name] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor.position, end_node.position)

                    # Only add the neighbor to open_heap if it's not in closed_set and in open_set
                    if neighbor.name not in closed_set and neighbor.name not in open_set:
                        heapq.heappush(open_heap, (f_score, neighbor.name))
                        open_set.add(neighbor.name)
                        self.animator.add_step_cell(neighbor, 'pq_push')

        # Path reconstruction
        path_names_array = []
        current_name = end_node.name
        while current_name is not None:
            path_names_array.append(current_name)
            self.animator.add_step_cell(self.graph.nodes[current_name], 'backtrack_path')
            current_name = came_from.get(current_name, None)

        self.animator.add_step_cell(start_node, 'search_start_node')
        self.animator.add_step_cell(end_node, 'search_end_node')

        self.timer.stop()

        return path_names_array
