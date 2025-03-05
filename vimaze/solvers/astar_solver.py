import heapq
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vimaze.ds.graph import Graph
    from vimaze.animator import MazeAnimator
    from vimaze.timer import Timer

    # A * Search Algorithm
    # 1.Initialize the open list
    # 2.Initialize the closed list put the starting node on the open list (you can leave its f at zero)
    # 3.while the open list is not empty
    #     a) find the node with the least f on
    #         the open list, call it "q"
    #         b) pop q off the open list
    #         c) generate q's 8 successors and set their parents to q
    #         d) for each successor
    #             i) if successor is the goal, stop search
    #
    #             ii) else, compute both g and h for successor
    #
    #             successor.g = q.g + distance between successor and q
    #             successor.h = distance from goal to successor (This can be done using many
    #             ways, we will discuss three heuristics- Manhattan, Diagonal and Euclidean Heuristics)
    #
    #             successor.f = successor.g + successor.h
    #
    #             iii) if a node with the same position as successor is in the OPEN list which has a
    #             lower f than successor, skip this successor
    #
    #             iV) if a node with the same position as successor is in the CLOSED list which has
    #             a lower f than successor, skip this successor otherwise, add  the node to the open list
    #             end ( for loop)
    #
    #         e) push q on the closed list
    #         end (while loop)


def heuristic(node_pos: tuple[int, int], goal_pos: tuple[int, int]) -> int:
    """Calculate the Manhattan distance heuristic."""
    return abs(node_pos[0] - goal_pos[0]) + abs(node_pos[1] - goal_pos[1])


class AStarSolver:
    def __init__(self, graph: 'Graph', animator: 'MazeAnimator', timer: 'Timer'):
        self.graph = graph
        self.animator = animator
        self.timer = timer

    def solve(self, start_pos: tuple[int, int], end_pos: tuple[int, int]):
        self.animator.start_recording('solving', 'astar')
        self.timer.start('solving', 'astar')

        # Initialize open list (priority queue) and closed list
        open_list = []
        closed_list = set()

        # Initialize g, h, and f scores
        g_scores = {node.name: sys.maxsize for node in self.graph.nodes.values()}
        f_scores = {node.name: sys.maxsize for node in self.graph.nodes.values()}

        # Start node
        start_node = self.graph.get_node(start_pos)
        g_scores[start_node.name] = 0
        f_scores[start_node.name] = heuristic(start_pos, end_pos)

        # Push start node to the open list
        heapq.heappush(open_list, (f_scores[start_node.name], start_node.name))
        self.animator.add_step_cell(start_node, 'search_start_node')

        # Path reconstruction map
        came_from = {start_node.name: None}

        while open_list:
            # Get the node with the least f score
            current_f, current_name = heapq.heappop(open_list)
            current_node = self.graph.nodes[current_name]

            if current_name == self.graph.get_node(end_pos).name:
                # Reconstruct path
                path = []
                node_name = current_name
                while node_name is not None:
                    path.append(node_name)
                    node_name = came_from[node_name]
                path.reverse()
                self.animator.add_step_cell(current_node, 'search_end_node')
                self.timer.stop()
                return path

            closed_list.add(current_name)
            # self.animator.add_step_cell(current_node, 'pq_pop')

            for neighbor in current_node.neighbors:
                if neighbor.name in closed_list:
                    continue

                tentative_g_score = g_scores[current_name] + 1  # Assuming uniform cost

                if neighbor.name not in [name for (_, name) in open_list]:
                    heapq.heappush(open_list, (tentative_g_score + heuristic(neighbor.position, end_pos), neighbor.name))
                elif tentative_g_score >= g_scores[neighbor.name]:
                    continue

                came_from[neighbor.name] = current_name
                g_scores[neighbor.name] = tentative_g_score
                f_scores[neighbor.name] = tentative_g_score + heuristic(neighbor.position, end_pos)
                self.animator.add_step_cell(neighbor, 'add_neighbour')

        self.timer.stop()
        return []  # No path found
