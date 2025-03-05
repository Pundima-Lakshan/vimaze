import heapq
from typing import TYPE_CHECKING, Optional, List, Tuple

if TYPE_CHECKING:
    from vimaze.ds.graph import Graph
    from vimaze.animator import MazeAnimator
    from vimaze.timer import Timer


class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination


class AStarSolver:
    def __init__(self, graph: 'Graph', animator: 'MazeAnimator', timer: 'Timer'):
        self.graph = graph
        self.animator = animator
        self.timer = timer

    def calculate_h_value(self, row: int, col: int, dest: Tuple[int, int]) -> float:
        return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

    def is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < len(self.graph.grid) and 0 <= col < len(self.graph.grid[0])

    def is_unblocked(self, row: int, col: int) -> bool:
        return self.graph.get_node((row, col)).is_unblocked()

    def is_destination(self, row: int, col: Tuple[int, int]) -> bool:
        return (row, col) == self.graph.dest

    def trace_path(self, cell_details: List[List[Cell]], dest: Tuple[int, int]):
        path = []
        row, col = dest

        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            path.append((row, col))
            temp_row = cell_details[row][col].parent_i
            temp_col = cell_details[row][col].parent_j
            row, col = temp_row, temp_col

        path.append((row, col))
        path.reverse()

        for step in path:
            self.animator.add_step_cell(self.graph.get_node(step), 'backtrack_path')

    def a_star_search(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]):
        if not self.is_valid(start_pos[0], start_pos[1]) or not self.is_valid(end_pos[0], end_pos[1]):
            print("Source or destination is invalid")
            return

        if not self.is_unblocked(start_pos[0], start_pos[1]) or not self.is_unblocked(end_pos[0], end_pos[1]):
            print("Source or the destination is blocked")
            return

        if self.is_destination(start_pos[0], start_pos[1], end_pos):
            print("We are already at the destination")
            return

        closed_list = [[False for _ in range(len(self.graph.grid[0]))] for _ in range(len(self.graph.grid))]
        cell_details = [[Cell() for _ in range(len(self.graph.grid[0]))] for _ in range(len(self.graph.grid))]

        i, j = start_pos
        cell_details[i][j].f = 0
        cell_details[i][j].g = 0
        cell_details[i][j].h = 0
        cell_details[i][j].parent_i = i
        cell_details[i][j].parent_j = j

        open_list = []
        heapq.heappush(open_list, (0.0, i, j))

        while open_list:
            p = heapq.heappop(open_list)
            i, j = p[1], p[2]
            closed_list[i][j] = True
            self.animator.add_step_cell(self.graph.get_node((i, j)), 'pq_pop')

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dir in directions:
                new_i, new_j = i + dir[0], j + dir[1]

                if self.is_valid(new_i, new_j) and self.is_unblocked(new_i, new_j) and not closed_list[new_i][new_j]:
                    if self.is_destination(new_i, new_j, end_pos):
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        self.animator.add_step_cell(self.graph.get_node((new_i, new_j)), 'search_end_node')
                        self.trace_path(cell_details, end_pos)
                        return

                    g_new = cell_details[i][j].g + 1.0
                    h_new = self.calculate_h_value(new_i, new_j, end_pos)
                    f_new = g_new + h_new

                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        self.animator.add_step_cell(self.graph.get_node((new_i, new_j)), 'pq_push')

        print("Failed to find the destination cell")

    def solve(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]):
        self.animator.start_recording('solving', 'A*')
        self.timer.start('solving', 'A*')
        self.a_star_search(start_pos, end_pos)
        self.timer.stop()