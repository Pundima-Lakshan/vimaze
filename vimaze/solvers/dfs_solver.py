from collections import deque
from typing import TYPE_CHECKING, Optional

from vimaze.ds.indexed_set import IndexedSet

if TYPE_CHECKING:
    from vimaze.graph import Graph
    from vimaze.maze_animator import MazeAnimator
    from vimaze.timer import Timer


class DfsSolver:
    def __init__(self, graph: 'Graph', animator: 'MazeAnimator', timer: 'Timer'):
        self.graph = graph

        self.animator = animator
        self.timer = timer

    # Pseudocode
    # 
    # Search
    # Initialize stack
    # Initialize visited list
    # Initialize path map
    # Put the starting node to visited list and stack
    # While stack is not empty
    #   Get unvisited adjacent nodes 'an' of node at top of stack 's'
    #   If length of 'an' is 0
    #       Mark 's' as fully visited
    #       Pop 's'
    #   Else
    #       Put 'a' element from 'an' to stack
    #       Mark 'a' as visited
    #       If 'a' is end node
    #           break
    # 
    # Backtrack
    # Initialize path array 'p'
    # Put end node 'e' to it
    # While parent of last element in 'p' is not None
    #   Put that parent to path array
    
    def solve(self, start_pos: tuple[int, int], end_pos: tuple[int, int]):
        self.animator.start_recording('solving', 'dfs')
        self.timer.start('solving', 'dfs')
        
        visited_names: IndexedSet[str] = IndexedSet()
        names_stack: deque[str] = deque()
        path_names_map: dict[str, Optional[str]] = {}

        names_stack.append(self.graph.get_node(start_pos).name)
        visited_names.add(self.graph.get_node(start_pos).name)
        self.animator.add_step_cell(self.graph.get_node(start_pos), 'search_start_node')
        
        path_names_map[self.graph.get_node(start_pos).name] = None

        def get_unvisited_adjacent(node_name: str):
            adjacent_s = []
            for neighbour in self.graph.nodes[node_name].neighbors:
                if not visited_names.lookup(neighbour.name):
                    adjacent_s.append(neighbour)

            return adjacent_s

        while len(names_stack) != 0:
            s_name = names_stack[-1]
            an = get_unvisited_adjacent(s_name)
            if len(an) == 0:
                popped_name = names_stack.pop()
                self.animator.add_step_cell(self.graph.nodes[popped_name], 'fully_visited_update')
            else:
                new_visit_name = an.pop().name
                path_names_map[new_visit_name] = s_name
                names_stack.append(new_visit_name)
                
                visited_names.add(new_visit_name)
                self.animator.add_step_cell(self.graph.nodes[new_visit_name], 'visited_update')
                
                if new_visit_name == self.graph.get_node(end_pos).name:
                    self.animator.add_step_cell(self.graph.nodes[new_visit_name], 'search_end_node')
                    break
        
        path_names_array: list[str] = [self.graph.get_node(end_pos).name]

        while path_names_map[path_names_array[-1]] is not None:
            parent = path_names_map[path_names_array[-1]]
            path_names_array.append(parent)
            self.animator.add_step_cell(self.graph.nodes[parent], 'backtrack_path')

        self.animator.add_step_cell(self.graph.nodes[path_names_array[-1]], 'search_start_node')

        self.timer.stop()
        
        return path_names_array

        
