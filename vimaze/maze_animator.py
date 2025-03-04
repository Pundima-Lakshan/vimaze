from typing import TYPE_CHECKING

from vimaze.configs import maze_animator_options
from vimaze.steps import Step
from vimaze.steps import Steps

if TYPE_CHECKING:
    from customtkinter import CTkCanvas
    from vimaze.graph import Node
    from vimaze.maze_display import MazeDisplay
    from vimaze.maze import Maze


class MazeAnimator:
    def __init__(self, canvas: 'CTkCanvas', displayer: 'MazeDisplay', maze: 'Maze'):
        self.steps = Steps()
        self.canvas = canvas
        self.displayer = displayer
        self.maze = maze

        self.step_index = 0
        
        self.is_stop_animation = False
        self.animation_speed = 100
        
        self.operation = None
        self.algorithm = None

    def start_recording(self, operation: str, algorithm: str):
        self.operation = operation
        self.algorithm = algorithm

        self.steps.clear_steps()

    def add_step_cell(self, node: 'Node', action: str):
        self.steps.add_step(Step([node], action, 'cell'))

    def add_step_edge(self, nodes: list['Node'], action: str):
        self.steps.add_step(Step(nodes, action, 'edge'))

    def animate(self, speed: int):
        self.is_stop_animation = False
        self.animation_speed = speed
        
        cell_fill = maze_animator_options['defaults']['cell_fill']
        
        if self.operation == 'generation':
            self.canvas.delete("all")
            self.displayer.reset_maze_display(self.maze, cell_fill)
        elif self.operation == 'solving':
            self.displayer.display_maze(self.maze, cell_fill)
        
        self.step_index = 0  # Initialize step index
    
        self.render_frame(self.step_index)
        
    def render_frame(self, step_index):
        if step_index >= len(self.steps.steps):  # Stop when all steps are done
            return
        
        if self.is_stop_animation:
            self.displayer.display_maze(self.maze)
            return
    
        step = self.steps.steps[step_index]
        
        if step.category == 'cell':
            row, col = step.nodes[0].position
            color = maze_animator_options[self.operation][self.algorithm]['action_colors'][step.action]
            self.displayer.display_cell(row, col, color)
        elif step.category == 'edge':
            node_u, node_v = step.nodes
            self.displayer.display_walls(node_u.position, node_v.position, True)
    
        # Schedule the next frame with an increased step_index
        self.canvas.after(self.animation_speed, lambda: self.render_frame(step_index + 1))
    
    def stop_animation(self):
        self.is_stop_animation = True