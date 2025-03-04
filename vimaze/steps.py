from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vimaze.graph import Node


class Step:
    def __init__(self, nodes: list['Node'], action: str, category: str):
        self.nodes = nodes
        self.action = action
        self.category = category


class Steps:
    def __init__(self):
        self.steps: list[Step] = []
        self.actions_map = {}

    def add_action(self, action: str):
        if action not in self.actions_map:
            self.actions_map[action] = len(self.actions_map.items()) + 1

    def add_step(self, step: Step):
        self.steps.append(step)
        self.add_action(step.action)
    
    def clear_steps(self):
        self.steps.clear()
        self.actions_map.clear()
