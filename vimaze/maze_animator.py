from typing import TYPE_CHECKING

from vimaze.steps import Steps

if TYPE_CHECKING:
    from customtkinter import CTkCanvas
    from vimaze.steps import Step


class MazeAnimator:
    def __init__(self, canvas: 'CTkCanvas'):
        self.steps = Steps()
        self.canvas = canvas

    def add_step(self, step: 'Step'):
        self.steps.add_step(step)

    def animate(self):
        pass
