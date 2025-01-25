from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from customtkinter import CTkCanvas


class MazeAnimator:
    def __init__(self, canvas: CTkCanvas):
        self.steps = None
        self.canvas = canvas
