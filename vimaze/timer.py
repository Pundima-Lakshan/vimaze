import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vimaze.maze import Maze


class Timer:
    def __init__(self, maze: 'Maze'):
        self.start_time = None
        self.elapsed_time = 0

        self.maze = maze

        self.operation = None
        self.algorithm = None

    def start(self, operation: str, algorithm: str):
        """Start the timer."""
        if self.start_time is None:  # Prevent restarting without stopping
            self.operation = operation
            self.algorithm = algorithm

            timer_text = f"Started: {self.operation.capitalize()}    {self.algorithm.capitalize()}   {time.perf_counter() * 1000:.6f} milli seconds"
            self.maze.app.costs_str.set(timer_text)

            self.start_time = time.perf_counter()
        else:
            raise RuntimeError("Timer is already running. Stop it before restarting.")

    def stop(self):
        """Stop the timer and record elapsed time."""
        if self.start_time is not None:
            self.elapsed_time = time.perf_counter() - self.start_time
            self.start_time = None

            timer_text = f"Finished:    {self.operation.capitalize()}   {self.algorithm.capitalize()}    {self.elapsed_time * 1000:6f} milli seconds"
            self.maze.app.costs_str.set(timer_text)

            return self.elapsed_time
        else:
            raise RuntimeError("Timer is not running. Start it first.")

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.elapsed_time = 0

    def elapsed(self):
        """Return elapsed time without stopping the timer."""
        if self.start_time is not None:
            timer_text = f"Elapsed: {self.operation.capitalize()}    {self.algorithm.capitalize()}   {(time.perf_counter() - self.start_time) * 1000:6f} milli seconds"
            self.maze.app.costs_str = timer_text

            return time.perf_counter() - self.start_time
        return self.elapsed_time  # Return last recorded time
