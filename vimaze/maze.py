from vimaze.maze_generator import MazeGenerator


class Maze:
    def __init__(self, master, maze_frame):
        self.master = master
        self.maze_frame = maze_frame
        self.maze = None
        self.controls = None
        self.speed = 5
        self.theme = "System"
        self.algorithm = "A*"
        self.generate_maze()

    def generate_maze(self):
        self.maze = MazeGenerator(30, 30)
        self.maze.generate_maze()
        self.maze.draw_maze(self.maze_frame)

    def set_speed(self, value):
        self.speed = value

    def set_theme(self, value):
        self.theme = value

    def set_algorithm(self, value):
        self.algorithm = value