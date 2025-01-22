class MazeGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = [[0 for _ in range(width)] for _ in range(height)]
        self.visited = [[False for _ in range(width)] for _ in range(height)]
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def generate(self):
        self._generate(0, 0)

    def _generate(self, x, y):
        self.visited[y][x] = True
        random.shuffle(self.directions)
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                continue
            if self.visited[ny][nx]:
                continue
            self.maze[y][x] |= 1 << self.directions.index((dx, dy))
            self.maze[ny][nx] |= 1 << self.directions.index((-dx, -dy))
            self._generate(nx, ny)

    def print(self):
        for row in self.maze:
            print(row)
