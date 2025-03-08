import cv2
import numpy as np
import os
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from vimaze.ds.graph import Graph
    from vimaze.timer import Timer

class SimpleMazeProcessor:
    """
    A simpler processor specifically for clean black-walled mazes on white backgrounds.
    This approach uses a more direct method to extract the maze structure.
    """
    
    def __init__(self, timer: Optional['Timer'] = None):
        self.timer = timer
        self.debug_mode = False
        
        # Processing parameters
        self.cell_size = 20       # Target cell size for the grid
        self.wall_threshold = 50  # Threshold for detecting walls (0-255)
        
    def process_image(self, image_path: str) -> Tuple['Graph', int, int]:
        """
        Process a maze image and return a Graph representation along with dimensions.
        
        Args:
            image_path: Path to the maze image file
            
        Returns:
            A tuple containing:
            - Graph: The Graph representation of the maze
            - int: Number of rows in the maze
            - int: Number of columns in the maze
        """
        
        if self.timer:
            self.timer.start('processing', 'image')
        
        # Create debug directory if needed
        if self.debug_mode and not os.path.exists("simple_debug"):
            os.makedirs("simple_debug")
        
        # Step 1: Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        if self.debug_mode:
            cv2.imwrite("simple_debug/01_original.png", image)
        
        # Step 2: Convert to grayscale and threshold to get binary image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        if self.debug_mode:
            cv2.imwrite("simple_debug/02_binary.png", binary)
        
        # Step 3: Extract maze area by finding the largest contour
        maze_binary, maze_bounds = self._extract_maze_area(binary)
        
        if self.debug_mode and maze_binary is not None:
            cv2.imwrite("simple_debug/03_maze_area.png", maze_binary)
        
        # Step 4: Create a grid over the maze
        rows, cols, grid = self._create_grid(maze_binary)
        
        if self.debug_mode:
            grid_viz = cv2.cvtColor(maze_binary.copy(), cv2.COLOR_GRAY2BGR)
            # Draw grid
            for r in range(rows+1):
                y = r * self.cell_size
                cv2.line(grid_viz, (0, y), (grid_viz.shape[1], y), (0, 255, 0), 1)
            for c in range(cols+1):
                x = c * self.cell_size
                cv2.line(grid_viz, (x, 0), (x, grid_viz.shape[0]), (0, 0, 255), 1)
            cv2.imwrite("simple_debug/04_grid.png", grid_viz)
        
        # Step 5: Identify walls in the grid
        wall_map = self._identify_walls(maze_binary, rows, cols, grid)
        
        # Step 6: Create graph representation
        graph = self._create_graph(rows, cols, wall_map)
        
        # Step 7: Visualize the final maze
        if self.debug_mode:
            self._visualize_maze(maze_binary, rows, cols, wall_map)
        
        if self.timer:
            self.timer.stop()
        
        return graph, rows, cols
    
    def _extract_maze_area(self, binary: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Extract the maze area from the binary image.
        
        Args:
            binary: Binary image
            
        Returns:
            Tuple of (extracted maze binary image, bounding box)
        """
        # Invert binary for contour detection
        inverted = cv2.bitwise_not(binary)
        
        # Find contours
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found, using entire image")
            return binary, None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract the maze area with margin
        margin = 5
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(binary.shape[1], x + w + margin)
        y_end = min(binary.shape[0], y + h + margin)
        
        # Extract the maze area
        maze_binary = binary[y_start:y_end, x_start:x_end].copy()
        
        if self.debug_mode:
            contour_viz = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_viz, [largest_contour], 0, (0, 255, 0), 2)
            cv2.rectangle(contour_viz, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imwrite("simple_debug/03a_contour.png", contour_viz)
        
        return maze_binary, (x_start, y_start, x_end, y_end)
    
    def _create_grid(self, maze_binary: np.ndarray) -> Tuple[int, int, np.ndarray]:
        """
        Create a grid over the maze area.
        
        Args:
            maze_binary: Binary image of the maze area
            
        Returns:
            Tuple of (rows, columns, grid)
        """
        # Get maze dimensions
        height, width = maze_binary.shape[:2]
        
        # Calculate number of rows and columns
        rows = height // self.cell_size
        cols = width // self.cell_size
        
        # Make sure we have at least one row and column
        rows = max(1, rows)
        cols = max(1, cols)
        
        # Create a grid representation
        grid = np.zeros((rows, cols, 4), dtype=bool)  # 4 walls per cell: N, E, S, W
        
        return rows, cols, grid
    
    def _identify_walls(self, maze_binary: np.ndarray, rows: int, cols: int, grid: np.ndarray) -> set:
        """
        Identify walls in the grid.
        
        Args:
            maze_binary: Binary image of the maze area
            rows: Number of rows
            cols: Number of columns
            grid: Grid representation
            
        Returns:
            Set of wall positions
        """
        wall_map = set()
        
        if self.debug_mode:
            wall_viz = cv2.cvtColor(maze_binary.copy(), cv2.COLOR_GRAY2BGR)
        
        for row in range(rows):
            for col in range(cols):
                # Cell coordinates
                y1 = row * self.cell_size
                x1 = col * self.cell_size
                y2 = (row + 1) * self.cell_size
                x2 = (col + 1) * self.cell_size
                
                # Cell center
                cy = (y1 + y2) // 2
                cx = (x1 + x2) // 2
                
                # Check if cell center is a wall (black pixel in binary image)
                center_value = maze_binary[cy, cx]
                is_wall = center_value < self.wall_threshold
                
                if self.debug_mode:
                    center_color = (0, 0, 255) if is_wall else (0, 255, 0)
                    cv2.circle(wall_viz, (cx, cy), 2, center_color, -1)
                
                # If this is a wall cell, add walls to all neighboring cells
                if is_wall:
                    neighbors = [
                        (row-1, col),  # North
                        (row, col+1),  # East
                        (row+1, col),  # South
                        (row, col-1)   # West
                    ]
                    
                    for i, (nr, nc) in enumerate(neighbors):
                        if 0 <= nr < rows and 0 <= nc < cols:
                            wall_map.add(((row, col), (nr, nc)))
                            wall_map.add(((nr, nc), (row, col)))
                            
                            if self.debug_mode:
                                # Get neighbor center
                                ny1 = nr * self.cell_size
                                nx1 = nc * self.cell_size
                                ny2 = (nr + 1) * self.cell_size
                                nx2 = (nc + 1) * self.cell_size
                                ncy = (ny1 + ny2) // 2
                                ncx = (nx1 + nx2) // 2
                                
                                # Draw wall line
                                cv2.line(wall_viz, (cx, cy), (ncx, ncy), (0, 0, 255), 1)
        
        if self.debug_mode:
            cv2.imwrite("simple_debug/05_wall_detection.png", wall_viz)
        
        return wall_map
    
    def _create_graph(self, rows: int, cols: int, wall_map: set) -> 'Graph':
        """
        Create a graph representation of the maze.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            wall_map: Set of wall positions
            
        Returns:
            Graph: The Graph representation of the maze
        """
        from vimaze.ds.graph import Graph
        
        graph = Graph()
        
        # Add nodes for all cells
        for row in range(rows):
            for col in range(cols):
                graph.add_node((row, col))
        
        # Connect nodes where there are no walls
        for row in range(rows):
            for col in range(cols):
                # Check each of the four possible neighbors
                neighbors = [
                    (row-1, col),  # North
                    (row, col+1),  # East
                    (row+1, col),  # South
                    (row, col-1)   # West
                ]
                
                for nr, nc in neighbors:
                    # Check if neighbor is within bounds and there's no wall
                    if 0 <= nr < rows and 0 <= nc < cols:
                        wall_exists = ((row, col), (nr, nc)) in wall_map
                        if not wall_exists:
                            graph.connect_nodes((row, col), (nr, nc))
        
        return graph
    
    def _visualize_maze(self, maze_binary: np.ndarray, rows: int, cols: int, wall_map: set):
        """
        Create a visualization of the detected maze.
        
        Args:
            maze_binary: Binary image of the maze
            rows: Number of rows
            cols: Number of columns
            wall_map: Set of wall positions
        """
        # Create a clean visualization
        viz_size = max(300, rows * self.cell_size, cols * self.cell_size)
        viz = np.ones((viz_size, viz_size, 3), dtype=np.uint8) * 255
        
        # Scale factor to fit maze in viz
        scale = min(viz_size / (rows * self.cell_size), viz_size / (cols * self.cell_size)) * 0.9
        
        # Offset to center maze
        offset_x = (viz_size - cols * self.cell_size * scale) // 2
        offset_y = (viz_size - rows * self.cell_size * scale) // 2
        
        # Draw grid
        for r in range(rows+1):
            y = int(r * self.cell_size * scale + offset_y)
            cv2.line(viz, (int(offset_x), y), (int(cols * self.cell_size * scale + offset_x), y), (200, 200, 200), 1)
        
        for c in range(cols+1):
            x = int(c * self.cell_size * scale + offset_x)
            cv2.line(viz, (x, int(offset_y)), (x, int(rows * self.cell_size * scale + offset_y)), (200, 200, 200), 1)
        
        # Draw walls
        for (r1, c1), (r2, c2) in wall_map:
            # Only draw each wall once
            if r1 > r2 or (r1 == r2 and c1 > c2):
                continue
                
            # Calculate cell centers
            y1 = int((r1 + 0.5) * self.cell_size * scale + offset_y)
            x1 = int((c1 + 0.5) * self.cell_size * scale + offset_x)
            y2 = int((r2 + 0.5) * self.cell_size * scale + offset_y)
            x2 = int((c2 + 0.5) * self.cell_size * scale + offset_x)
            
            # Draw wall
            cv2.line(viz, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        # Save visualization
        cv2.imwrite("simple_debug/06_maze_visualization.png", viz)
        
        # Create a simple maze visualization (similar to the original)
        simple_viz = np.ones((rows * 20, cols * 20, 3), dtype=np.uint8) * 255
        
        # Draw walls
        for row in range(rows):
            for col in range(cols):
                cell_y = row * 20 + 10
                cell_x = col * 20 + 10
                
                # Center circle
                if ((row, col), (row, col+1)) in wall_map and ((row, col), (row+1, col)) in wall_map:
                    cv2.circle(simple_viz, (cell_x, cell_y), 2, (0, 0, 0), -1)
                
                # Check neighbors
                neighbors = [
                    (row-1, col, 0, -10),  # North
                    (row, col+1, 10, 0),   # East
                    (row+1, col, 0, 10),   # South
                    (row, col-1, -10, 0)   # West
                ]
                
                for nr, nc, dx, dy in neighbors:
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if ((row, col), (nr, nc)) in wall_map:
                            cv2.line(simple_viz, (cell_x, cell_y), (cell_x + dx, cell_y + dy), (0, 0, 0), 2)
        
        # Add border
        cv2.rectangle(simple_viz, (0, 0), (cols * 20 - 1, rows * 20 - 1), (0, 0, 0), 2)
        
        # Save simple visualization
        cv2.imwrite("simple_debug/07_simple_visualization.png", simple_viz)

        