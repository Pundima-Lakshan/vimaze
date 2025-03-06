import cv2
import numpy as np
import os
from typing import TYPE_CHECKING, Tuple, List, Set, Dict, Optional
import logging
from collections import Counter
from PIL import Image

if TYPE_CHECKING:
    from vimaze.ds.graph import Graph
    from vimaze.timer import Timer

class MazeImageProcessor:
    """
    Improved maze image processor that accurately detects the cell grid
    and represents walls as edges between cells, using PIL-based approach.
    """
    
    def __init__(self, timer: Optional['Timer'] = None):
        """
        Initialize the MazeImageProcessor.
        
        Args:
            timer: Timer object for performance measurement
        """
        self.timer = timer
        self.debug_mode = False
        
    def process_image(self, image_path: str) -> Tuple['Graph', int, int]:
        """
        Process a maze image and return a Graph representation along with dimensions.
        """
        from vimaze.ds.graph import Graph

        if self.timer:
            self.timer.start('processing', 'image')

        # Create debug directory if needed
        if self.debug_mode and not os.path.exists("debug"):
            os.makedirs("debug")

        # Step 1: Load and preprocess the image using PIL
        logging.debug(f"Loading image: {image_path}")
        image = Image.open(image_path).convert("L")
        binary_image = image.point(lambda pixel: 0 if pixel < 128 else 1, mode="1")
        width, height = binary_image.size
        pixel_matrix = binary_image.load()
        
        if self.debug_mode:
            # Convert PIL image to OpenCV format for debug saving
            binary_np = np.array(binary_image, dtype=np.uint8) * 255
            cv2.imwrite("debug/01_original.png", binary_np)

        # Step 2: Find start and end points
        start = self._find_point_start(pixel_matrix, width, height)
        end = self._find_point_end(pixel_matrix, width, height)
        
        if self.debug_mode:
            binary_debug = np.array(binary_image, dtype=np.uint8) * 255
            debug_img = cv2.cvtColor(binary_debug, cv2.COLOR_GRAY2BGR)
            if start != (-1, -1):
                cv2.circle(debug_img, (start[0], start[1]), 5, (0, 0, 255), -1)
            if end != (-1, -1):
                cv2.circle(debug_img, (end[0], end[1]), 5, (0, 255, 0), -1)
            cv2.imwrite("debug/02_points.png", debug_img)

        # Step 3: Extract maze area
        maze_binary = self._get_maze(pixel_matrix, binary_image, start, end)
        
        if self.debug_mode:
            maze_debug = np.array(maze_binary, dtype=np.uint8) * 255
            cv2.imwrite("debug/03_maze_area.png", maze_debug)

        # Step 4: Detect grid parameters (cell size and alignment)
        transition, wall = self._get_sub(maze_binary)
        rows, cols = self._calculate_grid_dimensions(maze_binary, transition, wall)
        
        logging.debug(f"Detected grid: path_width={transition}, wall_width={wall}, size={rows}x{cols}")

        if self.debug_mode:
            cell_size = transition + wall
            maze_debug = np.array(maze_binary, dtype=np.uint8) * 255
            grid_viz = cv2.cvtColor(maze_debug, cv2.COLOR_GRAY2BGR)
            
            # Draw detected grid
            for r in range(rows + 1):
                y = r * cell_size
                if y < grid_viz.shape[0]:
                    cv2.line(grid_viz, (0, y), (grid_viz.shape[1], y), (0, 255, 0), 1)
            
            for c in range(cols + 1):
                x = c * cell_size
                if x < grid_viz.shape[1]:
                    cv2.line(grid_viz, (x, 0), (x, grid_viz.shape[0]), (0, 255, 0), 1)
            
            cv2.imwrite("debug/04_grid_detection.png", grid_viz)

        # Step 5: Detect walls between cells
        wall_edges = self._detect_wall_edges(maze_binary, rows, cols, transition, wall)

        if self.debug_mode:
            maze_debug = np.array(maze_binary, dtype=np.uint8) * 255
            wall_viz = cv2.cvtColor(maze_debug, cv2.COLOR_GRAY2BGR)
            
            # Draw cell centers and walls
            cell_size = transition + wall
            for r in range(rows):
                for c in range(cols):
                    # Cell center
                    cy = r * cell_size + cell_size // 2
                    cx = c * cell_size + cell_size // 2
                    
                    if cy < wall_viz.shape[0] and cx < wall_viz.shape[1]:
                        # Draw cell center
                        cv2.circle(wall_viz, (cx, cy), 2, (0, 0, 255), -1)
                        
                        # Draw connections or walls
                        directions = [
                            ((r-1, c), 0, -cell_size // 2),   # North
                            ((r, c+1), cell_size // 2, 0),    # East
                            ((r+1, c), 0, cell_size // 2),    # South
                            ((r, c-1), -cell_size // 2, 0)    # West
                        ]
                        
                        for (nr, nc), dx, dy in directions:
                            if 0 <= nr < rows and 0 <= nc < cols:
                                # Wall or path?
                                if ((r, c), (nr, nc)) in wall_edges:
                                    # Wall - red line
                                    nx = cx + dx
                                    ny = cy + dy
                                    if ny < wall_viz.shape[0] and nx < wall_viz.shape[1]:
                                        cv2.line(wall_viz, (cx, cy), (nx, ny), (0, 0, 255), 1)
                                else:
                                    # Path - green line
                                    nx = cx + dx
                                    ny = cy + dy
                                    if ny < wall_viz.shape[0] and nx < wall_viz.shape[1]:
                                        cv2.line(wall_viz, (cx, cy), (nx, ny), (0, 255, 0), 1)
            
            cv2.imwrite("debug/05_wall_detection.png", wall_viz)

        # Step 6: Create graph representation
        graph = self._create_graph_with_walls(rows, cols, wall_edges)

        if self.debug_mode:
            self._visualize_maze_graph(graph, rows, cols, wall_edges)

        if self.timer:
            self.timer.stop()

        logging.debug(f"Maze processed successfully: {rows}x{cols} cells")

        return graph, rows, cols
    
    def _find_point_start(self, pixel_matrix, width, height):
        """
        Find the starting point of the maze. This function looks for walls that 
        extend a significant distance (>50% of width/height) to identify possible entrances.
        
        Args:
            pixel_matrix: Binary image pixel matrix
            width: Image width
            height: Image height
            
        Returns:
            Tuple of (x, y) coordinates of the start point, or (-1, -1) if not found
        """
        start = (-1, -1)
        for i in range(width):
            for j in range(height):
                if pixel_matrix[i, j] == 0:  # Wall pixel found
                    k = j
                    while (k < height and pixel_matrix[i, k] == 0):
                        k += 1
                    l = i
                    while (l < height and pixel_matrix[l, j] == 0):
                        l += 1
                    if k - j > height * 0.5 or l - i > width * 0.5:
                        start = (i, j)
                        break
            if start != (-1, -1):
                break
        return start
    
    def _find_point_end(self, pixel_matrix, width, height):
        """
        Find the ending point of the maze, searching from the bottom-right.
        
        Args:
            pixel_matrix: Binary image pixel matrix
            width: Image width
            height: Image height
            
        Returns:
            Tuple of (x, y) coordinates of the end point, or (-1, -1) if not found
        """
        finish = (-1, -1)
        for j in range(height - 1, -1, -1):
            for i in range(width - 1, -1, -1):
                if pixel_matrix[i, j] == 0:  # Wall pixel found
                    k = j
                    while k >= 0 and pixel_matrix[i, k] == 0:
                        k -= 1
                    l = i
                    while l >= 0 and pixel_matrix[l, j] == 0:
                        l -= 1
                    if (j - k) > height * 0.5 or (i - l) > width * 0.5:
                        finish = (i, j)
                        return finish
        return finish
    
    def _get_maze(self, pixel_matrix, binary_image, start, end):
        """
        Extract the maze area from the binary image based on start and end points.
        
        Args:
            pixel_matrix: Binary image pixel matrix
            binary_image: Binary image
            start: Start point (x, y)
            end: End point (x, y)
            
        Returns:
            NumPy array representing the maze area
        """
        width, height = binary_image.size
        
        # Use full image if start/end not found
        if start == (-1, -1) or end == (-1, -1):
            indexNorth, indexWest = 0, 0
            indexSout, indexEast = width - 1, height - 1
        else:
            indexNorth, indexWest = start
            indexSout, indexEast = end
        
        # Calculate dimensions and create maze matrix
        w, h = indexSout - indexNorth + 1, indexEast - indexWest + 1
        maze = np.zeros((w, h), dtype=np.uint8)
        
        # Fill maze matrix with pixel values
        for i in range(indexNorth, indexSout + 1):
            for j in range(indexWest, indexEast + 1):
                if 0 <= i < width and 0 <= j < height:
                    maze[i - indexNorth, j - indexWest] = pixel_matrix[i, j]
        
        return maze
    
    def _get_sub(self, maze):
        """
        Detect the wall and path widths by analyzing patterns in the maze.
        
        Args:
            maze: Binary maze matrix (0=wall, 1=path)
            
        Returns:
            Tuple of (transition_width, wall_width)
        """
        height, width = maze.shape
        wall = 0
        common_wall = [0] * max(height, width)  # Size adjusted to avoid index errors
        
        # Find wall width by looking at continuous wall segments
        for x in range(height - 1):
            for y in range(width - 1):
                if maze[x, y] == 0:  # Found a wall pixel
                    i = x
                    j = y
                    wall_i = 0
                    wall_j = 0
                    
                    # Count horizontal wall width
                    while i < height - 1 and maze[i, y] == 0:
                        wall_i += 1
                        i += 1
                    
                    # Count vertical wall width
                    while j < width - 1 and maze[x, j] == 0:
                        wall_j += 1
                        j += 1
                    
                    # Take the smallest dimension as the wall width
                    if wall_i > 0 and wall_j > 0:
                        this_wall = min(wall_i, wall_j)
                        if this_wall < len(common_wall):
                            common_wall[this_wall] += 1
                        wall = this_wall
                        break
        
        # Find most common wall width
        max_count = 0
        wall_width = 1  # Default
        for i, count in enumerate(common_wall):
            if i > 0 and count > max_count:
                max_count = count
                wall_width = i
        
        # Find transition (path) width
        transition = 0
        common_t = [0] * max(height, width)  # Size adjusted to avoid index errors
        
        # Similar approach for paths (value 1)
        for x in range(height - 1):
            for y in range(width - 1):
                if maze[x, y] == 1:  # Found a path pixel
                    i = x
                    j = y
                    transition_i = 0
                    transition_j = 0
                    
                    # Count horizontal path width
                    while i < height - 1 and maze[i, y] == 1:
                        transition_i += 1
                        i += 1
                    
                    # Count vertical path width
                    while j < width - 1 and maze[x, j] == 1:
                        transition_j += 1
                        j += 1
                    
                    # Take the smallest dimension as the path width
                    if transition_i > 0 and transition_j > 0:
                        this_transition = min(transition_i, transition_j)
                        if this_transition < len(common_t):
                            common_t[this_transition] += 1
                        transition = this_transition
                        break
        
        # Find most common path width
        max_count = 0
        path_width = 1  # Default
        min_index = 1 if wall_width == 1 else 0  # Avoid confusion if wall width is 1
        
        for i, count in enumerate(common_t):
            if i > min_index and count > max_count:
                max_count = count
                path_width = i
        
        # Ensure we have valid values (defaults if detection fails)
        wall_width = max(1, wall_width)
        path_width = max(1, path_width)
        
        return path_width, wall_width
    
    def _calculate_grid_dimensions(self, maze_matrix, transition, wall):
        """
        Calculate grid dimensions based on cell size.
        
        Args:
            maze_matrix: Binary maze matrix
            transition: Width of path
            wall: Width of wall
            
        Returns:
            Tuple of (rows, cols)
        """
        height, width = maze_matrix.shape
        cell_size = transition + wall
        
        # Calculate number of rows and columns
        rows = height // cell_size
        cols = width // cell_size
        
        # Ensure we have at least one row and column
        rows = max(1, rows)
        cols = max(1, cols)
        
        return rows, cols
    
    def _detect_wall_edges(self, maze_matrix, rows, cols, transition, wall):
        """
        Detect walls between cells by checking paths between cell centers.
        
        Args:
            maze_matrix: Binary maze matrix
            rows: Number of rows
            cols: Number of columns
            transition: Width of path
            wall: Width of wall
            
        Returns:
            Set of wall edges as ((row1, col1), (row2, col2))
        """
        wall_edges = set()
        cell_size = transition + wall
        
        # For each cell
        for r in range(rows):
            for c in range(cols):
                # Calculate cell center
                cy = r * cell_size + cell_size // 2
                cx = c * cell_size + cell_size // 2
                
                # Ensure cell center is within image bounds
                height, width = maze_matrix.shape
                if not (0 <= cy < height and 0 <= cx < width):
                    continue
                
                # Check all four adjacent cells
                directions = [
                    ((r-1, c), 0, -cell_size),   # North
                    ((r, c+1), cell_size, 0),    # East
                    ((r+1, c), 0, cell_size),    # South
                    ((r, c-1), -cell_size, 0)    # West
                ]
                
                for (nr, nc), dx, dy in directions:
                    # Skip if neighbor is out of bounds
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    
                    # Calculate neighbor cell center
                    ncy = nr * cell_size + cell_size // 2
                    ncx = nc * cell_size + cell_size // 2
                    
                    # Ensure neighbor cell center is within image bounds
                    if not (0 <= ncy < height and 0 <= ncx < width):
                        continue
                    
                    # Check if there's a wall between the cells
                    is_wall = self._check_for_wall(maze_matrix, (cy, cx), (ncy, ncx))
                    
                    # If there's a wall, add the edge to the set
                    if is_wall:
                        wall_edges.add(((r, c), (nr, nc)))
        
        return wall_edges
    
    def _check_for_wall(self, maze_matrix, pt1, pt2):
        """
        Check if there's a wall between two points by sampling pixels along the line.
        
        Args:
            maze_matrix: Binary maze matrix
            pt1: First point (y, x)
            pt2: Second point (y, x)
            
        Returns:
            True if there's a wall, False otherwise
        """
        # Get coordinates
        y1, x1 = pt1
        y2, x2 = pt2
        
        # Make sure points are within image bounds
        height, width = maze_matrix.shape
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # Sample points along the line
        num_samples = 7  # Sample more points for better accuracy
        wall_count = 0
        
        for i in range(num_samples):
            # Calculate sample point (t goes from 0 to 1)
            t = (i + 1) / (num_samples + 1)
            y = int((1-t) * y1 + t * y2)
            x = int((1-t) * x1 + t * x2)
            
            # Make sure sample point is within image bounds
            y = max(0, min(y, height-1))
            x = max(0, min(x, width-1))
            
            # Check if sample point is a wall (0 in the maze_matrix)
            if maze_matrix[y, x] == 0:
                wall_count += 1
        
        # If a significant number of samples are walls, consider it a wall
        return wall_count >= num_samples // 3
    
    def _create_graph_with_walls(self, rows, cols, wall_edges):
        """
        Create a graph with nodes for all cells and connections where there are no walls.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            wall_edges: Set of wall edges
            
        Returns:
            Graph: Graph representation of the maze
        """
        from vimaze.ds.graph import Graph
        
        # Create a new graph
        graph = Graph()
        
        # Add nodes for all cells
        for r in range(rows):
            for c in range(cols):
                graph.add_node((r, c))
        
        # Add connections where there are no walls
        for r in range(rows):
            for c in range(cols):
                # Check all four adjacent cells
                adjacent_cells = [
                    (r-1, c),  # North
                    (r, c+1),  # East
                    (r+1, c),  # South
                    (r, c-1)   # West
                ]
                
                for adj_r, adj_c in adjacent_cells:
                    # Skip if adjacent cell is out of bounds
                    if not (0 <= adj_r < rows and 0 <= adj_c < cols):
                        continue
                    
                    # Connect cells if there's no wall between them
                    if ((r, c), (adj_r, adj_c)) not in wall_edges and ((adj_r, adj_c), (r, c)) not in wall_edges:
                        graph.connect_nodes((r, c), (adj_r, adj_c))
        
        return graph
    
    def _visualize_maze_graph(self, graph, rows, cols, wall_edges):
        """
        Create visualizations for debugging.
        
        Args:
            graph: Graph representation of the maze
            rows: Number of rows
            cols: Number of columns
            wall_edges: Set of wall edges
        """
        # Create visualization of graph structure
        cell_size = 30  # Size of cells in visualization
        viz_width = cols * cell_size + 30
        viz_height = rows * cell_size + 30
        graph_viz = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 255
        
        # Draw grid
        for r in range(rows + 1):
            y = r * cell_size + 15
            cv2.line(graph_viz, (15, y), (viz_width - 15, y), (200, 200, 200), 1)
        
        for c in range(cols + 1):
            x = c * cell_size + 15
            cv2.line(graph_viz, (x, 15), (x, viz_height - 15), (200, 200, 200), 1)
        
        # Draw nodes and connections
        for r in range(rows):
            for c in range(cols):
                # Node center
                cy = r * cell_size + 30
                cx = c * cell_size + 30
                
                # Draw node
                cv2.circle(graph_viz, (cx, cy), 4, (0, 0, 255), -1)
                
                # Get the node
                node_name = f"{r},{c}"
                if node_name in graph.nodes:
                    node = graph.nodes[node_name]
                    
                    # Draw connections
                    for neighbor in node.neighbors:
                        n_r, n_c = neighbor.position
                        n_cy = n_r * cell_size + 30
                        n_cx = n_c * cell_size + 30
                        
                        cv2.line(graph_viz, (cx, cy), (n_cx, n_cy), (0, 255, 0), 2)
        
        # Save graph visualization
        cv2.imwrite("debug/06_graph_representation.png", graph_viz)
        
        # Create maze visualization with just the walls
        maze_viz = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 255
        
        # Draw grid cells
        for r in range(rows):
            for c in range(cols):
                y = r * cell_size + 15
                x = c * cell_size + 15
                cv2.rectangle(maze_viz, (x, y), (x + cell_size, y + cell_size), (220, 220, 220), 1)
        
        # Draw cell centers
        for r in range(rows):
            for c in range(cols):
                cy = r * cell_size + 30
                cx = c * cell_size + 30
                cv2.circle(maze_viz, (cx, cy), 2, (0, 0, 255), -1)
        
        # Draw wall edges
        for (r1, c1), (r2, c2) in wall_edges:
            # Calculate cell centers
            y1 = r1 * cell_size + 30
            x1 = c1 * cell_size + 30
            y2 = r2 * cell_size + 30
            x2 = c2 * cell_size + 30
            
            # Draw wall
            cv2.line(maze_viz, (x1, y1), (x2, y2), (0, 0, 0), 3)
        
        # Save maze visualization
        cv2.imwrite("debug/07_maze_visualization.png", maze_viz)
    
    def test_maze_image_processing(self, image_path, debug=True):
        """
        Test the maze image processing with debug mode.
        
        Args:
            image_path: Path to the maze image file
            debug: Whether to enable debug mode
            
        Returns:
            Tuple of (success, message)
        """
        logging.info(f"Testing maze image processing: {image_path}")
        logging.info(f"Parameters: debug={debug}")
        
        try:
            self.debug_mode = debug
            
            graph, rows, cols = self.process_image(image_path)
            
            logging.info(f"Maze loaded successfully. Size: {rows}x{cols}")
            return True, f"Maze loaded successfully. Size: {rows}x{cols}"
        except Exception as e:
            logging.error(f"Failed to process maze image: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return False, f"Error: {str(e)}"