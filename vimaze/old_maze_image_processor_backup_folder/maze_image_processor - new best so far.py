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
    and represents walls as edges between cells, using enhanced cell-based analysis.
    """
    
    def __init__(self, timer: Optional['Timer'] = None):
        """
        Initialize the MazeImageProcessor.
        
        Args:
            timer: Timer object for performance measurement
        """
        self.timer = timer
        self.debug_mode = False
        
        # Processing parameters
        self.wall_threshold = 127    # Threshold for wall detection (0-255)
        self.invert_binary = False   # Whether to invert the binary image
        
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

        # Step 5: Extract cells and analyze walls using the improved method
        cell_size = transition + wall
        cells = self._extract_cells(maze_binary, rows, cols, cell_size)
        wall_indicators = self._analyze_all_cells(cells, rows, cols)
        
        # Step 6: Create graph representation based on wall indicators
        graph = self._create_graph_from_wall_indicators(rows, cols, wall_indicators)

        # Visualize the walls and graph if in debug mode
        if self.debug_mode:
            self._visualize_walls(wall_indicators, rows, cols)
            self._visualize_maze_graph(graph, rows, cols)

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
    
    def _find_min_spacing(self, maze_binary):
        """
        Find the minimum spacing between walls by analyzing horizontal and vertical distances.
        
        Args:
            maze_binary: Binary maze image (0=wall, 1=path)
            
        Returns:
            Tuple of (path_width, wall_width)
        """
        height, width = maze_binary.shape
        
        # Find horizontal spacing (distances between vertical walls)
        horizontal_spaces = []
        horizontal_walls = []
        
        for row in range(height):
            # Analyze path spaces
            spaces = []
            current_space = 0
            # Analyze wall segments
            walls = []
            current_wall = 0
            
            for col in range(width):
                if maze_binary[row, col] == 1:  # Path pixel
                    # Count path length
                    current_space += 1
                    # End of wall segment check
                    if current_wall > 0:
                        walls.append(current_wall)
                        current_wall = 0
                else:  # Wall pixel
                    # Count wall length
                    current_wall += 1
                    # End of path segment check
                    if current_space > 0:
                        spaces.append(current_space)
                        current_space = 0
            
            # Add the last segments if they exist
            if current_space > 0:
                spaces.append(current_space)
            if current_wall > 0:
                walls.append(current_wall)
            
            # Add to overall lists
            horizontal_spaces.extend(spaces)
            horizontal_walls.extend(walls)
        
        # Find vertical spacing (distances between horizontal walls)
        vertical_spaces = []
        vertical_walls = []
        
        for col in range(width):
            # Analyze path spaces
            spaces = []
            current_space = 0
            # Analyze wall segments
            walls = []
            current_wall = 0
            
            for row in range(height):
                if maze_binary[row, col] == 1:  # Path pixel
                    # Count path length
                    current_space += 1
                    # End of wall segment check
                    if current_wall > 0:
                        walls.append(current_wall)
                        current_wall = 0
                else:  # Wall pixel
                    # Count wall length
                    current_wall += 1
                    # End of path segment check
                    if current_space > 0:
                        spaces.append(current_space)
                        current_space = 0
            
            # Add the last segments if they exist
            if current_space > 0:
                spaces.append(current_space)
            if current_wall > 0:
                walls.append(current_wall)
            
            # Add to overall lists
            vertical_spaces.extend(spaces)
            vertical_walls.extend(walls)
        
        # Filter out very large values (probably outer boundaries)
        max_valid_size = min(height, width) // 4
        horizontal_spaces = [s for s in horizontal_spaces if 0 < s < max_valid_size]
        vertical_spaces = [s for s in vertical_spaces if 0 < s < max_valid_size]
        horizontal_walls = [w for w in horizontal_walls if 0 < w < max_valid_size]
        vertical_walls = [w for w in vertical_walls if 0 < w < max_valid_size]
        
        # Calculate most common space width (path width)
        if horizontal_spaces and vertical_spaces:
            # Create histograms using Counter to find the most common values
            from collections import Counter
            h_counter = Counter(horizontal_spaces)
            v_counter = Counter(vertical_spaces)
            
            # Get the most common path width
            h_common = h_counter.most_common(3)
            v_common = v_counter.most_common(3)
            
            # Select the most reliable (highest count) from small values
            # Sort by count (frequency) and take smallest value with high frequency
            h_candidates = sorted(h_common, key=lambda x: (-x[1], x[0]))
            v_candidates = sorted(v_common, key=lambda x: (-x[1], x[0]))
            
            path_width_h = h_candidates[0][0] if h_candidates else 1
            path_width_v = v_candidates[0][0] if v_candidates else 1
            
            # Use the smaller of the two (more likely to be a single path unit)
            path_width = min(path_width_h, path_width_v)
        else:
            path_width = 1  # Default
        
        # Calculate most common wall width using the same approach
        if horizontal_walls and vertical_walls:
            from collections import Counter
            h_counter = Counter(horizontal_walls)
            v_counter = Counter(vertical_walls)
            
            h_common = h_counter.most_common(3)
            v_common = v_counter.most_common(3)
            
            # Favor thinner walls with high frequency
            h_candidates = sorted(h_common, key=lambda x: (-x[1], x[0]))
            v_candidates = sorted(v_common, key=lambda x: (-x[1], x[0]))
            
            wall_width_h = h_candidates[0][0] if h_candidates else 1
            wall_width_v = v_candidates[0][0] if v_candidates else 1
            
            # Use the smaller value (more likely to be an actual wall)
            wall_width = min(wall_width_h, wall_width_v)
        else:
            wall_width = 1  # Default
        
        # Ensure minimum sizes
        path_width = max(1, path_width)
        wall_width = max(1, wall_width)
        
        logging.debug(f"Detected path width: {path_width}, wall width: {wall_width}")
        logging.debug(f"Horizontal spaces: {h_common if 'h_common' in locals() else []}")
        logging.debug(f"Vertical spaces: {v_common if 'v_common' in locals() else []}")
        logging.debug(f"Horizontal walls: {h_common if 'h_common' in locals() else []}")
        logging.debug(f"Vertical walls: {v_common if 'v_common' in locals() else []}")
        
        return path_width, wall_width

    def _get_sub(self, maze):
        """
        Detect the wall and path widths using improved minimum spacing algorithm.
        
        Args:
            maze: Binary maze matrix (0=wall, 1=path)
            
        Returns:
            Tuple of (transition_width, wall_width)
        """
        # Use the improved spacing detection algorithm
        return self._find_min_spacing(maze)
    
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
    
    def _extract_cells(self, maze_img: np.ndarray, rows: int, cols: int, cell_size: int) -> List[List[np.ndarray]]:
        """
        Extract individual cells from the maze.
        
        Args:
            maze_img: Binary image of the maze
            rows: Number of rows
            cols: Number of columns
            cell_size: Size of each cell in pixels
            
        Returns:
            2D list of cell images
        """
        h, w = maze_img.shape
        
        cells = []
        for i in range(rows):
            row = []
            for j in range(cols):
                # Calculate cell boundaries
                y_start = i * cell_size
                y_end = min((i+1) * cell_size, h)
                x_start = j * cell_size
                x_end = min((j+1) * cell_size, w)
                
                # Extract cell
                if y_start < h and x_start < w:
                    cell = maze_img[y_start:y_end, x_start:x_end]
                    row.append(cell)
                else:
                    # Create empty cell if out of bounds
                    row.append(np.zeros((cell_size, cell_size), dtype=np.uint8))
            cells.append(row)
        
        return cells

    def _analyze_cell_walls(self, cell: np.ndarray, row: int, col: int, rows: int, cols: int) -> List[int]:
        """
        Analyze a cell to detect which walls are present.
        
        Args:
            cell: Cell image
            row: Row index
            col: Column index
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            List of wall indicators [top, right, bottom, left] (1=wall, 0=no wall)
        """
        h, w = cell.shape
        
        # Define narrow regions along each edge to check for walls
        wall_thickness = max(2, min(h, w) // 5)  # Adjust based on cell size
        threshold_percentage = 0.4  # Percentage of wall pixels needed to identify a wall
        
        # Check top wall
        top_region = cell[0:wall_thickness, :]
        top_wall = 1 if np.sum(top_region == 0) / top_region.size > threshold_percentage else 0
        
        # Check right wall
        right_region = cell[:, w-wall_thickness:w] if w > wall_thickness else cell
        right_wall = 1 if np.sum(right_region == 0) / right_region.size > threshold_percentage else 0
        
        # Check bottom wall
        bottom_region = cell[h-wall_thickness:h, :] if h > wall_thickness else cell
        bottom_wall = 1 if np.sum(bottom_region == 0) / bottom_region.size > threshold_percentage else 0
        
        # Check left wall
        left_region = cell[:, 0:wall_thickness]
        left_wall = 1 if np.sum(left_region == 0) / left_region.size > threshold_percentage else 0
        
        # Special handling for edge cells - outer boundary should always have walls
        if row == 0:  # Top row
            top_wall = 1
        if col == cols - 1:  # Rightmost column
            right_wall = 1
        if row == rows - 1:  # Bottom row
            bottom_wall = 1
        if col == 0:  # Leftmost column
            left_wall = 1
        
        return [top_wall, right_wall, bottom_wall, left_wall]
    
    def _analyze_all_cells(self, cells: List[List[np.ndarray]], rows: int, cols: int) -> List[List[List[int]]]:
        """
        Analyze all cells in the maze to detect walls.
        
        Args:
            cells: 2D list of cell images
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            3D list of wall indicators for each cell [row][col][wall_direction]
        """
        wall_indicators = []
        
        for i in range(rows):
            row_indicators = []
            for j in range(cols):
                cell_walls = self._analyze_cell_walls(cells[i][j], i, j, rows, cols)
                row_indicators.append(cell_walls)
            wall_indicators.append(row_indicators)
        
        # Ensure wall consistency between adjacent cells
        self._ensure_wall_consistency(wall_indicators, rows, cols)
        
        return wall_indicators
    
    def _ensure_wall_consistency(self, wall_indicators: List[List[List[int]]], rows: int, cols: int) -> None:
        """
        Ensure consistency of walls between adjacent cells (modifies wall_indicators in-place).
        
        Args:
            wall_indicators: 3D list of wall indicators
            rows: Number of rows
            cols: Number of columns
        """
        # Fix inconsistencies
        for i in range(rows):
            for j in range(cols):
                # Check right wall consistency
                if j < cols - 1:
                    # If cell has right wall, adjacent cell should have left wall
                    if wall_indicators[i][j][1] == 1:
                        wall_indicators[i][j+1][3] = 1
                    # If adjacent cell has left wall, cell should have right wall
                    elif wall_indicators[i][j+1][3] == 1:
                        wall_indicators[i][j][1] = 1
                
                # Check bottom wall consistency
                if i < rows - 1:
                    # If cell has bottom wall, cell below should have top wall
                    if wall_indicators[i][j][2] == 1:
                        wall_indicators[i+1][j][0] = 1
                    # If cell below has top wall, cell should have bottom wall
                    elif wall_indicators[i+1][j][0] == 1:
                        wall_indicators[i][j][2] = 1
    
    def _create_graph_from_wall_indicators(self, rows: int, cols: int, wall_indicators: List[List[List[int]]]) -> 'Graph':
        """
        Create a graph representation from detected wall indicators.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            wall_indicators: 3D list of wall indicators [row][col][wall_direction]
            
        Returns:
            Graph: Graph representation of the maze
        """
        from vimaze.ds.graph import Graph
        
        # Create a new graph
        graph = Graph()
        
        # Add nodes for all cells
        for i in range(rows):
            for c in range(cols):
                graph.add_node((i, c))
        
        # Add connections where there are no walls
        for i in range(rows):
            for j in range(cols):
                # Check all four directions
                # Top (no top wall)
                if i > 0 and wall_indicators[i][j][0] == 0:
                    graph.connect_nodes((i, j), (i-1, j))
                
                # Right (no right wall)
                if j < cols - 1 and wall_indicators[i][j][1] == 0:
                    graph.connect_nodes((i, j), (i, j+1))
                
                # Bottom (no bottom wall)
                if i < rows - 1 and wall_indicators[i][j][2] == 0:
                    graph.connect_nodes((i, j), (i+1, j))
                
                # Left (no left wall)
                if j > 0 and wall_indicators[i][j][3] == 0:
                    graph.connect_nodes((i, j), (i, j-1))
        
        return graph
        
    def _visualize_walls(self, wall_indicators: List[List[List[int]]], rows: int, cols: int) -> None:
        """
        Create a visualization of the detected walls.
        
        Args:
            wall_indicators: 3D list of wall indicators for each cell
            rows: Number of rows
            cols: Number of columns
        """
        # Create a debug grid image
        viz_cell_size = 50  # Size for visualization
        debug_grid = np.ones((rows * viz_cell_size, cols * viz_cell_size, 3), dtype=np.uint8) * 255
        
        for i in range(rows):
            for j in range(cols):
                cell_walls = wall_indicators[i][j]
                
                cell_y, cell_x = i * viz_cell_size, j * viz_cell_size
                # Draw the cell
                cv2.rectangle(debug_grid, (cell_x, cell_y), (cell_x + viz_cell_size, cell_y + viz_cell_size), (200, 200, 200), 1)
                
                # Draw detected walls
                if cell_walls[0]:  # Top
                    cv2.line(debug_grid, (cell_x, cell_y), (cell_x + viz_cell_size, cell_y), (0, 0, 0), 2)
                if cell_walls[1]:  # Right
                    cv2.line(debug_grid, (cell_x + viz_cell_size, cell_y), (cell_x + viz_cell_size, cell_y + viz_cell_size), (0, 0, 0), 2)
                if cell_walls[2]:  # Bottom
                    cv2.line(debug_grid, (cell_x, cell_y + viz_cell_size), (cell_x + viz_cell_size, cell_y + viz_cell_size), (0, 0, 0), 2)
                if cell_walls[3]:  # Left
                    cv2.line(debug_grid, (cell_x, cell_y), (cell_x, cell_y + viz_cell_size), (0, 0, 0), 2)
                
                # Add labels
                cv2.putText(debug_grid, f"{i},{j}", (cell_x + 15, cell_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        cv2.imwrite("debug/05_wall_detection.png", debug_grid)

    def _visualize_maze_graph(self, graph: 'Graph', rows: int, cols: int) -> None:
        """
        Create visualizations for debugging.
        
        Args:
            graph: Graph representation of the maze
            rows: Number of rows
            cols: Number of columns
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
                node_key = f"{r},{c}"
                if node_key in graph.nodes:
                    node = graph.nodes[node_key]
                    
                    # Draw connections
                    for neighbor in node.neighbors:
                        n_r, n_c = neighbor.position
                        n_cy = n_r * cell_size + 30
                        n_cx = n_c * cell_size + 30
                        
                        cv2.line(graph_viz, (cx, cy), (n_cx, n_cy), (0, 255, 0), 2)
        
        # Save graph visualization
        cv2.imwrite("debug/06_graph_representation.png", graph_viz)
        
        # Create a visualization showing the maze with paths
        maze_viz = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 255
        
        # Iterate through all cells
        for r in range(rows):
            for c in range(cols):
                cell_x = c * cell_size + 15
                cell_y = r * cell_size + 15
                
                # Draw cell outline
                cv2.rectangle(maze_viz, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), 
                             (220, 220, 220), 1)
                
                # Check neighboring cells to determine walls
                node_key = f"{r},{c}"
                if node_key in graph.nodes:
                    neighbors = graph.nodes[node_key].neighbors
                    neighbor_positions = [neighbor.position for neighbor in neighbors]
                else:
                    neighbor_positions = []
                
                # Draw walls where there are no connections
                # Top wall
                if (r-1, c) not in neighbor_positions:
                    cv2.line(maze_viz, (cell_x, cell_y), (cell_x + cell_size, cell_y), (0, 0, 0), 2)
                    
                # Right wall
                if (r, c+1) not in neighbor_positions:
                    cv2.line(maze_viz, (cell_x + cell_size, cell_y), 
                            (cell_x + cell_size, cell_y + cell_size), (0, 0, 0), 2)
                    
                # Bottom wall
                if (r+1, c) not in neighbor_positions:
                    cv2.line(maze_viz, (cell_x, cell_y + cell_size), 
                            (cell_x + cell_size, cell_y + cell_size), (0, 0, 0), 2)
                    
                # Left wall
                if (r, c-1) not in neighbor_positions:
                    cv2.line(maze_viz, (cell_x, cell_y), (cell_x, cell_y + cell_size), (0, 0, 0), 2)
                
                # Label the cell
                cv2.putText(maze_viz, f"{r},{c}", (cell_x + 5, cell_y + cell_size - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Save maze visualization
        cv2.imwrite("debug/07_maze_visualization.png", maze_viz)
    
    def test_maze_image_processing(self, image_path, debug=True, invert_binary=False, wall_threshold=127):
        """
        Test the maze image processing with debug mode.
        
        Args:
            image_path: Path to the maze image file
            debug: Whether to enable debug mode
            invert_binary: Whether to invert the binary image
            wall_threshold: Threshold for wall detection
            
        Returns:
            Tuple of (success, message)
        """
        logging.info(f"Testing maze image processing: {image_path}")
        logging.info(f"Parameters: debug={debug}, invert_binary={invert_binary}, wall_threshold={wall_threshold}")
        
        try:
            self.debug_mode = debug
            self.invert_binary = invert_binary
            self.wall_threshold = wall_threshold
            
            graph, rows, cols = self.process_image(image_path)
            
            logging.info(f"Maze loaded successfully. Size: {rows}x{cols}")
            return True, f"Maze loaded successfully. Size: {rows}x{cols}"
        except Exception as e:
            logging.error(f"Failed to process maze image: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return False, f"Error: {str(e)}"