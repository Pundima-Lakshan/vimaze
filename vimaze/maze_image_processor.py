import cv2
import numpy as np
from typing import TYPE_CHECKING, Tuple, List, Set, Dict, Optional
import logging

if TYPE_CHECKING:
    from vimaze.graph import Graph
    from vimaze.timer import Timer


class MazeImageProcessor:
    def __init__(self, timer: Optional['Timer'] = None):
        """
        Initialize the MazeImageProcessor.
        
        Args:
            timer: Timer object for performance measurement
        """
        self.timer = timer
        self.debug_mode = False
        
        # Processing parameters - these can be adjusted for different maze images
        self.wall_threshold = 127  # Threshold for wall detection
        self.line_min_length = 50  # Minimum length for line detection
        self.line_max_gap = 10     # Maximum gap between line segments
        self.wall_thickness = 5    # Thickness of walls for checking
        self.invert_binary = False # Whether to invert the binary image
        
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
        from vimaze.graph import Graph
        
        if self.timer:
            self.timer.start('processing', 'image')
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Preprocess the image
        preprocessed = self._preprocess_image(image)
        
        # Detect grid
        grid_lines, cell_size = self._detect_grid(preprocessed)
        
        # Determine maze dimensions
        rows, cols = self._determine_dimensions(grid_lines, cell_size, preprocessed.shape)
        
        # Extract walls
        wall_map = self._detect_walls(preprocessed, grid_lines, rows, cols)
        
        # Create graph representation
        graph = self._create_graph(rows, cols, wall_map)
        
        if self.timer:
            self.timer.stop()
        
        # Save debug visualizations if debug mode is enabled
        if self.debug_mode:
            self._save_debug_visualizations(image, preprocessed, grid_lines, rows, cols, wall_map)
        
        return graph, rows, cols
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for maze detection.
        
        Args:
            image: Input color image
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better handling of different lighting conditions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Invert if needed (depending on the maze image - walls might be white instead of black)
        if self.invert_binary:
            binary = cv2.bitwise_not(binary)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _detect_grid(self, binary_image: np.ndarray) -> Tuple[Dict[str, List[float]], float]:
        """
        Detect grid lines in the maze.
        
        Args:
            binary_image: Binary image where walls are white (255) and paths are black (0)
            
        Returns:
            A tuple containing:
            - dict: Grid lines as {'horizontal': list of y-coordinates, 'vertical': list of x-coordinates}
            - float: Estimated cell size
        """
        # Detect edges
        edges = cv2.Canny(binary_image, 50, 150)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100, 
            minLineLength=self.line_min_length, 
            maxLineGap=self.line_max_gap
        )
        
        if lines is None or len(lines) == 0:
            raise ValueError("No lines detected in the image. Try adjusting processing parameters.")
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
                horizontal_lines.append(min(y1, y2) + abs(y2 - y1) // 2)
            else:  # Vertical line
                vertical_lines.append(min(x1, x2) + abs(x2 - x1) // 2)
        
        # Cluster lines to find grid lines
        horizontal_grid = self._cluster_lines(horizontal_lines)
        vertical_grid = self._cluster_lines(vertical_lines)
        
        # Calculate cell size
        h_diffs = [horizontal_grid[i+1] - horizontal_grid[i] for i in range(len(horizontal_grid)-1)]
        v_diffs = [vertical_grid[i+1] - vertical_grid[i] for i in range(len(vertical_grid)-1)]
        
        if not h_diffs or not v_diffs:
            raise ValueError("Could not determine cell size. The maze grid may not be clearly visible.")
        
        # Use the median distance as the cell size
        cell_size = (np.median(h_diffs) + np.median(v_diffs)) / 2
        
        return {'horizontal': horizontal_grid, 'vertical': vertical_grid}, cell_size
    
    def _cluster_lines(self, lines: List[float], threshold: int = 10) -> List[float]:
        """
        Cluster lines that are close to each other to identify unique grid lines.
        
        Args:
            lines: List of line positions
            threshold: Distance threshold for clustering
            
        Returns:
            List of clustered line positions
        """
        if not lines:
            return []
        
        # Sort lines
        lines = sorted(lines)
        
        # Cluster lines
        clusters = []
        current_cluster = [lines[0]]
        
        for i in range(1, len(lines)):
            if lines[i] - lines[i-1] <= threshold:
                current_cluster.append(lines[i])
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [lines[i]]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def _determine_dimensions(self, grid_lines: Dict[str, List[float]], cell_size: float, 
                             image_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """
        Determine maze dimensions from grid lines.
        
        Args:
            grid_lines: Grid lines as {'horizontal': list of y-coordinates, 'vertical': list of x-coordinates}
            cell_size: Estimated cell size
            image_shape: Shape of the image
            
        Returns:
            A tuple containing (rows, columns)
        """
        # Number of cells is one less than number of grid lines
        rows = len(grid_lines['horizontal']) - 1
        cols = len(grid_lines['vertical']) - 1
        
        # If grid lines were not detected properly, estimate from image size and cell size
        if rows <= 0 or cols <= 0:
            height, width = image_shape[:2]
            rows = max(1, round(height / cell_size) - 1)
            cols = max(1, round(width / cell_size) - 1)
        
        return rows, cols
    
    def _detect_walls(self, binary_image: np.ndarray, grid_lines: Dict[str, List[float]], 
                     rows: int, cols: int) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect walls between cells.
        
        Args:
            binary_image: Binary image
            grid_lines: Grid lines
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            Set of wall positions as ((row1, col1), (row2, col2))
        """
        wall_map = set()
        
        # For each cell, check if there are walls on each side
        for row in range(rows):
            for col in range(cols):
                # Get cell boundaries
                top = int(grid_lines['horizontal'][row])
                bottom = int(grid_lines['horizontal'][row+1])
                left = int(grid_lines['vertical'][col])
                right = int(grid_lines['vertical'][col+1])
                
                # Check top wall (shared with cell above)
                if row > 0:
                    top_wall_roi = binary_image[top-self.wall_thickness:top+self.wall_thickness, 
                                               left+self.wall_thickness:right-self.wall_thickness]
                    if np.mean(top_wall_roi) > self.wall_threshold:  # Wall exists
                        wall_map.add(((row, col), (row-1, col)))
                        wall_map.add(((row-1, col), (row, col)))  # Add both directions
                
                # Check left wall (shared with cell to the left)
                if col > 0:
                    left_wall_roi = binary_image[top+self.wall_thickness:bottom-self.wall_thickness, 
                                                left-self.wall_thickness:left+self.wall_thickness]
                    if np.mean(left_wall_roi) > self.wall_threshold:  # Wall exists
                        wall_map.add(((row, col), (row, col-1)))
                        wall_map.add(((row, col-1), (row, col)))  # Add both directions
                
                # For cells at the edge of the maze, check for outer walls
                # Bottom outer wall (for last row)
                if row == rows - 1:
                    bottom_wall_roi = binary_image[bottom-self.wall_thickness:min(bottom+self.wall_thickness, binary_image.shape[0]), 
                                                  left+self.wall_thickness:right-self.wall_thickness]
                    if np.mean(bottom_wall_roi) > self.wall_threshold:  # Wall exists
                        wall_map.add(((row, col), (row+1, col)))
                        wall_map.add(((row+1, col), (row, col)))
                
                # Right outer wall (for last column)
                if col == cols - 1:
                    right_wall_roi = binary_image[top+self.wall_thickness:bottom-self.wall_thickness, 
                                                 right-self.wall_thickness:min(right+self.wall_thickness, binary_image.shape[1])]
                    if np.mean(right_wall_roi) > self.wall_threshold:  # Wall exists
                        wall_map.add(((row, col), (row, col+1)))
                        wall_map.add(((row, col+1), (row, col)))
        
        return wall_map
    
    def _create_graph(self, rows: int, cols: int, 
                     wall_map: Set[Tuple[Tuple[int, int], Tuple[int, int]]]) -> 'Graph':
        """
        Create a Graph representation of the maze.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            wall_map: Set of wall positions
            
        Returns:
            Graph: The Graph representation of the maze
        """
        from vimaze.graph import Graph
        
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
                    (row, col-1),  # West
                    (row+1, col),  # South
                    (row, col+1)   # East
                ]
                
                for nr, nc in neighbors:
                    # Check if neighbor is within bounds and there's no wall
                    if 0 <= nr < rows and 0 <= nc < cols:
                        wall_exists = ((row, col), (nr, nc)) in wall_map
                        if not wall_exists:
                            graph.connect_nodes((row, col), (nr, nc))
        
        return graph
    
    def _save_debug_visualizations(self, image, preprocessed, grid_lines, rows, cols, wall_map):
        """
        Save debug visualizations to help diagnose processing issues.
        
        Args:
            image: Original image
            preprocessed: Preprocessed binary image
            grid_lines: Detected grid lines
            rows: Number of rows
            cols: Number of columns
            wall_map: Detected walls
        """
        import os
        
        # Create debug directory if it doesn't exist
        if not os.path.exists("debug"):
            os.makedirs("debug")
        
        # Save original image
        cv2.imwrite("debug/1_original.png", image)
        
        # Save preprocessed binary image
        cv2.imwrite("debug/2_binary.png", preprocessed)
        
        # Create grid visualization
        grid_viz = image.copy()
        for y in grid_lines['horizontal']:
            cv2.line(grid_viz, (0, int(y)), (image.shape[1], int(y)), (0, 255, 0), 2)
        for x in grid_lines['vertical']:
            cv2.line(grid_viz, (int(x), 0), (int(x), image.shape[0]), (0, 0, 255), 2)
        cv2.imwrite("debug/3_grid_detected.png", grid_viz)
        
        # Create wall visualization
        wall_viz = image.copy()
        for (r1, c1), (r2, c2) in wall_map:
            # Only process each wall once (not both directions)
            if r1 > r2 or (r1 == r2 and c1 > c2):
                continue
                
            # Calculate pixel positions based on grid lines
            if 0 <= r1 < len(grid_lines['horizontal']) and 0 <= c1 < len(grid_lines['vertical']) and \
               0 <= r2 < len(grid_lines['horizontal']) and 0 <= c2 < len(grid_lines['vertical']):
                
                p1_y = int((grid_lines['horizontal'][r1] + grid_lines['horizontal'][min(r1+1, len(grid_lines['horizontal'])-1)]) / 2)
                p1_x = int((grid_lines['vertical'][c1] + grid_lines['vertical'][min(c1+1, len(grid_lines['vertical'])-1)]) / 2)
                
                p2_y = int((grid_lines['horizontal'][r2] + grid_lines['horizontal'][min(r2+1, len(grid_lines['horizontal'])-1)]) / 2)
                p2_x = int((grid_lines['vertical'][c2] + grid_lines['vertical'][min(c2+1, len(grid_lines['vertical'])-1)]) / 2)
                
                cv2.line(wall_viz, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)
        
        cv2.imwrite("debug/4_walls_detected.png", wall_viz)
        
        # Add this at the end of maze_image_processor.py


    def test_maze_image_processing(self, image_path, debug=True, invert_binary=False, wall_threshold=127):
        """
        Test the maze image processing with debug mode.
        This function can be called from a command line or script for testing purposes.
        
        Args:
            image_path: Path to the maze image file
            debug: Whether to enable debug mode
            invert_binary: Whether to invert the binary image
            wall_threshold: Threshold for wall detection
        """
        logging.info(f"Testing maze image processing: {image_path}")
        logging.info(f"Parameters: debug={debug}, invert_binary={invert_binary}, wall_threshold={wall_threshold}")
        
        try:
            # Configure the maze image processor through the maze instance
            self.maze.init_from_image_with_params(
                image_path, 
                invert_binary=invert_binary,
                wall_threshold=wall_threshold,
                debug_mode=debug
            )
            
            logging.info(f"Maze loaded successfully. Size: {self.maze.rows}x{self.maze.cols}")
            self.maze_canvas.delete("all")
            self.maze.display_maze()
            
            return True, f"Maze loaded successfully. Size: {self.maze.rows}x{self.maze.cols}"
        except Exception as e:
            logging.error(f"Failed to process maze image: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return False, f"Error: {str(e)}"
