import cv2
import numpy as np
import os
from typing import TYPE_CHECKING, Tuple, List, Set, Dict, Optional
import logging
from collections import Counter

if TYPE_CHECKING:
    from vimaze.ds.graph import Graph
    from vimaze.timer import Timer

class MazeImageProcessor:
    """
    Improved maze image processor that accurately detects the cell grid
    and represents walls as edges between cells.
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
        
        Args:
            image_path: Path to the maze image file
            
        Returns:
            A tuple containing:
            - Graph: The Graph representation of the maze
            - int: Number of rows in the maze
            - int: Number of columns in the maze
        """
        from vimaze.ds.graph import Graph
        
        if self.timer:
            self.timer.start('processing', 'image')
        
        # Create debug directory if needed
        if self.debug_mode and not os.path.exists("debug"):
            os.makedirs("debug")
        
        # Step 1: Load and preprocess the image
        logging.debug(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        if self.debug_mode:
            cv2.imwrite("debug/01_original.png", image)
        
        # Step 2: Convert to grayscale and apply thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, self.wall_threshold, 255, cv2.THRESH_BINARY)
        
        # Invert if needed (walls should be black, paths white)
        if self.invert_binary:
            binary = cv2.bitwise_not(binary)
        
        if self.debug_mode:
            cv2.imwrite("debug/02_binary.png", binary)
        
        # Step 3: Extract maze area and remove border
        maze_binary, border_removed = self._extract_maze_and_remove_border(binary)
        
        if self.debug_mode:
            cv2.imwrite("debug/03_maze_area.png", maze_binary)
            if border_removed is not None:
                cv2.imwrite("debug/03b_border_removed.png", border_removed)
        
        # Step 4: Detect grid parameters (cell size and alignment)
        cell_size, offset_x, offset_y, rows, cols = self._detect_grid_parameters(maze_binary)
        
        logging.debug(f"Detected grid: cell_size={cell_size}, offset=({offset_x}, {offset_y}), size={rows}x{cols}")
        
        if self.debug_mode:
            grid_viz = cv2.cvtColor(maze_binary.copy(), cv2.COLOR_GRAY2BGR)
            
            # Draw detected grid
            for r in range(rows + 1):
                y = offset_y + r * cell_size
                cv2.line(grid_viz, (0, y), (grid_viz.shape[1], y), (0, 255, 0), 1)
            
            for c in range(cols + 1):
                x = offset_x + c * cell_size
                cv2.line(grid_viz, (x, 0), (x, grid_viz.shape[0]), (0, 255, 0), 1)
            
            cv2.imwrite("debug/04_grid_detection.png", grid_viz)
        
        # Step 5: Detect walls between cells
        wall_edges = self._detect_wall_edges(maze_binary, rows, cols, cell_size, offset_x, offset_y)
        
        if self.debug_mode:
            wall_viz = cv2.cvtColor(maze_binary.copy(), cv2.COLOR_GRAY2BGR)
            
            # Draw cell centers and walls
            for r in range(rows):
                for c in range(cols):
                    # Cell center
                    cy = offset_y + r * cell_size + cell_size // 2
                    cx = offset_x + c * cell_size + cell_size // 2
                    
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
                                cv2.line(wall_viz, (cx, cy), (cx + dx, cy + dy), (0, 0, 255), 2)
                            else:
                                # Path - green line
                                cv2.line(wall_viz, (cx, cy), (cx + dx, cy + dy), (0, 255, 0), 1)
            
            cv2.imwrite("debug/05_wall_detection.png", wall_viz)
        
        # Step 6: Create graph representation
        graph = self._create_graph_with_walls(rows, cols, wall_edges)
        
        if self.debug_mode:
            self._visualize_maze_graph(graph, rows, cols, wall_edges)
        
        if self.timer:
            self.timer.stop()
            
        logging.debug(f"Maze processed successfully: {rows}x{cols} cells")
        
        return graph, rows, cols
    
    def _extract_maze_and_remove_border(self, binary: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract the maze area and attempt to remove any border around it.
        
        Args:
            binary: Binary image
            
        Returns:
            Tuple of (processed maze binary image, border removed image or None)
        """
        # First extract the maze area
        inverted = cv2.bitwise_not(binary)
        
        # Find contours
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logging.warning("No contours found, using entire image")
            return binary, None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract the maze area with small margin
        margin = 5
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(binary.shape[1], x + w + margin)
        y_end = min(binary.shape[0], y + h + margin)
        
        maze_binary = binary[y_start:y_end, x_start:x_end].copy()
        
        # Now try to detect and remove any border
        # For simplicity, we'll just crop a few pixels from each side if they're mostly black
        border_removed = None
        
        try:
            # Check if there's a thick border
            border_size = self._detect_border_size(maze_binary)
            
            if border_size > 0:
                # Crop the border
                height, width = maze_binary.shape
                border_removed = maze_binary[border_size:height-border_size, border_size:width-border_size]
                return border_removed, maze_binary
        except Exception as e:
            logging.warning(f"Border removal failed: {str(e)}")
        
        return maze_binary, border_removed
    
    def _detect_border_size(self, binary: np.ndarray) -> int:
        """
        Detect the size of the border around the maze, if any.
        
        Args:
            binary: Binary image
            
        Returns:
            Estimated border size in pixels
        """
        height, width = binary.shape
        
        # Check top, bottom, left, right borders
        top_border = 0
        for y in range(height // 4):
            if np.mean(binary[y, :]) < 127:
                top_border = y + 1
            else:
                break
        
        bottom_border = 0
        for y in range(height - 1, height - height // 4, -1):
            if np.mean(binary[y, :]) < 127:
                bottom_border = height - y
            else:
                break
        
        left_border = 0
        for x in range(width // 4):
            if np.mean(binary[:, x]) < 127:
                left_border = x + 1
            else:
                break
        
        right_border = 0
        for x in range(width - 1, width - width // 4, -1):
            if np.mean(binary[:, x]) < 127:
                right_border = width - x
            else:
                break
        
        # Use the minimum consistent border size
        border_sizes = [top_border, bottom_border, left_border, right_border]
        consistent_sizes = [size for size in border_sizes if size > 0]
        
        if not consistent_sizes:
            return 0
        
        return min(consistent_sizes)
    
    def _detect_grid_parameters(self, maze_binary: np.ndarray) -> Tuple[int, int, int, int, int]:
        """
        Detect grid parameters including cell size, offset, and dimensions.
        
        Args:
            maze_binary: Binary image of the maze
            
        Returns:
            Tuple of (cell_size, offset_x, offset_y, rows, cols)
        """
        height, width = maze_binary.shape
        
        # Use projection profiles to detect cell size
        horizontal_profile = np.sum(255 - maze_binary, axis=1)
        vertical_profile = np.sum(255 - maze_binary, axis=0)
        
        # Find peaks in the profiles - these correspond to walls
        h_peaks = self._find_peaks_in_profile(horizontal_profile)
        v_peaks = self._find_peaks_in_profile(vertical_profile)
        
        # Calculate spacing between peaks - these correspond to cell sizes
        h_spacings = np.diff(h_peaks)
        v_spacings = np.diff(v_peaks)
        
        # Filter out very small spacings (noise)
        h_spacings = h_spacings[h_spacings > 5]
        v_spacings = v_spacings[v_spacings > 5]
        
        # Get most common spacing
        h_cell_size = self._most_common_value(h_spacings) if len(h_spacings) > 0 else 20
        v_cell_size = self._most_common_value(v_spacings) if len(v_spacings) > 0 else 20
        
        # For mazes with square cells, use the average
        cell_size = (h_cell_size + v_cell_size) // 2
        
        # Determine grid offset (where the grid starts)
        offset_y = h_peaks[0] if len(h_peaks) > 0 else 0
        offset_x = v_peaks[0] if len(v_peaks) > 0 else 0
        
        # Now adjust offsets to center cells on corridors rather than walls
        offset_y = max(0, offset_y - cell_size // 2)
        offset_x = max(0, offset_x - cell_size // 2)
        
        # Calculate grid dimensions
        rows = (height - offset_y) // cell_size
        cols = (width - offset_x) // cell_size
        
        # Ensure we have at least one row and column
        rows = max(1, rows)
        cols = max(1, cols)
        
        return cell_size, offset_x, offset_y, rows, cols
    
    def _find_peaks_in_profile(self, profile: np.ndarray) -> np.ndarray:
        """
        Find peaks in a projection profile.
        
        Args:
            profile: 1D array of projection values
            
        Returns:
            Array of peak indices
        """
        # Normalize profile
        profile = profile / np.max(profile) if np.max(profile) > 0 else profile
        
        # Find peaks using a simple threshold approach
        threshold = 0.5
        peaks = []
        peak_candidate = False
        
        for i in range(1, len(profile) - 1):
            if profile[i] > threshold:
                if not peak_candidate:
                    peak_candidate = True
                    peaks.append(i)
            else:
                peak_candidate = False
        
        return np.array(peaks)
    
    def _most_common_value(self, values: np.ndarray) -> int:
        """
        Find the most common value in an array, rounded to an integer.
        
        Args:
            values: Array of values
            
        Returns:
            Most common value as an integer
        """
        if len(values) == 0:
            return 20  # Default cell size
        
        # Round values to nearest integer
        rounded = np.round(values).astype(int)
        
        # Count occurrences
        unique, counts = np.unique(rounded, return_counts=True)
        
        # Find most common
        idx = np.argmax(counts)
        
        return unique[idx]
    
    def _detect_wall_edges(self, maze_binary: np.ndarray, rows: int, cols: int, 
                          cell_size: int, offset_x: int, offset_y: int) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect walls between cells by checking the pixels between adjacent cell centers.
        
        Args:
            maze_binary: Binary image of the maze
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            cell_size: Size of each cell in pixels
            offset_x: X offset of the grid
            offset_y: Y offset of the grid
            
        Returns:
            Set of wall edges as ((row1, col1), (row2, col2))
        """
        wall_edges = set()
        height, width = maze_binary.shape
        
        # For each cell
        for r in range(rows):
            for c in range(cols):
                # Calculate cell center
                cy = offset_y + r * cell_size + cell_size // 2
                cx = offset_x + c * cell_size + cell_size // 2
                
                # Ensure cell center is within image bounds
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
                    ncy = offset_y + nr * cell_size + cell_size // 2
                    ncx = offset_x + nc * cell_size + cell_size // 2
                    
                    # Ensure neighbor cell center is within image bounds
                    if not (0 <= ncy < height and 0 <= ncx < width):
                        continue
                    
                    # Check if there's a wall between the cells
                    is_wall = self._check_for_wall(maze_binary, (cy, cx), (ncy, ncx))
                    
                    # If there's a wall, add the edge to the set
                    if is_wall:
                        wall_edges.add(((r, c), (nr, nc)))
        
        return wall_edges
    
    def _check_for_wall(self, maze_binary: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> bool:
        """
        Check if there's a wall between two points by sampling pixels along the line.
        
        Args:
            maze_binary: Binary image
            pt1: First point (y, x)
            pt2: Second point (y, x)
            
        Returns:
            True if there's a wall, False otherwise
        """
        # Get coordinates
        y1, x1 = pt1
        y2, x2 = pt2
        
        # Make sure points are within image bounds
        height, width = maze_binary.shape
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
            
            # Check if sample point is a wall (black pixel)
            if maze_binary[y, x] < self.wall_threshold:
                wall_count += 1
        
        # If a significant number of samples are black, consider it a wall
        return wall_count >= num_samples // 3
    
    def _create_graph_with_walls(self, rows: int, cols: int, 
                               wall_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]]) -> 'Graph':
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
    
    def _visualize_maze_graph(self, graph: 'Graph', rows: int, cols: int, 
                            wall_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]]) -> None:
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
    
    def test_maze_image_processing(self, image_path, debug=True, invert_binary=False, wall_threshold=127):
        """
        Test the maze image processing with debug mode.
        This function can be called from a command line or script for testing purposes.
        
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