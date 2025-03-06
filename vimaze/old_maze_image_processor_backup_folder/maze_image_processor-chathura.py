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
        """
        from vimaze.ds.graph import Graph

        if self.timer:
            self.timer.start('processing', 'image')

        # Create debug directory if needed
        if self.debug_mode and not os.path.exists("debug"):
            os.makedirs("debug")

        # Step 1: Load and preprocess the image
        logging.debug(f"Loading image: {image_path}")
        original_img, binary_img, gray_img = self._preprocess_image(image_path)
        
        if self.debug_mode:
            cv2.imwrite("debug/01_original.png", original_img)
            cv2.imwrite("debug/02_binary.png", binary_img)

        # Step 2: Detect grid size and extract maze area
        grid_size, maze_binary, maze_bounds = self._detect_grid_size(binary_img, gray_img)
        
        if self.debug_mode:
            cv2.imwrite("debug/03_maze_area.png", maze_binary)
            
            # Draw grid on original image for debugging
            debug_grid = original_img.copy()
            x0, y0, w, h = maze_bounds
            cell_h, cell_w = h // grid_size, w // grid_size
            
            for i in range(grid_size + 1):
                y_pos = y0 + int(i * cell_h)
                x_pos = x0 + int(i * cell_w)
                cv2.line(debug_grid, (x0, y_pos), (x0 + w, y_pos), (0, 255, 0), 1)  # Horizontal
                cv2.line(debug_grid, (x_pos, y0), (x_pos, y0 + h), (0, 255, 0), 1)  # Vertical
                
            cv2.imwrite("debug/04_grid_detection.png", debug_grid)

        # Step 3: Extract cells from the maze
        cells = self._extract_cells(maze_binary, grid_size)
        
        # Step 4: Analyze cells to detect walls
        walls = self._analyze_all_cells(cells, grid_size)
        
        if self.debug_mode:
            self._visualize_walls(walls, grid_size)

        # Step 5: Create graph representation from walls
        graph = self._create_graph_from_walls(grid_size, walls)
        
        if self.debug_mode:
            self._visualize_maze_graph(graph, grid_size)

        if self.timer:
            self.timer.stop()

        logging.debug(f"Maze processed successfully: {grid_size}x{grid_size} cells")

        return graph, grid_size, grid_size
    
    def _preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the maze image for analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (original image, binary image, grayscale image)
        """
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        # Resize image if too large while maintaining aspect ratio
        max_dimension = 1000
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        
        # Clean up noise using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Invert binary image if needed
        if self.invert_binary:
            binary = cv2.bitwise_not(binary)
            
        return img, binary, gray
    
    def _detect_grid_size(self, binary_img: np.ndarray, gray_img: np.ndarray) -> Tuple[int, np.ndarray, Tuple[int, int, int, int]]:
        """
        Detect the grid size and extract the maze area.
        
        Args:
            binary_img: Binary image
            gray_img: Grayscale image
            
        Returns:
            Tuple of (grid_size, maze_binary, maze_bounds)
        """
        # Get image dimensions
        h, w = binary_img.shape
        
        # Find contours to get the maze boundary
        contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in the image")
        
        # Get the main contour (the maze boundary)
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Crop the image to the maze boundaries
        maze_binary = binary_img[y:y+h, x:x+w].copy()
        maze_gray = gray_img[y:y+h, x:x+w].copy()
        
        # Use edge detection for line detection
        edges = cv2.Canny(maze_binary, 50, 150, apertureSize=3)
        
        # Detect lines using probabilistic Hough transform
        min_line_length = min(w, h) // 16
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                            minLineLength=min_line_length, maxLineGap=10)
        
        if lines is None:
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                minLineLength=min_line_length//2, maxLineGap=30)
            if lines is None:
                raise ValueError("Could not detect grid lines in the maze")
        
        # Categorize lines as horizontal or vertical
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle of the line
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Horizontal lines (angle close to 0 or 180 degrees)
            if angle < 30 or angle > 150:
                horizontal_lines.append((y1 + y2) // 2)  # y-coordinate
            # Vertical lines (angle close to 90 degrees)
            elif 60 < angle < 120:
                vertical_lines.append((x1 + x2) // 2)  # x-coordinate
        
        # Group similar coordinates to handle slight variations
        def group_coordinates(coords, threshold=None):
            if not coords:
                return []
            if threshold is None:
                threshold = min(w, h) // 30  # Adaptive threshold based on image size
            coords = sorted(coords)
            groups = [[coords[0]]]
            for coord in coords[1:]:
                if coord - groups[-1][-1] <= threshold:
                    groups[-1].append(coord)
                else:
                    groups.append([coord])
            return [sum(group) // len(group) for group in groups]
        
        # Group similar coordinates
        horizontal_positions = group_coordinates(horizontal_lines)
        vertical_positions = group_coordinates(vertical_lines)
        
        # Calculate approximate cell size and count
        cell_width = w // len(vertical_positions) if vertical_positions else w
        cell_height = h // len(horizontal_positions) if horizontal_positions else h
        
        # Count number of grid cells
        num_cells_x = len(vertical_positions) + 1
        num_cells_y = len(horizontal_positions) + 1
        
        # Determine grid size (typical maze sizes are 9x9 or 16x16)
        grid_size = max(num_cells_x, num_cells_y)
        
        # Additional refinement based on potential standard grid sizes
        potential_sizes = [9, 16]
        optimal_grid_size = None
        best_score = float('-inf')
        
        for size in potential_sizes:
            approx_cell_width = w / size
            approx_cell_height = h / size
            
            # Score how well this grid size matches the detected lines
            score = 0
            
            # Check horizontal lines
            for i in range(1, size):
                expected_y = int(y + i * approx_cell_height)
                # Find closest detected horizontal line
                closest_diff = min([abs(expected_y - pos) for pos in horizontal_positions], default=w)
                score -= closest_diff
            
            # Check vertical lines
            for i in range(1, size):
                expected_x = int(x + i * approx_cell_width)
                # Find closest detected vertical line
                closest_diff = min([abs(expected_x - pos) for pos in vertical_positions], default=h)
                score -= closest_diff
            
            if score > best_score:
                best_score = score
                optimal_grid_size = size
        
        # Final decision: Use optimal_grid_size if available, otherwise use the count-based approach
        final_grid_size = optimal_grid_size if optimal_grid_size is not None else grid_size
        
        # If number of detected lines suggests 9x9 or 16x16, prefer those over other sizes
        if num_cells_x > 6 or num_cells_y > 6:
            if num_cells_x <= 12 or num_cells_y <= 12:
                final_grid_size = 9
            else:
                final_grid_size = 16

        # Match to closest standard size if not already one of the standard sizes
        if final_grid_size not in [9, 16]:
            diffs = [(abs(final_grid_size - size), size) for size in [9, 16]]
            final_grid_size = min(diffs)[1]
            
        logging.debug(f"Detected grid dimensions: {num_cells_x}x{num_cells_y}")
        logging.debug(f"Determined grid size: {final_grid_size}x{final_grid_size}")
        
        return final_grid_size, maze_binary, (x, y, w, h)
    
    def _extract_cells(self, maze_img: np.ndarray, grid_size: int) -> List[List[np.ndarray]]:
        """
        Extract individual cells from the maze.
        
        Args:
            maze_img: Binary image of the maze
            grid_size: Size of the grid (e.g., 9 for 9x9)
            
        Returns:
            2D list of cell images
        """
        h, w = maze_img.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        
        cells = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                # Extract cell
                cell = maze_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                row.append(cell)
            cells.append(row)
        
        return cells

    def _analyze_cell_walls(self, cell: np.ndarray, i: int, j: int, grid_size: int) -> List[int]:
        """
        Analyze a cell to detect which walls are present.
        
        Args:
            cell: Cell image
            i: Row index
            j: Column index
            grid_size: Grid size
            
        Returns:
            List of wall indicators [top, right, bottom, left] (1=wall, 0=no wall)
        """
        h, w = cell.shape
        
        # Define narrow regions along each edge to check for walls
        wall_thickness = max(2, min(h, w) // 10)  # Adjust based on cell size
        threshold_percentage = 0.4  # Percentage of white pixels needed to identify a wall
        
        # Check top wall
        top_region = cell[0:wall_thickness, :]
        top_wall = 1 if np.sum(top_region) / (top_region.size * 255) > threshold_percentage else 0
        
        # Check right wall
        right_region = cell[:, w-wall_thickness:w]
        right_wall = 1 if np.sum(right_region) / (right_region.size * 255) > threshold_percentage else 0
        
        # Check bottom wall
        bottom_region = cell[h-wall_thickness:h, :]
        bottom_wall = 1 if np.sum(bottom_region) / (bottom_region.size * 255) > threshold_percentage else 0
        
        # Check left wall
        left_region = cell[:, 0:wall_thickness]
        left_wall = 1 if np.sum(left_region) / (left_region.size * 255) > threshold_percentage else 0
        
        # Special handling for edge cells - outer boundary should always have walls
        if i == 0:  # Top row
            top_wall = 1
        if j == grid_size - 1:  # Rightmost column
            right_wall = 1
        if i == grid_size - 1:  # Bottom row
            bottom_wall = 1
        if j == 0:  # Leftmost column
            left_wall = 1
        
        return [top_wall, right_wall, bottom_wall, left_wall]
    
    def _ensure_wall_consistency(self, walls: List[List[int]], grid_size: int) -> List[List[int]]:
        """
        Ensure consistency of walls between adjacent cells.
        
        Args:
            walls: List of wall indicators for each cell
            grid_size: Grid size
            
        Returns:
            Consistent list of wall indicators
        """
        # Convert the flat list of walls into a 2D grid for easier adjacency checks
        grid_walls = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                idx = i * grid_size + j
                row.append(walls[idx])
            grid_walls.append(row)
        
        # Fix inconsistencies
        for i in range(grid_size):
            for j in range(grid_size):
                # Check right wall consistency
                if j < grid_size - 1:
                    # If cell has right wall, adjacent cell should have left wall
                    if grid_walls[i][j][1] == 1:
                        grid_walls[i][j+1][3] = 1
                    # If adjacent cell has left wall, cell should have right wall
                    elif grid_walls[i][j+1][3] == 1:
                        grid_walls[i][j][1] = 1
                
                # Check bottom wall consistency
                if i < grid_size - 1:
                    # If cell has bottom wall, cell below should have top wall
                    if grid_walls[i][j][2] == 1:
                        grid_walls[i+1][j][0] = 1
                    # If cell below has top wall, cell should have bottom wall
                    elif grid_walls[i+1][j][0] == 1:
                        grid_walls[i][j][2] = 1
        
        # Convert back to flat list
        consistent_walls = []
        for i in range(grid_size):
            for j in range(grid_size):
                consistent_walls.append(grid_walls[i][j])
        
        return consistent_walls
    
    def _analyze_all_cells(self, cells: List[List[np.ndarray]], grid_size: int) -> List[List[int]]:
        """
        Analyze all cells in the maze to detect walls.
        
        Args:
            cells: 2D list of cell images
            grid_size: Grid size
            
        Returns:
            List of wall indicators for each cell
        """
        walls = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell_walls = self._analyze_cell_walls(cells[i][j], i, j, grid_size)
                walls.append(cell_walls)
        
        # Apply wall consistency check
        walls = self._ensure_wall_consistency(walls, grid_size)
        
        return walls
    
    def _visualize_walls(self, walls: List[List[int]], grid_size: int) -> None:
        """
        Create a visualization of the detected walls.
        
        Args:
            walls: List of wall indicators for each cell
            grid_size: Grid size
        """
        # Create a debug grid image
        debug_grid = np.ones((grid_size * 50, grid_size * 50, 3), dtype=np.uint8) * 255
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                cell_walls = walls[idx]
                
                cell_y, cell_x = i * 50, j * 50
                # Draw the cell
                cv2.rectangle(debug_grid, (cell_x, cell_y), (cell_x + 50, cell_y + 50), (200, 200, 200), 1)
                # Draw detected walls
                if cell_walls[0]:  # Top
                    cv2.line(debug_grid, (cell_x, cell_y), (cell_x + 50, cell_y), (0, 0, 0), 2)
                if cell_walls[1]:  # Right
                    cv2.line(debug_grid, (cell_x + 50, cell_y), (cell_x + 50, cell_y + 50), (0, 0, 0), 2)
                if cell_walls[2]:  # Bottom
                    cv2.line(debug_grid, (cell_x, cell_y + 50), (cell_x + 50, cell_y + 50), (0, 0, 0), 2)
                if cell_walls[3]:  # Left
                    cv2.line(debug_grid, (cell_x, cell_y), (cell_x, cell_y + 50), (0, 0, 0), 2)
                
                # Add labels
                cv2.putText(debug_grid, f"{i},{j}", (cell_x + 15, cell_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        cv2.imwrite("debug/05_wall_detection.png", debug_grid)
    
    def _create_graph_from_walls(self, grid_size: int, walls: List[List[int]]) -> 'Graph':
        """
        Create a graph representation from detected walls.
        
        Args:
            grid_size: Grid size
            walls: List of wall indicators for each cell
            
        Returns:
            Graph: Graph representation of the maze
        """
        from vimaze.ds.graph import Graph
        
        # Create a new graph
        graph = Graph()
        
        # Add nodes for all cells
        for i in range(grid_size):
            for j in range(grid_size):
                graph.add_node((i, j))
        
        # Add connections where there are no walls
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                cell_walls = walls[idx]
                
                # Check each direction
                # If there's no wall, connect to the adjacent cell
                
                # Top (no top wall)
                if i > 0 and cell_walls[0] == 0:
                    graph.connect_nodes((i, j), (i-1, j))
                
                # Right (no right wall)
                if j < grid_size - 1 and cell_walls[1] == 0:
                    graph.connect_nodes((i, j), (i, j+1))
                
                # Bottom (no bottom wall)
                if i < grid_size - 1 and cell_walls[2] == 0:
                    graph.connect_nodes((i, j), (i+1, j))
                
                # Left (no left wall)
                if j > 0 and cell_walls[3] == 0:
                    graph.connect_nodes((i, j), (i, j-1))
        
        return graph
    
    def _visualize_maze_graph(self, graph: 'Graph', grid_size: int) -> None:
        """
        Create visualizations of the maze graph for debugging.
        
        Args:
            graph: Graph representation of the maze
            grid_size: Grid size
        """
        # Create visualization of graph structure
        cell_size = 30  # Size of cells in visualization
        viz_width = grid_size * cell_size + 30
        viz_height = grid_size * cell_size + 30
        graph_viz = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 255
        
        # Draw grid
        for r in range(grid_size + 1):
            y = r * cell_size + 15
            cv2.line(graph_viz, (15, y), (viz_width - 15, y), (200, 200, 200), 1)
        
        for c in range(grid_size + 1):
            x = c * cell_size + 15
            cv2.line(graph_viz, (x, 15), (x, viz_height - 15), (200, 200, 200), 1)
        
        # Draw nodes and connections
        for r in range(grid_size):
            for c in range(grid_size):
                # Node center
                cy = r * cell_size + 30
                cx = c * cell_size + 30
                
                # Draw node
                cv2.circle(graph_viz, (cx, cy), 4, (0, 0, 255), -1)
                
                # Get connections from the graph
                node_pos = (r, c)
                if node_pos in graph.nodes:
                    node = graph.nodes[str(node_pos[0]) + "," + str(node_pos[1])]
                    for neighbor_node in node.neighbors:
                        n_pos = neighbor_node.position
                        n_r, n_c = n_pos
                        n_cy = n_r * cell_size + 30
                        n_cx = n_c * cell_size + 30
                        
                        cv2.line(graph_viz, (cx, cy), (n_cx, n_cy), (0, 255, 0), 2)
        
        # Save graph visualization
        cv2.imwrite("debug/06_graph_representation.png", graph_viz)
        
        # Create a visualization showing the maze with paths
        maze_viz = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 255
        
        # Iterate through all cells
        for r in range(grid_size):
            for c in range(grid_size):
                cell_x = c * cell_size + 15
                cell_y = r * cell_size + 15
                
                # Draw cell outline
                cv2.rectangle(maze_viz, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), 
                             (220, 220, 220), 1)
                
                # Check neighboring cells to determine walls
                node_key = str(r) + "," + str(c)
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