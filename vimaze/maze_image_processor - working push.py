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
    Improved maze image processor using area-based wall detection with accurate
    cell size detection.
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
        self.wall_detection_threshold_h = 0.4  # Threshold for horizontal wall detection
        self.wall_detection_threshold_v = 0.4  # Threshold for vertical wall detection
        
    def process_image(self, image_path: str) -> Tuple['Graph', int, int]:
        """
        Process a maze image and return a Graph representation along with dimensions.
        """

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
        path_width, wall_width = self._find_min_spacing(maze_binary)
        cell_size = path_width + wall_width
        rows, cols = self._calculate_grid_dimensions(maze_binary, path_width, wall_width)
        
        logging.debug(f"Detected grid: path_width={path_width}, wall_width={wall_width}, cell_size={cell_size}, dimensions={rows}x{cols}")

        if self.debug_mode:
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

        # Step 5: Detect walls using area-based approach
        wall_indicators = self._detect_walls_cell_based(maze_binary, rows, cols, cell_size)
        
        # Step 6: Create graph representation from wall indicators
        graph = self._create_graph_from_wall_indicators(rows, cols, wall_indicators)

        if self.debug_mode:
            self._visualize_walls(wall_indicators, rows, cols)
            self._visualize_maze_graph(graph, rows, cols)

        if self.timer:
            self.timer.stop()

        logging.debug(f"Maze processed successfully: {rows}x{cols} cells")

        # Step 7: After wall detection, detect entry and exit points
        entry_point, exit_point = self._find_maze_entry_exit_points(rows, cols, wall_indicators)

        return graph, rows, cols, entry_point, exit_point
    
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
        Extract the maze area from the binary image based on start and end points,
        preserving the original orientation.
        
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
        
        # Calculate dimensions and create maze matrix with CORRECT orientation
        # In PIL, coordinates are (x,y) = (width, height) order
        # In NumPy arrays, dimensions are (height, width) order
        h = indexEast - indexWest + 1  # Height (j dimension)
        w = indexSout - indexNorth + 1  # Width (i dimension)
        
        # Create the maze with proper dimensions (height, width)
        maze = np.zeros((h, w), dtype=np.uint8)
        
        # Fill maze matrix preserving original orientation
        for i in range(indexNorth, indexSout + 1):
            for j in range(indexWest, indexEast + 1):
                if 0 <= i < width and 0 <= j < height:
                    # Map to the correct position in the maze array
                    # j-indexWest gives the row (y coordinate in numpy)
                    # i-indexNorth gives the column (x coordinate in numpy)
                    maze[j - indexWest, i - indexNorth] = pixel_matrix[i, j]
        
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
        
        if self.debug_mode:
            logging.debug(f"Detected path width: {path_width}, wall width: {wall_width}")
            if 'h_common' in locals() and 'v_common' in locals():
                logging.debug(f"Horizontal spaces: {h_common}")
                logging.debug(f"Vertical spaces: {v_common}")
                logging.debug(f"Horizontal walls: {h_common}")
                logging.debug(f"Vertical walls: {v_common}")
        
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
        # In NumPy arrays, shape[0] is height (rows) and shape[1] is width (columns)
        height, width = maze_matrix.shape
        cell_size = transition + wall
        
        # Calculate number of rows and columns
        rows = height // cell_size
        cols = width // cell_size
        
        # Log dimensions for debugging
        if self.debug_mode:
            logging.debug(f"Maze matrix shape: {maze_matrix.shape}")
            logging.debug(f"Cell size: {cell_size}")
            logging.debug(f"Calculated grid dimensions: {rows}x{cols}")
        
        # Ensure we have at least one row and column
        rows = max(1, rows)
        cols = max(1, cols)
        
        return rows, cols
    
    def _detect_walls_cell_based(self, maze_binary, rows, cols, cell_size):
        """
        Detect walls by analyzing entire cells for wall segments and determining
        which edges the walls are closest to.
        
        Args:
            maze_binary: Binary maze image (0=wall, 1=path)
            rows: Number of rows
            cols: Number of columns
            cell_size: Size of each cell
            
        Returns:
            3D list of wall indicators for each cell [row][col][wall_direction]
        """
        height, width = maze_binary.shape
        wall_indicators = [[[0, 0, 0, 0] for _ in range(cols)] for _ in range(rows)]
        
        # Debug visualizations
        if self.debug_mode:
            debug_img = np.zeros((height, width, 3), dtype=np.uint8)
            # Make path white
            debug_img[maze_binary == 1] = [255, 255, 255]
            # Make walls gray
            debug_img[maze_binary == 0] = [128, 128, 128]
        
        # First pass: analyze cells and detect walls
        for r in range(rows):
            for c in range(cols):
                # Calculate cell boundaries
                top = r * cell_size
                left = c * cell_size
                bottom = min((r + 1) * cell_size, height)
                right = min((c + 1) * cell_size, width)
                
                # Skip if out of bounds
                if top >= height or left >= width:
                    continue
                
                # Extract cell
                cell = maze_binary[top:bottom, left:right]
                
                # Add cell outline to debug visualization
                if self.debug_mode:
                    cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 1)
                
                # Analyze horizontal walls (look for rows with many wall pixels)
                h_wall_positions = []
                for y in range(cell.shape[0]):
                    # Count wall pixels in this row
                    wall_pixels = np.sum(cell[y, :] == 0)
                    
                    # If row is mostly wall pixels (e.g., 70%)
                    if wall_pixels > cell.shape[1] * 0.7:
                        h_wall_positions.append(y)
                
                # Group adjacent positions (within a few pixels) to handle thick walls
                h_walls = []
                if h_wall_positions:
                    current_wall = [h_wall_positions[0]]
                    for pos in h_wall_positions[1:]:
                        if pos - current_wall[-1] <= 3:  # Adjust threshold as needed
                            current_wall.append(pos)
                        else:
                            h_walls.append(sum(current_wall) / len(current_wall))  # Average position
                            current_wall = [pos]
                    if current_wall:
                        h_walls.append(sum(current_wall) / len(current_wall))  # Add last wall
                
                # Assign walls to top or bottom edge based on position
                for wall_pos in h_walls:
                    if wall_pos < cell.shape[0] / 2:
                        wall_indicators[r][c][0] = 1  # Top wall
                        if self.debug_mode:
                            y_pos = int(top + wall_pos)
                            cv2.line(debug_img, (left, y_pos), (right, y_pos), (0, 0, 255), 1)
                    else:
                        wall_indicators[r][c][2] = 1  # Bottom wall
                        if self.debug_mode:
                            y_pos = int(top + wall_pos)
                            cv2.line(debug_img, (left, y_pos), (right, y_pos), (0, 0, 255), 1)
                
                # Analyze vertical walls (look for columns with many wall pixels)
                v_wall_positions = []
                for x in range(cell.shape[1]):
                    # Count wall pixels in this column
                    wall_pixels = np.sum(cell[:, x] == 0)
                    
                    # If column is mostly wall pixels
                    if wall_pixels > cell.shape[0] * 0.7:
                        v_wall_positions.append(x)
                
                # Group adjacent positions to handle thick walls
                v_walls = []
                if v_wall_positions:
                    current_wall = [v_wall_positions[0]]
                    for pos in v_wall_positions[1:]:
                        if pos - current_wall[-1] <= 3:  # Adjust threshold as needed
                            current_wall.append(pos)
                        else:
                            v_walls.append(sum(current_wall) / len(current_wall))  # Average position
                            current_wall = [pos]
                    if current_wall:
                        v_walls.append(sum(current_wall) / len(current_wall))  # Add last wall
                
                # Assign walls to left or right edge based on position
                for wall_pos in v_walls:
                    if wall_pos < cell.shape[1] / 2:
                        wall_indicators[r][c][3] = 1  # Left wall
                        if self.debug_mode:
                            x_pos = int(left + wall_pos)
                            cv2.line(debug_img, (x_pos, top), (x_pos, bottom), (0, 0, 255), 1)
                    else:
                        wall_indicators[r][c][1] = 1  # Right wall
                        if self.debug_mode:
                            x_pos = int(left + wall_pos)
                            cv2.line(debug_img, (x_pos, top), (x_pos, bottom), (0, 0, 255), 1)
        
        if self.debug_mode:
            cv2.imwrite("debug/05a_cell_analysis.png", debug_img)
        
        # Second pass: ensure wall consistency between adjacent cells
        for r in range(rows):
            for c in range(cols):
                # Right-left consistency
                if c < cols - 1:
                    if wall_indicators[r][c][1] == 1:  # Right wall
                        wall_indicators[r][c+1][3] = 1  # Left wall for cell to the right
                    elif wall_indicators[r][c+1][3] == 1:  # Left wall for cell to the right
                        wall_indicators[r][c][1] = 1  # Right wall
                
                # Bottom-top consistency
                if r < rows - 1:
                    if wall_indicators[r][c][2] == 1:  # Bottom wall
                        wall_indicators[r+1][c][0] = 1  # Top wall for cell below
                    elif wall_indicators[r+1][c][0] == 1:  # Top wall for cell below
                        wall_indicators[r][c][2] = 1  # Bottom wall
        
        # Third pass: Handle border walls while preserving entry/exit points
        # Count openings in each border to identify potential entry/exit
        top_openings = [c for c in range(cols) if wall_indicators[0][c][0] == 0]
        right_openings = [r for r in range(rows) if wall_indicators[r][cols-1][1] == 0]
        bottom_openings = [c for c in range(cols) if wall_indicators[rows-1][c][2] == 0]
        left_openings = [r for r in range(rows) if wall_indicators[r][0][3] == 0]
        
        # Log the openings found
        if self.debug_mode:
            logging.debug(f"Border openings - Top: {len(top_openings)}, Right: {len(right_openings)}, Bottom: {len(bottom_openings)}, Left: {len(left_openings)}")
        
        # Fill in top border walls (preserving potential entry/exit)
        preserve_top = []
        if len(top_openings) <= 2:  # Reasonable number of openings
            preserve_top = top_openings
        for c in range(cols):
            if c not in preserve_top:
                wall_indicators[0][c][0] = 1  # Add top wall
        
        # Fill in right border walls (preserving potential entry/exit)
        preserve_right = []
        if len(right_openings) <= 2:  # Reasonable number of openings
            preserve_right = right_openings
        for r in range(rows):
            if r not in preserve_right:
                wall_indicators[r][cols-1][1] = 1  # Add right wall
        
        # Fill in bottom border walls (preserving potential entry/exit)
        preserve_bottom = []
        if len(bottom_openings) <= 2:  # Reasonable number of openings
            preserve_bottom = bottom_openings
        for c in range(cols):
            if c not in preserve_bottom:
                wall_indicators[rows-1][c][2] = 1  # Add bottom wall
        
        # Fill in left border walls (preserving potential entry/exit)
        preserve_left = []
        if len(left_openings) <= 2:  # Reasonable number of openings
            preserve_left = left_openings
        for r in range(rows):
            if r not in preserve_left:
                wall_indicators[r][0][3] = 1  # Add left wall
        
        # Save the entry/exit points for later use if needed
        if hasattr(self, 'entry_exit_points'):
            # Find entry (typically on top or left border)
            entry_candidates = []
            if preserve_top:
                entry_candidates.extend([(0, c) for c in preserve_top])
            if preserve_left:
                entry_candidates.extend([(r, 0) for r in preserve_left])
            
            # Find exit (typically on bottom or right border)
            exit_candidates = []
            if preserve_bottom:
                exit_candidates.extend([(rows-1, c) for c in preserve_bottom])
            if preserve_right:
                exit_candidates.extend([(r, cols-1) for r in preserve_right])
            
            # Use closest to origin as entry, furthest from origin as exit
            if entry_candidates:
                entry_candidates.sort(key=lambda pos: pos[0] + pos[1])
                entry = entry_candidates[0]
            else:
                entry = (0, 0)  # Default
                
            if exit_candidates:
                exit_candidates.sort(key=lambda pos: pos[0] + pos[1], reverse=True)
                exit_point = exit_candidates[0]
            else:
                exit_point = (rows-1, cols-1)  # Default
                
            self.entry_exit_points = (entry, exit_point)
        
        # Debug visualization of final walls
        if self.debug_mode:
            border_debug = np.ones((rows * 50, cols * 50, 3), dtype=np.uint8) * 255
            for r in range(rows):
                for c in range(cols):
                    cell_y, cell_x = r * 50, c * 50
                    for wall_dir in range(4):
                        if wall_indicators[r][c][wall_dir] == 1:
                            if wall_dir == 0:  # Top
                                cv2.line(border_debug, (cell_x, cell_y), (cell_x + 50, cell_y), (0, 0, 0), 2)
                            elif wall_dir == 1:  # Right
                                cv2.line(border_debug, (cell_x + 50, cell_y), (cell_x + 50, cell_y + 50), (0, 0, 0), 2)
                            elif wall_dir == 2:  # Bottom
                                cv2.line(border_debug, (cell_x, cell_y + 50), (cell_x + 50, cell_y + 50), (0, 0, 0), 2)
                            elif wall_dir == 3:  # Left
                                cv2.line(border_debug, (cell_x, cell_y), (cell_x, cell_y + 50), (0, 0, 0), 2)
                    
            cv2.imwrite("debug/05b_border_handling.png", border_debug)
        
        return wall_indicators
    
    def _create_graph_from_wall_indicators(self, rows, cols, wall_indicators):
        """
        Create a graph representation from wall indicators.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            wall_indicators: 3D list [row][col][wall_direction]
            
        Returns:
            Graph representation
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
                # Check top (no top wall)
                if r > 0 and wall_indicators[r][c][0] == 0:
                    graph.connect_nodes((r, c), (r-1, c))
                
                # Check right (no right wall)
                if c < cols - 1 and wall_indicators[r][c][1] == 0:
                    graph.connect_nodes((r, c), (r, c+1))
                
                # Check bottom (no bottom wall)
                if r < rows - 1 and wall_indicators[r][c][2] == 0:
                    graph.connect_nodes((r, c), (r+1, c))
                
                # Check left (no left wall)
                if c > 0 and wall_indicators[r][c][3] == 0:
                    graph.connect_nodes((r, c), (r, c-1))
        
        return graph
    
    def _find_maze_entry_exit_points(self, rows, cols, wall_indicators):
        """
        Find the entry and exit points of the maze by identifying cells along the border
        that have openings to the outside.
        
        Args:
            rows: Number of rows in the maze
            cols: Number of columns in the maze
            wall_indicators: 3D list [row][col][wall_direction] where wall_direction is:
                            0: top, 1: right, 2: bottom, 3: left
                            and 1 indicates a wall, 0 indicates no wall
            
        Returns:
            Tuple of (entry_point, exit_point) where each is a tuple (row, col)
        """
        # Find all border cells with openings
        border_openings = []
        
        # Check top row
        for c in range(cols):
            if wall_indicators[0][c][0] == 0:  # No top wall
                border_openings.append((0, c, "top"))
        
        # Check right column
        for r in range(rows):
            if wall_indicators[r][cols-1][1] == 0:  # No right wall
                border_openings.append((r, cols-1, "right"))
        
        # Check bottom row
        for c in range(cols):
            if wall_indicators[rows-1][c][2] == 0:  # No bottom wall
                border_openings.append((rows-1, c, "bottom"))
        
        # Check left column
        for r in range(rows):
            if wall_indicators[r][0][3] == 0:  # No left wall
                border_openings.append((r, 0, "left"))
        
        # If we found no openings, the maze might be unusual or have errors
        if not border_openings:
            logging.warning("No border openings found, using corners as entry/exit")
            return (0, 0), (rows-1, cols-1)
        
        # If we found exactly one opening, the maze might be unusual
        if len(border_openings) == 1:
            logging.warning(f"Found only one border opening, using opposite corner for exit")
            entry_point = (border_openings[0][0], border_openings[0][1])
            # For exit, use the opposite corner
            exit_point = (rows-1 if entry_point[0] == 0 else 0, cols-1 if entry_point[1] == 0 else 0)
            return entry_point, exit_point
        
        # If we found more than two openings, prioritize based on position
        if len(border_openings) > 2:
            logging.warning(f"Found {len(border_openings)} border openings, expected 2")
            
        # Calculate "distance" from corners to determine entry vs exit
        # Usually entry is closer to top-left, exit closer to bottom-right
        from_top_left = [(opening[0] + opening[1], opening) for opening in border_openings]
        from_top_left.sort()  # Sort by distance from top-left
        
        # The one closest to the top-left is the entry
        entry_opening = from_top_left[0][1]
        
        # The one furthest from the top-left is the exit (if we have at least 2 openings)
        if len(from_top_left) > 1:
            exit_opening = from_top_left[-1][1]
        else:
            # Fallback if somehow we only have one opening at this point
            exit_opening = entry_opening
            
        # Optional: add debug visualization if self.debug_mode is True
        if self.debug_mode:
            debug_img = np.ones((rows * 50, cols * 50, 3), dtype=np.uint8) * 255
            
            # Draw the maze grid
            for r in range(rows):
                for c in range(cols):
                    cell_x, cell_y = c * 50, r * 50
                    # Draw walls
                    for wall_dir in range(4):
                        if wall_indicators[r][c][wall_dir] == 1:
                            if wall_dir == 0:  # Top
                                cv2.line(debug_img, (cell_x, cell_y), (cell_x + 50, cell_y), (0, 0, 0), 2)
                            elif wall_dir == 1:  # Right
                                cv2.line(debug_img, (cell_x + 50, cell_y), (cell_x + 50, cell_y + 50), (0, 0, 0), 2)
                            elif wall_dir == 2:  # Bottom
                                cv2.line(debug_img, (cell_x, cell_y + 50), (cell_x + 50, cell_y + 50), (0, 0, 0), 2)
                            elif wall_dir == 3:  # Left
                                cv2.line(debug_img, (cell_x, cell_y), (cell_x, cell_y + 50), (0, 0, 0), 2)
            
            # Highlight entry and exit
            entry_x, entry_y = entry_opening[1] * 50 + 25, entry_opening[0] * 50 + 25
            exit_x, exit_y = exit_opening[1] * 50 + 25, exit_opening[0] * 50 + 25
            cv2.circle(debug_img, (entry_x, entry_y), 15, (0, 255, 0), -1)
            cv2.circle(debug_img, (exit_x, exit_y), 15, (0, 0, 255), -1)
            cv2.putText(debug_img, "Entry", (entry_x - 20, entry_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(debug_img, "Exit", (exit_x - 20, exit_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            cv2.imwrite("debug/08_entry_exit_points.png", debug_img)
            
        # Return the positions (row, col)
        return (entry_opening[0], entry_opening[1]), (exit_opening[0], exit_opening[1])
    
    def _visualize_walls(self, wall_indicators, rows, cols):
        """
        Create a visualization of the detected walls.
        
        Args:
            wall_indicators: 3D list of wall indicators
            rows: Number of rows
            cols: Number of columns
        """
        # Create a debug grid image
        viz_cell_size = 50  # Size for visualization
        debug_grid = np.ones((rows * viz_cell_size, cols * viz_cell_size, 3), dtype=np.uint8) * 255
        
        for r in range(rows):
            for c in range(cols):
                cell_walls = wall_indicators[r][c]
                
                cell_y, cell_x = r * viz_cell_size, c * viz_cell_size
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
                cv2.putText(debug_grid, f"{r},{c}", (cell_x + 15, cell_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        cv2.imwrite("debug/05_wall_detection.png", debug_grid)
    
    def _visualize_maze_graph(self, graph, rows, cols):
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