import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def detect_wall_angle(binary, side='left'):
    """
    Detect the angle of a wall on either the left or right side of the maze.
    
    Args:
        binary (numpy.ndarray): Binary image of the maze
        side (str): 'left' or 'right' to indicate which side to detect
        
    Returns:
        tuple: (angle_deg, wall_points, (x_start, y_start, x_end, y_end))
    """
    h, w = binary.shape
    search_width = w // 4  # Search in the 1/4th of the image on each side
    wall_points = []

    # Skip pixels at the border to avoid artifacts
    border_skip = 5

    if side == 'left':
        # Scan from left to right to find the first black pixel in each row
        for y in range(border_skip, h - border_skip):
            for x in range(border_skip, search_width):
                if binary[y, x] > 0:  # Found a wall pixel
                    wall_points.append((x, y))
                    break
    else:  # right side
        # Scan from right to left to find the first black pixel in each row
        for y in range(border_skip, h - border_skip):
            for x in range(w - 1 - border_skip, w - search_width, -1):
                if binary[y, x] > 0:  # Found a wall pixel
                    wall_points.append((x, y))
                    break

    # Sort points by y-coordinate and filter outliers
    wall_points.sort(key=lambda p: p[1])

    # Calculate angle if we have enough points
    if len(wall_points) > h // 5:  # At least 20% of image height
        # Extract x and y coordinates
        x_coords = [p[0] for p in wall_points]
        y_coords = [p[1] for p in wall_points]

        # Use linear regression to fit a line
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_coords, x_coords)

        # Calculate angle in degrees
        angle_rad = np.arctan(slope)
        angle_deg = np.degrees(angle_rad)

        # Calculate endpoints of the line for visualization
        y_start, y_end = min(y_coords), max(y_coords)
        x_start = int(slope * y_start + intercept)
        x_end = int(slope * y_end + intercept)

        return angle_deg, wall_points, (x_start, y_start, x_end, y_end)
    else:
        return 0, wall_points, (0, 0, 0, 0)


def preprocess_maze(image_path, output_path=None):
    """
    Preprocess a maze image by detecting and correcting various types of skew
    based on angles detected from both left and right walls.
    
    Args:
        image_path (str): Path to the input maze image
        output_path (str, optional): Path to save the processed image
        
    Returns:
        numpy.ndarray: The processed binary maze image
    """
    # Read the image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Basic thresholding - assuming walls are dark (black walls on white background)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Clean image borders to remove artifacts
    border_size = 5
    # Create a mask of the same size as the binary image
    mask = np.ones_like(binary)
    # Set the border pixels to zero in the mask
    mask[0:border_size, :] = 0  # Top border
    mask[-border_size:, :] = 0  # Bottom border
    mask[:, 0:border_size] = 0  # Left border
    mask[:, -border_size:] = 0  # Right border
    # Apply the mask to the binary image
    binary = binary * mask

    # Detect angles of left and right walls
    left_angle, left_points, left_line = detect_wall_angle(binary, 'left')
    right_angle, right_points, right_line = detect_wall_angle(binary, 'right')

    print(f"Left wall angle: {left_angle:.2f} degrees")
    print(f"Right wall angle: {right_angle:.2f} degrees")

    # Get image dimensions
    h, w = binary.shape

    # Determine the type of skew and set up appropriate correction
    if abs(left_angle) < 0.5 and abs(right_angle) < 0.5:
        # Both walls are nearly vertical - no significant skew
        print("No significant skew detected.")
        corrected = binary
    else:
        # Calculate how much the top and bottom need to be adjusted
        # based on the angles of both walls

        # For left wall
        left_slope = np.tan(np.radians(left_angle))
        left_shift_top = 0
        left_shift_bottom = int(h * left_slope)

        # For right wall
        right_slope = np.tan(np.radians(right_angle))
        right_shift_top = 0
        right_shift_bottom = int(h * right_slope)

        # Determine if it's a parallelogram or trapezoid skew
        parallel_skew = abs(left_angle - right_angle) < 5.0  # Within 5 degrees

        if parallel_skew:
            # Parallelogram skew (left and right are roughly parallel)
            print("Detected parallelogram skew.")

            # Use average angle for correction
            avg_angle = (left_angle + right_angle) / 2
            avg_slope = np.tan(np.radians(avg_angle))
            shift = int(h * avg_slope)

            # Source points (current image corners)
            src_points = np.array([
                [0, 0],  # top-left
                [w - 1, 0],  # top-right
                [w - 1, h - 1],  # bottom-right
                [0, h - 1]  # bottom-left
            ], dtype=np.float32)

            # Destination points (corners after removing skew)
            if shift > 0:
                # Shift top to the right
                dst_points = np.array([
                    [shift, 0],  # top-left
                    [w - 1, 0],  # top-right
                    [w - 1, h - 1],  # bottom-right
                    [0, h - 1]  # bottom-left
                ], dtype=np.float32)
            else:
                # Shift bottom to the right
                dst_points = np.array([
                    [0, 0],  # top-left
                    [w - 1, 0],  # top-right
                    [w - 1 + shift, h - 1],  # bottom-right
                    [-shift, h - 1]  # bottom-left
                ], dtype=np.float32)
        else:
            # Trapezoid skew (perspective distortion)
            print("Detected trapezoid skew.")

            # Source points (current image corners)
            src_points = np.array([
                [0, 0],  # top-left
                [w - 1, 0],  # top-right
                [w - 1, h - 1],  # bottom-right
                [0, h - 1]  # bottom-left
            ], dtype=np.float32)

            # Destination points (corners after removing keystone effect)
            # We use the detected shifts for left and right sides
            dst_points = np.array([
                [max(0, -left_shift_top), 0],  # top-left
                [w - 1 + min(0, -right_shift_top), 0],  # top-right
                [w - 1 + min(0, -right_shift_bottom), h - 1],  # bottom-right
                [max(0, -left_shift_bottom), h - 1]  # bottom-left
            ], dtype=np.float32)

        # Calculate and apply the perspective transform
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Determine output image size
        max_width = w + max(abs(left_shift_bottom), abs(right_shift_bottom))

        # Apply the perspective transformation
        corrected = cv2.warpPerspective(binary, transform_matrix, (max_width, h))

    # Filter the binary image to remove any single-pixel noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(corrected, cv2.MORPH_OPEN, kernel)

    # Apply blur to reduce pixelation
    blurred = cv2.GaussianBlur(cleaned, (3, 3), 0)

    # Sharpen the image
    blur_amount = cv2.GaussianBlur(blurred, (9, 9), 2)
    sharpened = cv2.addWeighted(blurred, 1.5, blur_amount, -0.5, 0)

    # Final binary image
    _, final = cv2.threshold(sharpened, 128, 255, cv2.THRESH_BINARY)

    # Invert the image to get black walls on white background
    final = cv2.bitwise_not(final)

    # Clean the borders of the final image
    border_size = 10
    h_final, w_final = final.shape
    final[0:border_size, :] = 255  # Top border (white after inversion)
    final[-border_size:, :] = 255  # Bottom border (white after inversion)
    final[:, 0:border_size] = 255  # Left border (white after inversion)
    final[:, -border_size:] = 255  # Right border (white after inversion)

    # Save the output if path is provided
    if output_path:
        cv2.imwrite(output_path, final)
        print(f"Processed maze saved to {output_path}")

    wall_info = {
        'left_angle': left_angle,
        'right_angle': right_angle,
        'left_points': left_points,
        'right_points': right_points,
        'left_line': left_line,
        'right_line': right_line,
        'parallel_skew': parallel_skew if abs(left_angle) > 0.5 or abs(right_angle) > 0.5 else None
    }

    return final, wall_info


def visualize_processing_steps(image_path):
    """
    Visualize each step of the maze preprocessing algorithm.
    
    Args:
        image_path (str): Path to the input maze image
    """
    # Read the image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Basic thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Clean the borders for visualization
    border_size = 5
    binary_viz = binary.copy()
    binary_viz[0:border_size, :] = 0  # Top border
    binary_viz[-border_size:, :] = 0  # Bottom border
    binary_viz[:, 0:border_size] = 0  # Left border
    binary_viz[:, -border_size:] = 0  # Right border

    # Run preprocessing to get processed image and wall info
    final, wall_info = preprocess_maze(image_path)

    # Create visualization of the wall detection
    wall_detection = cv2.cvtColor(binary_viz.copy(), cv2.COLOR_GRAY2BGR)

    # Draw left wall points
    for point in wall_info['left_points']:
        cv2.circle(wall_detection, point, 2, (0, 0, 255), -1)

    # Draw right wall points
    for point in wall_info['right_points']:
        cv2.circle(wall_detection, point, 2, (255, 0, 0), -1)

    # Draw the fitted lines if we have valid angles
    if wall_info['left_angle'] != 0:
        x1, y1, x2, y2 = wall_info['left_line']
        cv2.line(wall_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if wall_info['right_angle'] != 0:
        x1, y1, x2, y2 = wall_info['right_line']
        cv2.line(wall_detection, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Add text showing the detected angles
    cv2.putText(wall_detection, f"Left: {wall_info['left_angle']:.2f}°", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(wall_detection, f"Right: {wall_info['right_angle']:.2f}°", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Add text showing the skew type if detected
    if wall_info['parallel_skew'] is not None:
        skew_type = "Parallelogram" if wall_info['parallel_skew'] else "Trapezoid"
        cv2.putText(wall_detection, f"Skew: {skew_type}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Display all steps
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title('1. Original Image')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 2)
    plt.title('2. Binary')
    plt.imshow(binary, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('3. Wall Detection')
    plt.imshow(cv2.cvtColor(wall_detection, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 4)
    plt.title('4. Final Deskewed Result')
    plt.imshow(final, cmap='gray')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Preprocess the maze image
    input_path = "premaze.jpg"  # Change to your maze image path
    output_path = "processed_maze.jpg"

    # Process and save the result
    processed_maze, wall_info = preprocess_maze(input_path, output_path)

    # Visualize the processing steps
    visualize_processing_steps(input_path)
