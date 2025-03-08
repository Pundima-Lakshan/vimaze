import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_maze_lines(image_path, output_path=None, line_thickness=2, smoothing_level=1):
    """
    Enhance maze lines by smoothing and adjusting thickness.
    
    Args:
        image_path (str): Path to the input maze image
        output_path (str, optional): Path to save the processed image
        line_thickness (int): Target thickness for maze lines (1-3)
        smoothing_level (int): Level of smoothing to apply (0-2)
        
    Returns:
        numpy.ndarray: The enhanced maze image
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Ensure binary image (black lines on white background)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Make sure the image has black lines on white background
    # If more than 50% of pixels are black, invert the image
    if np.sum(binary == 255) > (binary.size / 2):
        binary = cv2.bitwise_not(binary)
    
    # Step 1: Apply gentle blurring to reduce pixelation/jaggies
    if smoothing_level > 0:
        if smoothing_level == 1:
            # Mild smoothing
            smoothed = cv2.GaussianBlur(binary, (3, 3), 0.5)
        else:
            # Stronger smoothing
            smoothed = cv2.GaussianBlur(binary, (5, 5), 1.0)
        
        # Re-threshold to get back to binary
        _, smoothed = cv2.threshold(smoothed, 128, 255, cv2.THRESH_BINARY)
    else:
        smoothed = binary
    
    # Step 2: Find thin areas that need thickening
    # We'll use morphological operations to identify thin lines
    
    # Create the enhanced image based on target thickness
    if line_thickness == 1:
        # For thickness 1, just use the smoothed image
        enhanced = smoothed
    else:
        # For thickness > 1, apply selective thickening
        
        # Dilate the image to thicken lines
        kernel_size = line_thickness
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        enhanced = cv2.dilate(smoothed, kernel, iterations=1)
    
    # Invert back to get black lines on white background
    enhanced = cv2.bitwise_not(enhanced)
    
    # Clean borders if needed
    border_size = 5
    h, w = enhanced.shape
    enhanced[0:border_size, :] = 255  # Top border (white)
    enhanced[-border_size:, :] = 255  # Bottom border (white)
    enhanced[:, 0:border_size] = 255  # Left border (white)
    enhanced[:, -border_size:] = 255  # Right border (white)
    
    # Save the output if path is provided
    if output_path:
        cv2.imwrite(output_path, enhanced)
        print(f"Enhanced maze saved to {output_path}")
    
    return enhanced

def visualize_enhancement(image_path, line_thickness=2, smoothing_level=1):
    """
    Visualize the line enhancement process.
    
    Args:
        image_path (str): Path to the input maze image
        line_thickness (int): Target thickness for maze lines (1-3)
        smoothing_level (int): Level of smoothing to apply (0-2)
    """
    # Read the image
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Process the image with different parameters to show progression
    thickness_options = [1, 2, 3]
    smoothing_options = [0, 1, 2]
    
    # Create the grid of visualizations
    plt.figure(figsize=(12, 10))
    
    # Original image
    plt.subplot(3, 3, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    
    # Show different combinations
    for i, t in enumerate(thickness_options):
        for j, s in enumerate(smoothing_options):
            if i == 0 and j == 0:
                continue  # Skip, already showing original
            
            plt.subplot(3, 3, i*3 + j + 1)
            plt.title(f'Thickness={t}, Smooth={s}')
            
            # Process with current parameters
            enhanced = enhance_maze_lines(image_path, None, t, s)
            plt.imshow(enhanced, cmap='gray')
    
    # Recommended setting (based on input parameters)
    plt.subplot(3, 3, 9)
    plt.title(f'Recommended (T={line_thickness}, S={smoothing_level})')
    recommended = enhance_maze_lines(image_path, None, line_thickness, smoothing_level)
    plt.imshow(recommended, cmap='gray')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Enhance the maze image
    input_path = "processed_maze.jpg"  # Change to your maze image path
    output_path = "enhanced_maze.jpg"
    
    # Process with default parameters
    enhanced_maze = enhance_maze_lines(input_path, output_path, line_thickness=2, smoothing_level=1)
    
    # Visualize different enhancement options
    visualize_enhancement(input_path)