#!/usr/bin/env python
"""
Test script for maze image processing.
"""
import os
import sys
import argparse
import cv2
import numpy as np

def test_standard_processor(image_path, debug=True, invert_binary=False, wall_threshold=127):
    """Test the standard MazeImageProcessor."""
    from vimaze.maze_image_processor import MazeImageProcessor
    
    # Create debug directory
    if debug and not os.path.exists("debug"):
        os.makedirs("debug")
    
    # Initialize processor
    processor = MazeImageProcessor()
    processor.debug_mode = debug
    
    # Configure parameters
    processor.wall_threshold = wall_threshold
    processor.invert_binary = invert_binary
    
    try:
        print(f"Processing maze image with standard processor: {image_path}")
        print(f"Parameters: invert_binary={invert_binary}, wall_threshold={wall_threshold}, debug={debug}")
        
        graph, rows, cols, start, end = processor.process_image(image_path)
        print(f"Success! Maze dimensions: {rows}x{cols}")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {str(e)}")
        return False

def test_simple_processor(image_path, debug=True, wall_threshold=50, cell_size=20):
    """Test the simple black wall maze processor."""
    from vimaze.simple_maze_processor import SimpleMazeProcessor
    
    # Create debug directory
    if debug and not os.path.exists("simple_debug"):
        os.makedirs("simple_debug")
    
    # Initialize processor
    processor = SimpleMazeProcessor()
    processor.debug_mode = debug
    
    # Configure parameters
    processor.cell_size = cell_size
    processor.wall_threshold = wall_threshold
    
    try:
        print(f"Processing maze image with simple processor: {image_path}")
        print(f"Parameters: wall_threshold={wall_threshold}, cell_size={cell_size}, debug={debug}")
        
        graph, rows, cols = processor.process_image(image_path)
        print(f"Success! Maze dimensions: {rows}x{cols}")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test maze image processing.")
    parser.add_argument("image_path", help="Path to the maze image file")
    parser.add_argument("--processor", choices=["standard", "simple", "both"], default="both",
                        help="Which processor to use (default: both)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--invert", action="store_true", help="Invert binary image (for standard processor)")
    parser.add_argument("--wall-threshold", type=int, default=127, 
                        help="Wall threshold (0-255, default: 127 for standard, 50 for simple)")
    parser.add_argument("--cell-size", type=int, default=20,
                        help="Cell size for simple processor (default: 20)")
    
    args = parser.parse_args()
    
    if args.processor in ["standard", "both"]:
        print("\n=== Testing Standard Processor ===")
        test_standard_processor(args.image_path, args.debug, args.invert, args.wall_threshold)
    
    if args.processor in ["simple", "both"]:
        print("\n=== Testing Simple Processor ===")
        simple_threshold = 50 if args.wall_threshold == 127 else args.wall_threshold
        test_simple_processor(args.image_path, args.debug, simple_threshold, args.cell_size)

if __name__ == "__main__":
    main()