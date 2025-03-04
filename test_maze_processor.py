#!/usr/bin/env python
"""
Test script for maze image processing.
"""
import os
import sys
import argparse
import cv2
import numpy as np

def test_improved_processor(image_path, debug=True):
    """Test the improved MazeImageProcessor."""
    from vimaze.maze_image_processor import MazeImageProcessor
    
    # Create debug directory
    if debug and not os.path.exists("debug"):
        os.makedirs("debug")
    
    # Initialize processor
    processor = MazeImageProcessor()
    processor.debug_mode = debug
    
    # Configure parameters
    processor.wall_threshold = 127
    processor.invert_binary = False
    processor.line_min_length = 50
    processor.line_max_gap = 10
    processor.wall_thickness = 5
    
    try:
        print(f"Processing maze image with improved processor: {image_path}")
        graph, rows, cols = processor.process_image(image_path)
        print(f"Success! Maze dimensions: {rows}x{cols}")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {str(e)}")
        return False

def test_simple_processor(image_path, debug=True):
    """Test the simple black wall maze processor."""
    from vimaze.simple_maze_processor import SimpleMazeProcessor
    
    # Create debug directory
    if debug and not os.path.exists("debug"):
        os.makedirs("debug")
    
    # Initialize processor
    processor = SimpleMazeProcessor()
    processor.debug_mode = debug
    
    # Configure parameters
    processor.cell_size = 20
    processor.wall_threshold = 50
    
    try:
        print(f"Processing maze image with simple processor: {image_path}")
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
    parser.add_argument("--processor", choices=["improved", "simple", "both"], default="both",
                        help="Which processor to use (default: both)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.processor in ["improved", "both"]:
        print("\n=== Testing Improved Processor ===")
        test_improved_processor(args.image_path, args.debug)
    
    if args.processor in ["simple", "both"]:
        print("\n=== Testing Simple Processor ===")
        test_simple_processor(args.image_path, args.debug)

if __name__ == "__main__":
    main()
