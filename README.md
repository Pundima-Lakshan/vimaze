# Vimaze

## Overview

**Vimaze** is a graphical and image-based maze generator and solver. It allows users to generate mazes, solve them using various algorithms, and even process maze images to extract and solve real-world mazes. The application features a modern, configurable UI and supports both GUI and command-line workflows.

---

## Features

- **Maze Generation**: Generate mazes using Prim’s algorithm (with more algorithms planned).
- **Maze Solving**: Solve mazes using BFS, DFS, Dijkstra, and A* algorithms.
- **Image Processing**: Import maze images, detect and extract the maze structure, and solve them.
- **Configurable UI**: Adjust grid size, start/end positions, animation speed, and more.
- **Animation Support**: Visualize maze generation and solving step-by-step.
- **Manual Corner Selection**: Manually select maze corners for better image processing.
- **Debugging Tools**: Save debug images during processing for troubleshooting.

---

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Pundima-Lakshan/labyrinth.git
   cd vimaze
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   (Dependencies: opencv-python, numpy, customtkinter, Pillow)

---

## Usage

### GUI Application
Run the graphical application:
```sh
python __main__.py
```

### Command-Line Image Processing
Process a maze image and test the image processor:
```sh
python vimaze/image_processing/test_maze_processor.py <image_path> [--processor standard|simple|both] [--debug] [--invert] [--wall-threshold N] [--cell-size N]
```
Example:
```sh
python vimaze/image_processing/test_maze_processor.py maze.png --debug --processor standard
```

---

## Configuration

The application is highly configurable via the `vimaze/configs.py` file. Key options include:
- **Window size and title**
- **Grid layout** (rows, columns)
- **Frame and canvas settings**
- **Default algorithms for generation and solving**
- **Image processing parameters** (wall threshold, invert binary, cell size)

You can also adjust these options via the GUI controls (dropdowns, sliders, and input fields).

---

## Algorithms

### Maze Generation
- **Prim’s Algorithm**: Randomized, ensures a fully connected maze with no cycles. [See details in `vimaze/generators/README.md`]

### Maze Solving
- **Breadth-First Search (BFS)**: Finds the shortest path in unweighted mazes.
- **Depth-First Search (DFS)**: Explores as deep as possible before backtracking (not guaranteed shortest path).
- **Dijkstra’s Algorithm**: Finds the shortest path in weighted mazes.
- **A* Search**: Uses heuristics for efficient shortest-path finding.

See `vimaze/solvers/README.md` for pseudocode and properties of each algorithm.

---

## Image Processing

- **MazeImageProcessor**: Extracts maze structure from images using adaptive thresholding and area-based wall detection.
- **Manual Corner Input**: For images where automatic detection fails, you can manually select maze corners.
- **Debug Mode**: Saves intermediate images to a `debug/` folder for troubleshooting.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug fixes, new features, or documentation improvements.

---

## License

This project is licensed under the MIT License.

