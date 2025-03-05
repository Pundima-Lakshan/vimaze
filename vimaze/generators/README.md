## **Prim’s Algorithm for Maze Generation**

The **Prim’s Algorithm** is a randomized algorithm used for maze generation. It builds a spanning tree by progressively
adding frontier cells to the maze, ensuring a connected, non-cyclic path structure.

### **Pseudocode**

```plaintext
1. Initialize:
   - Set all cells with walls.
   - Choose a random start cell and mark it as visited.
   - Add its frontier cells (adjacent unvisited cells) to a frontier set.

2. While the frontier set is not empty:
   - Pick a random frontier cell and remove it from the set.
   - Select a random adjacent visited cell.
   - Remove the wall between them to create a passage.
   - Mark the frontier cell as visited.
   - Add its unvisited neighbors to the frontier set.

3. Repeat until all cells are visited.
4. The maze is fully generated.
```

### **Key Properties of Prim’s Algorithm**

- **Completeness:** Always generates a fully connected maze.
- **Optimality:** Ensures a spanning tree, avoiding loops.
- **Randomized Nature:** Produces unique mazes on each execution.
- **Bias:** Tends to create dense, twisty mazes with fewer long straight paths.
- **Time Complexity:** \(O(V + E)\) in the worst case.
- **Space Complexity:** \(O(V)\), storing visited cells and frontier set.