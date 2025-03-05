## **Breadth-First Search (BFS) Solver**

The **BFS Solver** finds the shortest path in a maze using the **Breadth-First Search (BFS) algorithm**. It explores all
possible paths level by level, ensuring an **optimal** solution in unweighted graphs.

### **Pseudocode**

```plaintext
1. Initialize:
   - Create an empty queue and enqueue the start node.
   - Mark the start node as visited.
   - Store parent nodes for path reconstruction.

2. While the queue is not empty:
   - Dequeue a node and process it.
   - If the node is the goal, terminate the search.
   - For each unvisited neighbor:
     - Mark it as visited.
     - Store its parent node.
     - Enqueue it.

3. Reconstruct the path by backtracking from the goal node to the start node.
4. Return the path.
```

### **Key Properties of BFS**

- **Completeness:** BFS is **guaranteed to find a solution** if one exists.
- **Optimality:** BFS always finds the **shortest path** in an unweighted graph.
- **Time Complexity:** \(O(V + E)\), where \(V\) is the number of nodes and \(E\) is the number of edges.
- **Space Complexity:** \(O(V)\), since it stores visited nodes and path mappings.

## **Dijkstra’s Algorithm Solver**

The **Dijkstra Solver** finds the shortest path in a weighted graph using **Dijkstra’s algorithm**. It ensures the most
efficient route by prioritizing nodes with the smallest known distance.

### **Pseudocode**

```plaintext
1. Initialize:
   - Create a priority queue (min-heap).
   - Create a distance map with all nodes set to infinity.
   - Set the start node’s distance to 0.
   - Store parent nodes for path reconstruction.

2. Add the start node to the priority queue with distance 0.

3. While the queue is not empty:
   - Extract the node with the smallest distance.
   - For each neighbor:
     - Calculate its new tentative distance.
     - If the new distance is smaller than the stored one:
       - Update the neighbor's distance.
       - Store its parent node.
       - Push the neighbor into the priority queue.

4. Reconstruct the path by backtracking from the goal node to the start node.
5. Return the shortest path.
```

### **Key Properties of Dijkstra’s Algorithm**

- **Completeness:** Always finds a solution if one exists.
- **Optimality:** Always finds the shortest path.
- **Time Complexity:** \(O((V + E) \log V)\) using a priority queue.
- **Space Complexity:** \(O(V + E)\), storing distances and priority queue.

## **Depth-First Search (DFS) Solver**

The **DFS Solver** explores paths in a maze using the **Depth-First Search (DFS) algorithm**. It follows one path as
deep as possible before backtracking.

### **Pseudocode**

```plaintext
1. Initialize:
   - Create an empty stack and push the start node.
   - Mark the start node as visited.
   - Store parent nodes for path reconstruction.

2. While the stack is not empty:
   - Peek at the top node.
   - Get unvisited neighbors of the node.
   - If no unvisited neighbors:
     - Mark the node as fully visited and pop it from the stack.
   - Else:
     - Pick an unvisited neighbor.
     - Mark it as visited and push it onto the stack.
     - Store its parent node.
     - If it is the goal node, terminate the search.

3. Reconstruct the path by backtracking from the goal node to the start node.
4. Return the path.
```

### **Key Properties of DFS**

- **Completeness:** DFS is **not always complete** in infinite graphs but is complete in finite ones.
- **Optimality:** DFS **does not guarantee** the shortest path.
- **Time Complexity:** \(O(V + E)\), where \(V\) is the number of nodes and \(E\) is the number of edges.
- **Space Complexity:**
    - **O(V) for recursive DFS**, due to the recursion stack.
    - **O(E) for iterative DFS**, using an explicit stack.