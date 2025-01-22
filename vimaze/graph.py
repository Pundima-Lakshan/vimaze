
class Node:
    def __init__(self, value, row, col):
        self.value = value
        self.row = row
        self.col = col
        self.edges = {
            'north': None,
            'west': None,
            'south': None,
            'east': None
        }

    def set_edge(self, direction, node):
        if direction in self.edges:
            self.edges[direction] = node

    def get_edge(self, direction):
        if direction in self.edges:
            return self.edges[direction]
        return None


class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.value] = node

    def get_node(self, value):
        return self.nodes.get(value)

    def connect_nodes(self, u_node_value, v_node_value):
        u_node = self.get_node(u_node_value)
        v_node = self.get_node(v_node_value)

        if not v_node or not u_node:
            return ValueError("Both nodes must exist in the graph", u_node, v_node)

        if not Graph.valid_4_neighbors(u_node, v_node):
            return ValueError("Nodes must be valid 4 neighbors", u_node, v_node)

        if u_node.row == v_node.row:
            if u_node.col < v_node.col:
                u_node.set_edge("east", v_node)
                v_node.set_edge("west", u_node)
            else:
                u_node.set_edge("west", v_node)
                v_node.set_edge("east", u_node)
        elif u_node.col == v_node.col:
            if u_node.row < v_node.row:
                u_node.set_edge("south", v_node)
                v_node.set_edge("north", u_node)
            else:
                u_node.set_edge("north", v_node)
                v_node.set_edge("south", u_node)

    @staticmethod
    def valid_4_neighbors(u_node, v_node):
        if u_node.row == v_node.row and abs(u_node.col - v_node.col) == 1:
            return True
        if u_node.col == v_node.col and abs(u_node.row - v_node.row) == 1:
            return True
        return False

    @staticmethod
    def get_neighbors(node):
        neighbors = []
        for direction, edge in node.edges.items():
            if edge:
                neighbors.append((direction, edge))
        return neighbors
