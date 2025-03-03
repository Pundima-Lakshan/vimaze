class Node:
    def __init__(self, position: tuple[int, int]):
        self.position = position
        self.neighbors: list['Node'] = []
        self.name = _get_node_name((position[0], position[1]))
    
    def add_neighbour(self, neighbour: 'Node'):
        self.neighbors.append(neighbour)
        
    def is_neighbour(self, node: 'Node'):
        for neighbour in self.neighbors:
            if neighbour.name is node.name:
                return True
        return False

class Graph:
    def __init__(self, node_positions: list[tuple[int, int]] = None):
        self.nodes = {}
        
        if node_positions is None:
            self.node_count = 0
            return
        
        self.node_count = len(node_positions)

        for node_pos in node_positions:
            new_node = Node(node_pos)
            self.nodes[new_node.name] = new_node

    def add_node(self, node_pos: tuple[int, int]):
        new_node = Node(node_pos)
        self.nodes[new_node.name] = new_node
        self.node_count += 1
        
    def connect_nodes(self, node_u_pos: tuple[int, int], node_v_pos: tuple[int, int]):
        node_u_name = _get_node_name(node_u_pos)
        node_v_name = _get_node_name(node_v_pos)

        self.nodes[node_u_name].add_neighbour(self.nodes[node_v_name])
        self.nodes[node_v_name].add_neighbour(self.nodes[node_u_name])

    def get_neighbors(self, node_pos: tuple[int, int]):
        node_name = _get_node_name(node_pos)
        return self.nodes[node_name].neighbors

    def get_node(self, node_pos: tuple[int, int]):
        node_name = _get_node_name(node_pos)
        return self.nodes[node_name]

    def display(self):
        for node in self.nodes.values():
            print(f"{node.name}: {[neighbor.name for neighbor in node.neighbors]}")
            
    @staticmethod       
    def get_node_name(pos: tuple[int, int]):
        return _get_node_name(pos)

def _get_node_name(pos: tuple[int, int]):
    return f"{pos[0]},{pos[1]}"