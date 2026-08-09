import numpy as np
import networkx as nx


# ==========================
# iThor Functions
# ==========================

def get_distance_to_source(controller, sourcePos):
    """Calculates the 2D Euclidean distance (ground plane) between the agent and a source.

    Parameters
    ----------
    controller : ai2thor.controller.Controller
        The AI2-THOR controller instance.
    sourcePos : np.ndarray or list-like
        The 3D coordinates [x, y, z] of the source position.

    Returns
    -------
    float
        The Euclidean distance between the agent's (x, z) and source's (x, z).
    """
    agent_pos = controller.last_event.metadata["agent"]["position"]
    robot_pos = np.array([agent_pos["x"], agent_pos["z"]])
    source_pos = np.array([sourcePos[0], sourcePos[2]])
    return np.linalg.norm(robot_pos - source_pos)


def get_iThor_object_centers(objects, target_names):
    """Filters AI2-THOR objects by name and returns their 3D center coordinates.

    Matches objects whose name either exactly matches a name in `target_names`
    or starts with a target name followed by an underscore (e.g., 'Target_123').
    It prioritizes the center from the 'axisAlignedBoundingBox' if available,
    otherwise uses the 'position' field.

    Parameters
    ----------
    objects : list[dict]
        A list of object metadata dictionaries from `controller.last_event.metadata["objects"]`.
    target_names : list[str]
        A list of base object names to search for (e.g., ["Mug", "Apple"]).

    Returns
    -------
    np.ndarray
        A NumPy array of shape (n, 3) containing the [x, y, z] coordinates
        of the centers of the matching objects found. Returns an empty array if
        no matching objects are found or if center coordinates are unavailable.
    """
    centers = []
    
    for obj in objects:
        name = obj.get("name", "")
        # This check is more robust: it looks for an exact match OR a match with an underscore suffix.
        if any(name == target or name.startswith(target + '_') for target in target_names):
            center = obj.get("axisAlignedBoundingBox", {}).get("center", obj.get("position"))
            
            if center and all(k in center for k in ["x", "y", "z"]):
                centers.append([center["x"], center["y"], center["z"]])
            else:
                print(f"Center coordinates not available for object: {name}")
                
    return np.array(centers)


# ==========================
# Mapping Functions
# ==========================

def parse_position_string(pos_str):
    """Converts a position string "x, y, z" into a NumPy array.

    Parameters
    ----------
    pos_str : str
        A string containing three comma-separated float values (e.g., "1.0, 0.9, -2.5").

    Returns
    -------
    np.ndarray
        A NumPy array of shape (3,) containing the [x, y, z] coordinates as floats.
    """
    # Expecting a string like "x, y, z"
    return np.array([float(val.strip()) for val in pos_str.split(',')])


def world_to_grid(x, z, x_points, z_points):
    """Converts world coordinates (x, z) to the nearest grid cell indices.

    Finds the indices corresponding to the closest values in the `x_points`
    and `z_points` arrays.

    Parameters
    ----------
    x : float
        The world x-coordinate.
    z : float
        The world z-coordinate.
    x_points : np.ndarray
        A 1D array of the x-coordinates defining the grid columns.
    z_points : np.ndarray
        A 1D array of the z-coordinates defining the grid rows.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (2,) containing the [row_index, column_index].
    """
    col = (np.abs(x_points - x)).argmin()
    row = (np.abs(z_points - z)).argmin()
    return np.array([row, col])


def grid_to_world(pos, x_points, z_points):
    """Converts grid cell indices [row, col] back to world coordinates (x, z).

    Parameters
    ----------
    pos : np.ndarray or list-like
        The grid cell indices [row_index, column_index].
    x_points : np.ndarray
        A 1D array of the x-coordinates defining the grid columns.
    z_points : np.ndarray
        A 1D array of the z-coordinates defining the grid rows.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (2,) containing the world [x, z] coordinates
        corresponding to the center of the grid cell.
    """
    row, col = pos
    return np.array([x_points[col], z_points[row]])


# ==========================
# Nav Functions
# ==========================

def create_graph_from_positions(positions, threshold=0.3):
    """Creates a NetworkX graph connecting nearby reachable positions.

    Nodes represent reachable positions, and edges connect positions within
    the specified Euclidean distance `threshold`. Edge weights store the distance.

    Parameters
    ----------
    positions : list[dict]
        List of reachable position dictionaries, each with keys 'x', 'y', 'z'.
        Obtained from controller.step(action="GetReachablePositions").
    threshold : float, optional
        Maximum Euclidean distance between two positions to be considered connected
        by an edge. Defaults to 0.3.

    Returns
    -------
    networkx.Graph
        An undirected graph where nodes are indices corresponding to the input `positions` list,
        node attribute 'pos' stores the (x, y, z) tuple, and edges represent connectivity
        with 'weight' attribute storing the distance.
    """
    G = nx.Graph()
    if not positions: # Handle empty list
        return G

    num_positions = len(positions)

    # Add nodes with position attributes
    for i, pos in enumerate(positions):
        # Ensure pos is a dict with 'x', 'y', 'z' before accessing
        if isinstance(pos, dict) and all(k in pos for k in ['x', 'y', 'z']):
             G.add_node(i, pos=(pos['x'], pos['y'], pos['z']))
        else:
             print(f"Warning: Invalid position format at index {i} in create_graph_from_positions: {pos}")

    for i in range(num_positions):
        # Skip if node i wasn't added due to bad format
        if i not in G: continue
        p1 = np.array(G.nodes[i]['pos'])
        for j in range(i + 1, num_positions):
            # Skip if node j wasn't added
            if j not in G: continue
            p2 = np.array(G.nodes[j]['pos'])
            dist = np.linalg.norm(p1 - p2)
            if dist <= threshold:
                G.add_edge(i, j, weight=dist) # Store distance as edge weight
    return G

def find_nearest_node(graph, position):
    """Finds the node in the graph closest to a given 3D world position.

    Iterates through all nodes in the graph, calculates the Euclidean distance
    between the node's position and the target `position`, and returns the index
    of the node with the minimum distance.

    Parameters
    ----------
    graph : networkx.Graph
        The graph created by `create_graph_from_positions`. Nodes must have a 'pos'
        attribute containing (x, y, z) coordinates.
    position : tuple, list, np.ndarray, or dict
        The target 3D position [x, y, z] to find the nearest node to.
        If a dict, expects keys 'x', 'y', 'z'.

    Returns
    -------
    tuple[int or None, float]
        A tuple containing:
        - The index (int) of the nearest node in the graph. Returns None if the graph is empty.
        - The Euclidean distance (float) between the target position and the nearest node.
          Returns float('inf') if the graph is empty.
    """
    if not graph: # Handle empty graph
        return None, float('inf')

    # Convert target position to a NumPy array for calculations
    if isinstance(position, dict):
        # Ensure dict has required keys
        if all(k in position for k in ['x', 'y', 'z']):
             target_pos_np = np.array([position['x'], position['y'], position['z']])
        else:
             print(f"Warning: Invalid position dictionary format in find_nearest_node: {position}")
             return None, float('inf') # Cannot compare
    else:
        try:
             target_pos_np = np.array(position, dtype=float)
             if target_pos_np.shape != (3,): raise ValueError("Position must have 3 elements")
        except (ValueError, TypeError) as e:
             print(f"Warning: Could not convert position {position} to valid 3D NumPy array: {e}")
             return None, float('inf')


    min_dist_sq = float('inf') # Use squared distance to avoid sqrt initially
    nearest_node_idx = None

    for node_idx, data in graph.nodes(data=True):
        node_pos = data.get('pos')
        if node_pos is None:
             print(f"Warning: Node {node_idx} in graph is missing 'pos' attribute.")
             continue
        try:
             node_pos_np = np.array(node_pos, dtype=float)
             if node_pos_np.shape != (3,): raise ValueError("Node pos must have 3 elements")
             dist_sq = np.sum((node_pos_np - target_pos_np)**2) # Squared Euclidean distance
             if dist_sq < min_dist_sq:
                 min_dist_sq = dist_sq
                 nearest_node_idx = node_idx
        except (ValueError, TypeError) as e:
             print(f"Warning: Could not process position {node_pos} for node {node_idx}: {e}")
             continue

    min_dist = np.sqrt(min_dist_sq) if nearest_node_idx is not None else float('inf')
    return nearest_node_idx, min_dist