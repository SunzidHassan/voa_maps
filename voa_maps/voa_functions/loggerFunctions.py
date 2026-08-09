from matplotlib.patches import Circle
import networkx as nx
from shapely.geometry import Polygon
from skimage.morphology import closing, disk
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import numpy as np
import math

from voa_functions.olfactionFunctions import gaussian_plume

from voa_functions.utils import (
    world_to_grid
)

# ==========================
# Plotting Functions
# ==========================

def plot_detected_objects(itemDF, mask_closed, scene_bounds_tuple, save_path=None):
    """
    Generates a top-down map showing the reachable area and labels for
    objects detected by the vision branch.

    If save_path is None, it displays the plot. Otherwise, it saves to file.

    Parameters
    ----------
    itemDF : pd.DataFrame
        DataFrame from visionBranch, must have 'objectType' and 'Position' columns.
    mask_closed : np.ndarray
        The 2D boolean mask of reachable areas (H, W).
    scene_bounds_tuple : tuple
        Tuple of (min_x, max_x, min_z, max_z) for the scene.
    save_path : str, optional
        Path to save the generated plot image. If None, shows plot instead.
    """
    
    min_x, max_x, min_z, max_z = scene_bounds_tuple
    
    # Check if itemDF is empty. If so, don't bother plotting.
    if itemDF.empty:
        print("Skipping detected objects plot: itemDF is empty.")
        return
        
    print(f"Plotting {len(itemDF)} detected objects...")
    
    # Get the figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 12)) 

    # (A) Grey out non-reachable area
    grey = np.array([0.8, 0.8, 0.8, 1.0])
    
    if mask_closed.ndim == 2 and mask_closed.size > 0:
        ov = np.tile(grey, (mask_closed.shape[0], mask_closed.shape[1], 1))
        ov[..., 3] = (~mask_closed).astype(float) 
        ax.imshow(ov, extent=[min_x, max_x, min_z, max_z], origin='lower', zorder=2, alpha=0.5, aspect='auto')
    else:
         print("Warning: mask_closed is not valid. Skipping overlay.")

    # (B) Plot Object Labels from itemDF
    for _, row in itemDF.iterrows():
        name = row.get('objectType', 'N/A')
        pos_str = row.get('Position', None)
        
        if pos_str is None or not isinstance(pos_str, str):
            continue
            
        try:
            x, y, z = map(float, pos_str.split(','))
            ax.text(x, z, name, 
                     fontsize=8, 
                     ha='center', va='center',
                     zorder=3, color='black',
                     bbox=dict(facecolor='yellow', alpha=0.8, pad=0.1, boxstyle='round,pad=0.2'))
                     
        except (ValueError, TypeError, AttributeError):
            print(f"Could not parse position for {name}: {pos_str}")

    # (C) Final touches
    ax.set_xlabel('Robot X Position (m)')
    ax.set_ylabel('Robot Z Position (m)')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    try:
            # Use the figure object 'fig' to save
            fig.savefig(save_path, dpi=200) 
            print(f"Successfully saved detected objects map to {save_path}")
    except Exception as e:
            print(f"Error saving detected objects map: {e}")
    
    plt.close(fig)


def generate_heatmap(df, x_points, z_points, weight_key='goalSim', sigma=1, upsample=1):
    """Generates a heatmap based on weighted object locations from a DataFrame.

    Creates a 2D grid based on `x_points` and `z_points`. For each object in `df`,
    it adds the object's weight (from `df[weight_key]`, clamped >= 0) to the
    corresponding grid cell. The resulting map is optionally smoothed with a
    Gaussian filter, max-normalized to [0, 1], and optionally upsampled.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing object information. Must have a 'Position' column
        ("x, y, z" string) and the column specified by `weight_key`.
    x_points : np.ndarray
        1D array of x-coordinates defining the grid columns.
    z_points : np.ndarray
        1D array of z-coordinates defining the grid rows.
    weight_key : str, optional
        The column name in `df` to use for weighting each object's contribution
        to the heatmap. Defaults to 'goalSim'.
    sigma : float, optional
        Standard deviation for the Gaussian smoothing filter applied to the heatmap.
        Set to 0 or less to disable smoothing. Defaults to 1.0.
    upsample : int, optional
        Factor by which to upscale the heatmap resolution using linear interpolation.
        Set to 1 or less to disable upsampling. Defaults to 1.

    Returns
    -------
    np.ndarray
        The generated 2D heatmap as a NumPy array, normalized to the range [0, 1].
        Dimensions will be (len(z_points)*upsample, len(x_points)*upsample).
    """
    
    H, W = len(z_points), len(x_points)
    heatmap = np.zeros((H, W), dtype=float)

    if df.empty:
        print("Warning: DataFrame is empty in generate_heatmap. Returning zero map.")
        # Return zero map of the correct (potentially upsampled) size
        if upsample > 1:
            return np.zeros((H * upsample, W * upsample), dtype=float)
        else:
            return heatmap


    for idx, row in df.iterrows():
        try:
            pos_str = row["Position"]
            x_w, _, z_w = map(float, pos_str.split(','))
            grid_indices = world_to_grid(x_w, z_w, x_points, z_points)
            r, c = grid_indices[0], grid_indices[1] # Row, Col indices

            # Ensure indices are within bounds before adding weight
            if 0 <= r < H and 0 <= c < W:
                 # Get weight, clamp negative values to 0, handle missing values
                 w = max(0.0, float(row.get(weight_key, 0.0)))
                 heatmap[r, c] = max(heatmap[r, c], w)
            else:
                 print(f"Warning: Object at index {idx} ({row.get('objectType', 'Unknown')}) maps to out-of-bounds grid cell ({r}, {c}). Skipping.")

        except Exception as e:
            print(f"Error processing row {idx} in generate_heatmap: {e}. Row data: {row}")
            continue # Skip problematic row

    # if sigma > 0.0:
    #     # circular radius
    #     # radius = int(max(1, sigma))
    #     footprint = disk(sigma)
    #     heatmap = maximum_filter(heatmap, footprint=footprint)

    # Normalize heatmap so the maximum value is 1.0
    max_val = heatmap.max()
    min_val = heatmap.min()
    if max_val > 1e-8: # Avoid division by zero or near-zero
        # heatmap /= max_val
        heatmap = (heatmap-min_val)/(max_val - min_val)

    else:
        # If map is all zeros (or very close), keep it that way
        heatmap[:] = 0.0
    return heatmap


def map_entropy(raw):
    """Calculates the Shannon entropy of a raw (unnormalized) 2D map.

    Flattens the map, normalizes it to sum to 1, removes zero probabilities,
    and then calculates the entropy: -sum(p * log2(p)).

    Parameters
    ----------
    raw : np.ndarray
        The 2D input map (e.g., a heatmap or probability map). Values should be non-negative.

    Returns
    -------
    float
        The calculated entropy in bits. Returns 0 if the map sum is zero.
    """
    p = raw.flatten()
    map_sum = p.sum()

    if map_sum < 1e-12: # Check if map sum is effectively zero
        return 0.0 # Entropy of an empty/zero distribution is 0

    p = p / map_sum     # normalize
    # p = scaler.fit_transform(p)  # Scale to [0, 1]
    p = p[p > 1e-12]    # drop zeros/near-zeros to avoid log2(0) issues

    if p.size == 0: # Check if only zeros were present
        return 0.0

    return -np.sum(p * np.log2(p))


# --- FUNCTION TO GENERATE TRAJECTORY PLOT ---
def generate_trajectory_plot(controller, trajectory_log_path, save_path, odor_src, odor_items, scene_bounds, font_size=12):
    """Generates and saves a top-down trajectory plot overlayed on the odor field.

    - Uses 'gaussian_plume' with default parameters (q_s=2000, tau=1000, D=10).
    - Hidden: Color scale, axis labels, axis ticks, legend.
    - Title: 'Trajectory' with customizable font size.

    Parameters
    ----------
    controller : ai2thor.controller.Controller
    trajectory_log_path : str
    save_path : str
    odor_src : np.ndarray or list-like
        The 3D coordinates [x, y, z] of the source.
    odor_items : list[str]
    scene_bounds : dict
    font_size : int, optional
        Font size for the plot title (default 12).
    """
    print(f"Generating trajectory plot for {trajectory_log_path}...")

    # -----------------------------------------------------------------

    # 1. Obtain Scene Bounds
    corner_points = scene_bounds['cornerPoints']
    unique_xz_points = [(p[0], p[2]) for p in corner_points]
    
    if len(set(unique_xz_points)) < 3:
        print("Error: Cannot create valid polygon from scene bounds.")
        xs = [p[0] for p in corner_points]
        zs = [p[2] for p in corner_points]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        if min_x == max_x or min_z == max_z:
            return
    else:
        scene_polygon = Polygon(unique_xz_points)
        min_x, min_z, max_x, max_z = scene_polygon.bounds

    # 2. Compute the Plume Field
    grid_steps = 30
    X = np.linspace(min_x, max_x, grid_steps)
    Y = np.linspace(min_z, max_z, grid_steps)
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    # Extract just X and Z from the 3D odor_src for the function
    odor_source_2d = (odor_src[0], odor_src[2])

    base_map = np.zeros((grid_steps, grid_steps))
    for j in range(grid_steps):
        for i in range(grid_steps):
            world_x, world_z = X[i], Y[j]
            # Call gaussian_plume with default kwargs
            base_map[i, j] = gaussian_plume(world_x, world_z, odor_source_2d)

    # 3. Build Reachable Graph
    event = controller.step(action="GetReachablePositions")
    if not event or not event.metadata["actionReturn"]:
        print("Error: Failed to get reachable positions.")
        return
    
    reachable_positions = [(p["x"], p["z"]) for p in event.metadata["actionReturn"]]
    if not reachable_positions:
        return

    tree = KDTree(reachable_positions)
    gridSize = controller.last_event.metadata.get("gridSize", 0.25)
    if gridSize <= 0: gridSize = 0.25

    try:
        neighbors = tree.query_ball_point(reachable_positions, gridSize * 1.01)
    except ValueError as e:
        print(f"Error during KDTree query: {e}")
        return

    G = nx.Graph()
    for i in range(len(reachable_positions)):
        G.add_node(i)

    for i, nbrs in enumerate(neighbors):
        if i >= len(reachable_positions): continue
        pos_i = reachable_positions[i]
        for j in nbrs:
             if j >= len(reachable_positions): continue
             if i < j:
                 pos_j = reachable_positions[j]
                 dist = math.dist(pos_i, pos_j)
                 G.add_edge(i, j, weight=dist)

    # 4. Build Reachable Mask
    resolution = 0.2
    xg = np.arange(min_x, max_x + resolution, resolution)
    zg = np.arange(min_z, max_z + resolution, resolution)
    if len(xg) == 0 or len(zg) == 0:
        mask_closed = np.ones((1,1), dtype=bool)
    else:
        Xm, Zm = np.meshgrid(xg, zg)
        mask = np.zeros(Xm.shape, dtype=bool)
        reachable_np = np.array(reachable_positions)

        for r in range(Xm.shape[0]):
            for c in range(Xm.shape[1]):
                grid_point = np.array([Xm[r, c], Zm[r, c]])
                min_dist_sq = np.sum((reachable_np - grid_point)**2, axis=1).min()
                if min_dist_sq < (resolution * 1.5)**2:
                     mask[r, c] = True

        selem = disk(1)
        mask_closed = closing(mask, selem)

    # 5. Read Trajectory & Plot
    try:
        df = pd.read_csv(trajectory_log_path)
    except Exception as e:
         print(f"Error reading trajectory log {trajectory_log_path}: {e}")
         return

    required_cols = ['robot_x', 'robot_z', 'is_random']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Log missing columns {required_cols}.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    # (A) Odor field (No Colorbar)
    ax.contourf(X_grid, Y_grid, base_map.T, levels=np.linspace(base_map.min(), base_map.max(), 60), cmap='magma_r', zorder=1)
    
    # (B) Grey out non-reachable
    grey = np.array([0.8, 0.8, 0.8, 1.0])
    ov = np.tile(grey, (mask_closed.shape[0], mask_closed.shape[1], 1))
    ov[..., 3] = (~mask_closed).astype(float)
    ax.imshow(ov, extent=[min(xg), max(xg), min(zg), max(zg)], origin='lower', zorder=2, alpha=0.5, aspect='auto')

    # (C) CONNECT using paths
    final_endpoint = None
    if not df.empty:
        start_point = (df.robot_x.iloc[0], df.robot_z.iloc[0])
        final_endpoint = start_point

    for i in range(1, len(df)):
        s = (df.robot_x.iloc[i - 1], df.robot_z.iloc[i - 1])
        e = (df.robot_x.iloc[i], df.robot_z.iloc[i])
        if math.dist(s, e) < 1e-3:
            final_endpoint = e
            continue

        try:
            dist_s, si = tree.query(s)
            dist_e, ei = tree.query(e)

            if dist_s > gridSize * 1.5 or dist_e > gridSize * 1.5:
                ax.plot([s[0], e[0]], [s[1], e[1]], color='gray', linestyle=':', linewidth=1.0, zorder=3)
                final_endpoint = e
                continue

            if si == ei:
                 final_endpoint = e
                 continue

            path_idx = nx.shortest_path(G, source=si, target=ei, weight='weight')
            path = [reachable_positions[n] for n in path_idx]
            xs, zs = zip(*path)
            ax.plot(xs, zs, color='purple', linewidth=1.5, zorder=4)

            if len(path) >= 2:
                a0, a1 = path[-2], path[-1]
                ax.annotate(
                    '', xy=a1, xytext=a0,
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.5,
                                    mutation_scale=12, shrinkA=0, shrinkB=0),
                    zorder=5
                )
                final_endpoint = a1
        except Exception:
             ax.plot([s[0], e[0]], [s[1], e[1]], color='gray', linestyle=':', linewidth=1.0, zorder=3)
             final_endpoint = e

    # (D) Markers
    if not df.empty:
        point_colors = ['orange' if is_rand else 'purple' for is_rand in df['is_random']]
        ax.scatter(df.robot_x, df.robot_z, color=point_colors, s=30, zorder=5, edgecolors='black', linewidth=0.5)
        ax.scatter(df.robot_x.iloc[0], df.robot_z.iloc[0],
                   color='lime', s=100, marker='o', zorder=6, edgecolors='black', label='_nolegend_')

    if final_endpoint is not None:
         last_df_point = (df.robot_x.iloc[-1], df.robot_z.iloc[-1])
         ax.scatter(last_df_point[0], last_df_point[1],
                    color='cyan', s=100, marker='o', zorder=6, edgecolors='black', label='_nolegend_')

    # Odor Source marker
    odor_x, _, odor_z = odor_src
    ax.scatter(odor_x, odor_z, color='red', s=200, marker='*', zorder=7, edgecolors='black', label='_nolegend_')

    # Success region
    circle = Circle((odor_x, odor_z), radius=1.0, fill=False,
                    edgecolor='red', linewidth=1.5, linestyle='--', zorder=7)
    ax.add_patch(circle)

    # [REMOVED] Legend

    # Final Config
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Hide ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Title
    # ax.set_title("Trajectory", fontsize=font_size)

    fig.tight_layout()

    try:
        fig.savefig(save_path, dpi=300)
        print(f"Successfully saved trajectory plot to {save_path}")
    except Exception as e:
        print(f"Error saving trajectory plot to {save_path}: {e}")
    plt.close(fig)


# ==========================
# SUMMARY GENERATION FUNCTIONS
# ==========================

def calculate_total_distance(csv_file):
    """Helper to calculate distance, elapsed time, and final target from a single log file."""
    df = pd.read_csv(csv_file)

    if not {'robot_x', 'robot_z'}.issubset(df.columns):
        raise ValueError(f"File {csv_file} missing 'robot_x' or 'robot_z' columns")

    # Compute Euclidean distance
    df['distance'] = np.sqrt((df['robot_x'].diff())**2 + (df['robot_z'].diff())**2)
    total_distance = round(df['distance'].sum(), 2)
    steps = df['step'].iloc[-1]

    # Final distance = target coordinate estimation error.
    # Random Walk logs don't estimate a source coordinate, so this column
    # may be absent — fall back to gt_distance_from_source, else NaN.
    if 'target_coord_estimation_error' in df.columns:
        final_dist = df['target_coord_estimation_error'].iloc[-1]
    elif 'gt_distance_from_source' in df.columns:
        final_dist = df['gt_distance_from_source'].iloc[-1]
    else:
        final_dist = np.nan

    # Guard the rounding in case the value is NaN or non-numeric
    final_dist = round(final_dist, 3) if pd.notna(final_dist) else np.nan

    if 'step_time' in df.columns:
        total_step_time = round(df['step_time'].sum(), 2)
    else:
        total_step_time = np.nan

    # Extract final target object (useful for Fusion/VLM methods)
    final_target = df['target_object'].iloc[-1] if 'target_object' in df.columns else np.nan

    return total_distance, final_dist, steps, total_step_time, final_target

def generate_batch_summary(base_root, alg_folder, param_list, run_count, target_items, alg_choice='F'):
    """
    Generates the summary CSV for a specific Algorithm/Environment batch.
    Takes 'target_items' to validate that the robot identified the correct source.
    """
    print(f"\n--- Generating Summary for {alg_folder} ---")
    base_path = os.path.join(os.getcwd(), base_root) 
    run_numbers = list(range(1, run_count + 1))
    
    # Helper to check if target matches (handles both list and string inputs, ignores the _xyz instance ID)
    def is_target_match(f_target, targets, alg_choice):
        if alg_choice == 'O' or alg_choice == 'R':
            return True # Skip target prediction check for Olfaction-only/Random methods
        if pd.isna(f_target):
            return False
            
        # Clean the logged target by taking everything before the first underscore
        f_target_base = str(f_target).split('_')[0]
        
        # Safely unpack list or tuple target wrappers
        if isinstance(targets, (list, tuple)) and len(targets) > 0:
            target_raw = targets[0]
        else:
            target_raw = targets
            
        target_base = str(target_raw).split('_')[0]
        return f_target_base == target_base

    # Robust condition combining empty param checks and folder name pattern validation
    is_parameterless = (
        not param_list or 
        "save_R_" in alg_folder or 
        "_R_" in alg_folder or 
        "save_V_" in alg_folder
    )

    # --- LOGIC FOR ALGORITHMS WITHOUT PARAMETERS (e.g., Random Walk, VLM) ---
    if is_parameterless: 
        # FIXED: Variable naming aligned exactly with what's used in the loop below
        col_total_dist, col_steps, col_final_dist, col_total_time, col_final_target = [], [], [], [], []
        
        for run in run_numbers:
            csv_path = os.path.join(base_path, alg_folder, str(run), "trajectory_log.csv")
            if os.path.exists(csv_path):
                try:
                    t_dist, f_dist, steps, t_time, f_target = calculate_total_distance(csv_path)
                    col_total_dist.append(t_dist)
                    col_steps.append(steps)
                    col_final_dist.append(f_dist)
                    col_total_time.append(t_time)
                    col_final_target.append(f_target)
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")
                    [lst.append(np.nan) for lst in [col_total_dist, col_steps, col_final_dist, col_total_time, col_final_target]]
            else:
                [lst.append(np.nan) for lst in [col_total_dist, col_steps, col_final_dist, col_total_time, col_final_target]]

        df_combined = pd.DataFrame({
            'Total Distance Traveled': col_total_dist,
            'Steps Taken': col_steps,
            'Final Distance from Source': col_final_dist,
            'Total Step Time': col_total_time,
            'Final Target': col_final_target
        }, index=run_numbers)

    # --- LOGIC FOR ALGORITHMS WITH PARAMETERS (Fusion 'F' or Olfaction 'O') ---
    else:
        data_total_dist, data_steps, data_final_dist, data_total_time, data_final_target = {}, {}, {}, {}, {}

        for param in param_list:
            col_total_dist, col_steps, col_final_dist, col_total_time, col_final_target = [], [], [], [], []
            param_str = str(param) 
            
            for run in run_numbers:
                csv_path = os.path.join(base_path, alg_folder, param_str, str(run), "trajectory_log.csv")
                
                if os.path.exists(csv_path):
                    try:
                        t_dist, f_dist, steps, t_time, f_target = calculate_total_distance(csv_path)
                        col_total_dist.append(t_dist)
                        col_steps.append(steps)
                        col_final_dist.append(f_dist)
                        col_total_time.append(t_time)
                        col_final_target.append(f_target)
                    except Exception as e:
                        [lst.append(np.nan) for lst in [col_total_dist, col_steps, col_final_dist, col_total_time, col_final_target]]
                else:
                    [lst.append(np.nan) for lst in [col_total_dist, col_steps, col_final_dist, col_total_time, col_final_target]]

            data_total_dist[param_str] = col_total_dist
            data_steps[param_str] = col_steps
            data_final_dist[param_str] = col_final_dist
            data_total_time[param_str] = col_total_time
            data_final_target[param_str] = col_final_target

        df_total = pd.DataFrame(data_total_dist, index=run_numbers)
        df_steps = pd.DataFrame(data_steps, index=run_numbers)
        df_final = pd.DataFrame(data_final_dist, index=run_numbers)
        df_total_time = pd.DataFrame(data_total_time, index=run_numbers)
        df_target = pd.DataFrame(data_final_target, index=run_numbers)

        df_combined = pd.concat(
            [df_total, df_steps, df_final, df_total_time, df_target], 
            axis=1, 
            keys=['Total Distance Traveled', 'Steps Taken', 'Final Distance from Source', 'Total Step Time', 'Final Target']
        )

    # Save to CSV
    df_combined.index.name = "Run Number"
    savePath = os.path.join(base_path, alg_folder, f"trajectory_summary_{alg_folder}.csv")
    df_combined.to_csv(savePath)
    print(f"Summary saved to: {savePath}")


def generate_flat_summary(base_path):
    """
    Traverses the base_path directory, parses individual folder information, 
    and outputs a completely flat combined CSV file containing row-by-row data 
    for every single execution run.
    """
    print(f"=========================================================")
    print(f"Generating Flat Combined Run Summary from: {base_path}")
    print(f"=========================================================\n")

    # Map method directory characters to clean, descriptive experiment label fields
    METHOD_MAP = {
        'f': 'Fusion',
        'o': 'Olfactory Only',
        'r': 'Random Walk'
    }
    
    # Map the object substrings found in folder tokens to their clean targets
    OBJECT_MAP = {
        'stoveburner': 'Stove',
        'coffeemachine': 'Coffee Machine',
        'garbagecan': 'Garbage Can'
    }
    
    thresholds = ['0.2', '0.5', '0.8', '0.9']
    all_flat_rows = []

    if not os.path.exists(base_path):
        print(f"Error: Base path '{base_path}' does not exist.")
        return

    # Scan the base folder directly
    for alg_folder in os.listdir(base_path):
        alg_path = os.path.join(base_path, alg_folder)
        if not os.path.isdir(alg_path) or not alg_folder.startswith("save_"):
            continue
            
        folder_lower = alg_folder.lower()
        
        # 1. Parse out the specific Method Type Code Identifier
        parts = alg_folder.split('_')
        if len(parts) < 2:
            continue
        method_code = parts[1].lower()
        method_label = METHOD_MAP.get(method_code, f"Unknown ({parts[1]})")
        
        # 2. Parse out the structural Environment Target Object Type
        obj_label = None
        for key_substr, target_obj in OBJECT_MAP.items():
            if key_substr in folder_lower:
                obj_label = target_obj
                break
                
        if not method_label or not obj_label:
            continue  # Skip unmapped folder path structures safely

        # Locate the batch's underlying compiled summary table file
        csv_file = None
        for filename in os.listdir(alg_path):
            if "trajectory_summary" in filename and filename.endswith(".csv"):
                csv_file = os.path.join(alg_path, filename)
                break
        
        if not csv_file:
            print(f"Warning: Batch trajectory summary table file not found in {alg_path}")
            continue
            
        try:
            is_parameterless = method_code in ['r', 'vnone']
            
            if is_parameterless:
                # Flat single index layer per run data table configuration format
                df = pd.read_csv(csv_file, index_col=0)
                
                # Iterate row-by-row through your 16 individual test runs
                for run_number, row in df.iterrows():
                    flat_data_point = {
                        'Algorithm / Method': method_label,
                        'Entropy Threshold': 'N/A (Parameterless)',
                        'Target Object': obj_label,
                        'Run Number': run_number,
                        'Total Distance Traveled': row.get('Total Distance Traveled', np.nan),
                        'Steps Taken': row.get('Steps Taken', np.nan),
                        'Final Distance from Source': row.get('Final Distance from Source', np.nan),
                        'Total Step Time': row.get('Total Step Time', np.nan),
                        'Final Predicted Target': row.get('Final Target', np.nan)
                    }
                    all_flat_rows.append(flat_data_point)
            else:
                # Multi-index threshold tracking configuration layout format (Header consists of 2 lines)
                df = pd.read_csv(csv_file, header=[0, 1], index_col=0)
                
                # Unpack every run nested underneath each distinct fractional parameter threshold
                for thresh in thresholds:
                    for run_number in df.index:
                        # Safely fetch fields matching multi-level key indices
                        dist_val = df.at[run_number, ('Total Distance Traveled', thresh)] if ('Total Distance Traveled', thresh) in df.columns else np.nan
                        steps_val = df.at[run_number, ('Steps Taken', thresh)] if ('Steps Taken', thresh) in df.columns else np.nan
                        final_dist_val = df.at[run_number, ('Final Distance from Source', thresh)] if ('Final Distance from Source', thresh) in df.columns else np.nan
                        time_val = df.at[run_number, ('Total Step Time', thresh)] if ('Total Step Time', thresh) in df.columns else np.nan
                        target_pred_val = df.at[run_number, ('Final Target', thresh)] if ('Final Target', thresh) in df.columns else np.nan
                        
                        flat_data_point = {
                            'Algorithm / Method': method_label,
                            'Entropy Threshold': thresh,
                            'Target Object': obj_label,
                            'Run Number': run_number,
                            'Total Distance Traveled': dist_val,
                            'Steps Taken': steps_val,
                            'Final Distance from Source': final_dist_val,
                            'Total Step Time': time_val,
                            'Final Predicted Target': target_pred_val
                        }
                        all_flat_rows.append(flat_data_point)
                        
        except Exception as e:
            print(f"Error flattening run datasets inside file {csv_file}: {e}")

    # --- SAVE THE COMPREHENSIVE COMBINED DATAFRAME ---
    if len(all_flat_rows) == 0:
        print("Error: No data rows could be extracted. Flat CSV generation stopped.")
        return
        
    df_combined_flat = pd.DataFrame(all_flat_rows)
    
    # Save the absolute raw multi-run sheet safely inside your workspace directory
    save_path = os.path.join(base_path, "Flat_Runs_Combined_Summary.csv")
    df_combined_flat.to_csv(save_path, index=False)
    
    print(f"\n--- SUCCESS ---")
    print(f"Total raw iteration row blocks mapped: {len(df_combined_flat)}")
    print(f"Flat comprehensive spreadsheet compiled at: {save_path}")
    print("\nPreview of Compiled Run spreadsheet matrix (First 10 Rows):")
    print(df_combined_flat.head(10).to_string(index=False))

    return df_combined_flat