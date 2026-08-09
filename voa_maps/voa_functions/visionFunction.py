from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')
import cv2
import os
import pandas as pd
import numpy as np

from voa_functions.utils import world_to_grid

# ==========================
# Dirichlet Object Map  (config)
# ==========================

# Classes YOLO is allowed to report.
TARGET_NAMES = ['Apple', 'Book', 'Bottle', 'Bowl', 'Bread', 'ButterKnife', 'Cabinet',
                'CoffeeMachine', 'CounterTop', 'CreditCard', 'Cup', 'DishSponge',
                'Drawer', 'Egg', 'Faucet', 'Fork', 'Fridge', 'GarbageCan', 'HousePlant',
                'Kettle', 'Knife', 'Lettuce', 'LightSwitch', 'Microwave', 'Mug', 'Pan',
                'PaperTowelRoll', 'PepperShaker', 'Plate', 'Pot', 'Potato', 'SaltShaker',
                'Shelf', 'ShelvingUnit', 'Sink', 'SoapBottle', 'Spatula', 'Spoon',
                'Statue', 'Stool', 'StoveBurner', 'StoveKnob', 'Toaster', 'Tomato',
                'Vase', 'WineBottle']

# Classes dropped before they ever reach the map.
EXCLUDE_NAMES = ['Cabinet', 'Cabinet_opened', 'CounterTop', 'Drawer', 'Drawer_opened',
                 'Floor', 'Shelf', 'Window', 'Apple_sliced', 'Bowl_filled']

# Canonical class axis of the Dirichlet map (index k of the (H, W, K) array).
# "empty floor" is a real class, not the absence of one.
#
# Without it, a cell the robot has looked at and found bare is indistinguishable
# from a cell it has never looked at: both sit at the untouched prior. That is
# why the visual likelihood map stayed within 0.2% of uniform over 200
# detections -- a handful of object bumps against hundreds of cells that all
# still held the prior. Giving free space its own class lets an observed-empty
# cell accumulate evidence for something that scores LOW against the goal
# phrase, so looking at nothing is informative.
EMPTY_CLASS = 'empty floor'

CLASSES   = [n for n in TARGET_NAMES if n not in EXCLUDE_NAMES] + [EMPTY_CLASS]
K_CLASSES = len(CLASSES)
CLS_IDX   = {c: i for i, c in enumerate(CLASSES)}

# PRIOR_STRENGTH is the total pseudo-count mass, split evenly over K classes,
# so alpha_k = PRIOR_STRENGTH / K. This trades likelihood sharpness against
# entropy semantics, and the two want opposite values (K = 42 here):
#
#   PRIOR_STRENGTH = 1.0  -> alpha_k = 0.024 (sparse prior). Detections dominate
#       immediately, so the semantic likelihood map is sharp. BUT an unobserved
#       cell has Dirichlet entropy 1.39 of a possible 5.39 bits: empty cells look
#       *confident*, which is backwards if you use this entropy for frontier
#       scoring.
#   PRIOR_STRENGTH = K_CLASSES -> alpha_k = 1.0 (flat Dirichlet). Unobserved
#       cells sit at 4.80 bits, near max, so entropy reduction tracks coverage
#       properly. Costs likelihood sharpness: ~10 detections to move a cell.
#
# Default is the sparse prior because the likelihood map is the primary output.
# Switch to K_CLASSES if dirichlet_entropy drives exploration.
PRIOR_STRENGTH   = 1.0                              # Dirichlet pseudo-count mass
PRIOR            = np.ones(K_CLASSES) / K_CLASSES   # uniform prior over classes
USE_CONF_AS_EVIDENCE = False                        # False = perfect YOLO (+1)

# Free-space ("looked and saw nothing") evidence, see update_free_space.
FREE_FOV_DEG     = 90.0    # camera horizontal field of view
FREE_RANGE_M     = 2.5     # only mark cells this close as reliably observed
FREE_MIN_RANGE_M = 0.25    # skip the cell the robot stands in
FREE_EVIDENCE    = 1.0     # weaker than a detection's 1.0, on purpose
CLIP_NEGATIVE_SIM    = True                         # clip cosine sim at 0

_SIM_CACHE = {}                                     # goal_phrase -> (K,) sim vector


def configure_classes(names, prior_strength=None, add_empty=True):
    """Rebuild the Dirichlet class axis for a different detector.

    CLASSES defaults to the AI2-THOR label set; on hardware it must match your
    fine-tuned model.names exactly, or every detection falls through the
    CLS_IDX lookup and the object map silently stays at its prior.

    EMPTY_CLASS is appended automatically -- it is not something the detector
    reports, it is inferred from looking and seeing nothing.
    """
    global CLASSES, K_CLASSES, CLS_IDX, PRIOR, PRIOR_STRENGTH, _SIM_CACHE
    CLASSES = list(names)
    if add_empty and EMPTY_CLASS not in CLASSES:
        CLASSES.append(EMPTY_CLASS)
    K_CLASSES = len(CLASSES)
    CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
    PRIOR = np.ones(K_CLASSES) / K_CLASSES
    if prior_strength is not None:
        PRIOR_STRENGTH = float(prior_strength)
    _SIM_CACHE = {}          # similarities are per class list; stale entries are wrong
    return CLASSES


# ==========================
# Dirichlet Object Map  (functions)
# ==========================

def init_object_map(x_points, z_points):
    """Dirichlet parameters beta, shape (H, W, K) with H=len(z_points), W=len(x_points).

    Same [row, col] indexing convention as BayesianAgent.prob_map, so the two
    maps can be multiplied elementwise.
    """
    H, W = len(z_points), len(x_points)
    return np.ones((H, W, K_CLASSES)) * (PRIOR_STRENGTH * PRIOR)


def update_object_map(beta, className, x_world, z_world, x_points, z_points, evidence=1.0):
    """Perfect-YOLO Dirichlet update: +evidence to the detected class, +0 to the rest.

    beta is modified in place and also returned.
    """
    if className not in CLS_IDX:
        return beta
    r, c = world_to_grid(x_world, z_world, x_points, z_points)[:2]
    if 0 <= r < beta.shape[0] and 0 <= c < beta.shape[1]:
        beta[r, c, CLS_IDX[className]] += evidence
    return beta


def update_free_space(beta, x_points, z_points, robot_x, robot_z, robot_yaw_deg,
                      detected_cells=(), fov_deg=FREE_FOV_DEG,
                      max_range=FREE_RANGE_M, evidence=FREE_EVIDENCE,
                      min_range=FREE_MIN_RANGE_M):
    """Accumulate EMPTY_CLASS evidence for cells seen to contain nothing.

    Every cell inside the camera frustum that did not receive a detection this
    frame gets `evidence` on the empty class.

    Three deliberate conservatisms:

    * `evidence` defaults well below the 1.0 a detection carries. Absence of
      evidence is weaker than evidence of absence -- a small object can sit in
      a cell and be missed, so a "nothing here" observation should not
      out-vote a real detection.
    * `max_range` is short. Depth degrades with distance and the frustum widens,
      so far cells are both less reliably observed and more numerous.
    * OCCLUSION IS NOT HANDLED. A cell behind a wall or a large object is
      inside the frustum but was never actually seen, and this will mark it
      empty anyway. The short range limits the damage; if it matters, gate the
      update with the depth frame (sim) or the laser scan (hardware) before
      calling this.

    beta is modified in place and returned.
    """
    if EMPTY_CLASS not in CLS_IDX:
        return beta
    X, Z = np.meshgrid(np.asarray(x_points, float), np.asarray(z_points, float))
    dx, dz = X - robot_x, Z - robot_z
    rng = np.hypot(dx, dz)
    rel = (np.degrees(np.arctan2(dx, dz)) - robot_yaw_deg + 180.0) % 360.0 - 180.0
    seen = (np.abs(rel) <= fov_deg / 2.0) & (rng <= max_range) & (rng >= min_range)
    for (r, c) in detected_cells:
        if 0 <= r < seen.shape[0] and 0 <= c < seen.shape[1]:
            seen[r, c] = False
    beta[seen, CLS_IDX[EMPTY_CLASS]] += evidence
    return beta


def observed_mask(beta, threshold=None):
    """Cells with any accumulated evidence at all -- the explored footprint."""
    thr = threshold if threshold is not None else PRIOR_STRENGTH * 1.001
    return beta.sum(axis=2) > thr


def object_posterior(beta):
    """p(o_ci | z_c) — per-cell categorical posterior over classes, shape (H, W, K)."""
    return beta / beta.sum(axis=2, keepdims=True)


def dirichlet_entropy(beta, bits=True):
    """Expected Shannon entropy of the per-cell class distribution under the Dirichlet.

    H[c] = psi(a0 + 1) - (1/a0) * sum_k beta_k * psi(beta_k + 1)

    This is *semantic* (which-object) uncertainty per cell, not spatial
    uncertainty about where the source is. Use it for frontier / epistemic
    scoring; use map_entropy(visual_likelihood(...)) for the spatial quantity
    comparable to the olfactory Bayesian entropy.

    Returns
    -------
    np.ndarray
        (H, W) entropy map, in bits if `bits` else nats.
    """
    from scipy.special import digamma
    a0 = beta.sum(axis=2)
    H = digamma(a0 + 1.0) - (1.0 / a0) * np.sum(beta * digamma(beta + 1.0), axis=2)
    return H / np.log(2.0) if bits else H


def class_goal_similarity(goal_phrase, clip_negative=CLIP_NEGATIVE_SIM):
    """(K,) cosine similarity between the odor/goal phrase and each class name.

    Cached per goal_phrase — the SBERT encode only runs once per run.
    """
    key = (goal_phrase, clip_negative)
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]
    goal_emb = model.encode(goal_phrase, convert_to_tensor=True)
    cls_emb = model.encode(CLASSES, convert_to_tensor=True)
    s = np.asarray(util.cos_sim(goal_emb, cls_emb)[0].cpu().numpy(), dtype=float)
    if clip_negative:
        clipped = np.clip(s, 0.0, None)
        if clipped.sum() <= 1e-12:
            # Every class scored negative, so clipping wipes the vector out and
            # the visual map collapses. Shift instead: the map only uses the
            # ranking, and a shift preserves it where a clip destroys it.
            print(f"[vision] all class similarities negative for {goal_phrase!r}; "
                  f"shifting instead of clipping to preserve ranking")
            s = s - s.min()
            if s.sum() <= 1e-12:
                s = np.ones_like(s)
        else:
            s = clipped
    s = zero_empty_similarity(s)
    _SIM_CACHE[key] = s
    return s


def zero_empty_similarity(sim):
    """Force EMPTY_CLASS to exactly 0 similarity.

    Empty floor is not a candidate source, so its semantic score is zero by
    definition rather than whatever a sentence transformer returns for the
    phrase "empty floor". Two reasons this matters:

    * It removes an untestable dependency. The free-space mechanism only works
      if the empty class scores LOW -- measured, an empty-class similarity of
      0.05 gave a 0.019 entropy drop while 0.40 gave 0.0001, i.e. nothing. A
      hard zero guarantees the low end instead of hoping the encoder agrees.
    * It survives the negative-similarity fallback above. That path shifts
      rather than clips, which would otherwise hand the empty class a positive
      score again.

    A fully-empty cell still does not reach exactly zero likelihood, because
    the Dirichlet prior keeps some mass on the real classes -- so the map stays
    normalisable and vision can never veto a cell outright.
    """
    s = np.asarray(sim, dtype=float).copy()
    idx = CLS_IDX.get(EMPTY_CLASS)
    if idx is not None and idx < len(s):
        s[idx] = 0.0
    return s


def semantic_map_from_similarity(beta, sim, normalize=True):
    """Project the per-cell class posterior onto an arbitrary similarity vector.

        L(c) = sum_k p(o_k | z_c) * sim_k

    Generic backend shared by the visual (SBERT odor-vs-class) and auditory
    (CLAP sound-vs-class) semantic maps, so both branches stay symmetric.
    """
    post = object_posterior(beta)
    lik = np.tensordot(post, np.asarray(sim, float), axes=([2], [0]))   # (H, W)
    if normalize:
        tot = lik.sum()
        # A degenerate similarity vector must yield a UNIFORM map, not a map of
        # zeros. Zeros do not sum to 1, break the entropy, and turn into a
        # constant -log(eps) penalty in the log-space fusion.
        lik = lik / tot if tot > 1e-12 else np.full_like(lik, 1.0 / lik.size)
    return lik


def combine_similarity(text_sim, audio_sim, w_text=0.5, w_audio=0.5, mode='sum'):
    """Merge SBERT odor-vs-class and CLAP sound-vs-class similarity vectors.

    'sum'     : weighted arithmetic mean. Either cue alone can carry a class,
                which is what you want when a source is only odorous or only
                audible. This is the default.
    'product' : geometric-style AND. A class must score on both cues, which is
                sharper for genuinely bimodal sources but zeroes out any class
                that either encoder dislikes -- brittle when one modality is
                absent.

    Returns
    -------
    np.ndarray
        (K,) combined similarity, clipped at 0.
    """
    t = np.asarray(text_sim, float) if text_sim is not None else None
    a = np.asarray(audio_sim, float) if audio_sim is not None else None
    if t is None and a is None:
        raise ValueError("combine_similarity needs at least one similarity vector")
    if t is None:
        return np.clip(a, 0.0, None)
    if a is None:
        return np.clip(t, 0.0, None)
    if mode == 'product':
        return np.clip(t, 0.0, None) * np.clip(a, 0.0, None)
    return np.clip(w_text * t + w_audio * a, 0.0, None)


def fuse_log_evidence(maps, weights=None, eps=1e-12):
    """Fuse belief maps by accumulating log-evidence, then renormalise.

        log p_total = sum_m w_m * log p_m      ->      p_total = softmax(log p_total)

    Equivalent to a weighted product of experts, but computed in log space with
    the max subtracted before exponentiating, so a long run cannot underflow to
    an all-zero map the way repeated elementwise multiplication does. The
    weights are the per-modality reliabilities (lambda_V, lambda_O, lambda_A);
    setting one to 0 cleanly ablates that modality.

    Parameters
    ----------
    maps : list[np.ndarray]
        Normalised (H, W) belief maps. Entries that are None are skipped.
    weights : list[float], optional
        Per-map weight. Defaults to 1.0 for every map.

    Returns
    -------
    np.ndarray
        Fused (H, W) map summing to 1.
    """
    usable = [(m, 1.0) for m in maps if m is not None] if weights is None else \
             [(m, w) for m, w in zip(maps, weights) if m is not None and w != 0.0]
    if not usable:
        raise ValueError("fuse_log_evidence needs at least one map")

    log_total = np.zeros_like(usable[0][0], dtype=float)
    for m, w in usable:
        mm = np.asarray(m, float)
        s = mm.sum()
        if s > eps:
            mm = mm / s
        log_total += w * np.log(mm + eps)

    log_total -= log_total.max()          # stabilise before exponentiating
    p = np.exp(log_total)
    total = p.sum()
    return p / total if total > eps else np.full_like(p, 1.0 / p.size)


def visual_likelihood(beta, goal_phrase, normalize=True):
    """Visual likelihood map: object posterior projected onto semantic relevance.

        L(c) = sum_k p(o_k | z_c) * sim(class_k, goal_phrase)

    Cells with no evidence keep the uniform prior, so they take the mean
    similarity as a floor rather than going to zero — the map never collapses.

    Returns
    -------
    np.ndarray
        (H, W) map, normalized to sum to 1 when `normalize`.
    """
    post = object_posterior(beta)
    sim = class_goal_similarity(goal_phrase)
    lik = np.tensordot(post, sim, axes=([2], [0]))          # (H, W)
    if normalize:
        tot = lik.sum()
        lik = lik / tot if tot > 1e-12 else np.full_like(lik, 1.0 / lik.size)
    return lik


def visual_likelihood_multimodal(beta, goal_phrase=None, sound_sim=None,
                                 w_text=0.5, w_audio=0.5, mode='sum'):
    """Semantic map driven by odor text, sound (CLAP), or both.

    Condition-dependent use:
      odor only  -> goal_phrase set, sound_sim None
      sound only -> goal_phrase None, sound_sim set
      both       -> both set, combined via combine_similarity
    """
    text_sim = class_goal_similarity(goal_phrase) if goal_phrase else None
    sim = combine_similarity(text_sim, sound_sim, w_text, w_audio, mode)
    return semantic_map_from_similarity(beta, sim)


# ==========================
# Vision Functions
# ==========================

def boxDepth(x, y, w, h, controller):
    """Estimates the depth of an object based on its bounding box in the depth frame.

    Calculates the 90th percentile depth value within the bounding box region
    defined by (x, y, w, h) in the controller's last depth frame.

    Parameters
    ----------
    x : int
        Center x-coordinate of the bounding box (in pixels).
    y : int
        Center y-coordinate of the bounding box (in pixels).
    w : int
        Width of the bounding box (in pixels).
    h : int
        Height of the bounding box (in pixels).
    controller : ai2thor.controller.Controller
        The AI2-THOR controller instance.

    Returns
    -------
    float
        The estimated depth, rounded to one decimal place. Returns 0 or NaN if
        the box is invalid or depth data is missing.
    """
    # Calculate pixel bounds, ensuring they are within frame dimensions
    frame_h, frame_w = controller.last_event.depth_frame.shape[:2]
    vMin = max(0, y - h // 2)
    vMax = min(frame_h, y + h // 2)
    hMin = max(0, x - w // 2)
    hMax = min(frame_w, x + w // 2)

    # Check if the box has valid dimensions
    if vMin >= vMax or hMin >= hMax:
        print(f"Warning: Invalid bounding box dimensions [{vMin}:{vMax}, {hMin}:{hMax}] for depth calculation.")
        return 0.0 # Or np.nan

    depthFrame = controller.last_event.depth_frame
    # Extract depth values within the box
    depth_values = depthFrame[vMin:vMax, hMin:hMax]

    # Check if depth_values is empty (e.g., due to invalid box slicing)
    if depth_values.size == 0:
         print(f"Warning: No depth values found in box [{vMin}:{vMax}, {hMin}:{hMax}].")
         return 0.0 # Or np.nan

    # Calculate 90th percentile
    boxDepth = np.percentile(depth_values, 90)
    return round(boxDepth, 1)


def coord23D_focal(x, y, w, h, controller):
    """
    Converts 2D pixel coordinates to 3D world coordinates using the 
    Focal Length (Intrinsic) method.
    """
    # --- 1. Camera Intrinsics ---
    W, H = 300, 300
    fov = 90
    # Calculate focal length: f = W / (2 * tan(FOV/2))
    f = W / (2 * np.tan(np.deg2rad(fov / 2)))
    
    # Principal point (center of the image)
    cx, cy = W / 2.0, H / 2.0

    # --- 2. Estimate Depth ---
    # Assuming boxDepth is defined elsewhere as in your previous snippet
    d = boxDepth(x, y, w, h, controller)
    if d <= 0:
        return 0.0, 0.0, 0.0

    # --- 3. Back-projection to Camera Space ---
    # In AI2-THOR (Unity), Camera Space is: +X Right, +Y Up, +Z Forward
    # (u, v) pixels: u=0 is left, v=0 is top.
    X_c = (x - cx) * d / f
    Y_c = -(y - cy) * d / f  # Negative because pixel 'y' increases downwards
    Z_c = d

    # --- 4. Coordinate Transformation to World Space ---
    event = controller.last_event
    
    # Get Camera World Position (more accurate than Agent Position)
    cam_pos = event.metadata['cameraPosition']
    tx, ty, tz = cam_pos['x'], cam_pos['y'], cam_pos['z']

    # Get Rotation Angles (converted to Radians)
    # Yaw: Agent's rotation around Y axis
    # Pitch: Camera's horizon (rotation around X axis)
    yaw = np.deg2rad(event.metadata['agent']['rotation']['y'])
    pitch = np.deg2rad(event.metadata['agent']['cameraHorizon'])

    # Pitch Matrix (Rotation around X)
    # In AI2-THOR, positive pitch is looking DOWN.
    # R_pitch = np.array([
    #     [1, 0, 0],
    #     [0, np.cos(pitch), -np.sin(pitch)],
    #     [0, np.sin(pitch), np.cos(pitch)]
    # ])

    # Yaw Matrix (Rotation around Y)
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Combine: Local -> Pitched -> Yawed
    P_camera = np.array([X_c, Y_c, Z_c])
    P_world_rotated = R_yaw @ P_camera

    # Add translation to get Global Coordinates
    final_x = P_world_rotated[0] + tx
    final_y = P_world_rotated[1] + ty
    final_z = P_world_rotated[2] + tz

    return round(final_x, 3), round(final_y, 3), round(final_z, 3)

def visionBranch(model, itemDF, controller, save_dir, step_count, fusionMode = None, confThr=0.1,
                 beta=None, x_points=None, z_points=None,
                 robot_pose=None, free_space=True):
    """Detects objects using YOLO, estimates their 3D position, and updates a DataFrame.

    Runs YOLOv8 on the current camera frame. For each detection above `confThr`:
    1. Estimates the 3D world coordinates using `coord23D`.
    2. Checks if an object of the same class already exists in `itemDF` nearby (dist < 0.5).
    3. If nearby object exists, averages its position with the new detection.
    4. If no nearby object exists, adds the new detection as a new row in `itemDF`.

    Parameters
    ----------
    model : ultralytics.YOLO
        The loaded YOLOv8 model instance.
    itemDF : pd.DataFrame
        DataFrame containing information about previously detected objects.
        Expected columns: 'objectType' (str), 'Position' (str "x, y, z"), 'Conf' (float).
    controller : ai2thor.controller.Controller
        The AI2-THOR controller instance.
    confThr : float, optional
        Confidence threshold for YOLO detections. Defaults to 0.3.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with new or averaged object detections.
    """
    # Get current frame and run YOLO detection
    current_frame = np.array(controller.last_event.frame)

    object_metadata = controller.last_event.metadata["objects"]

    # Class filters now live at module scope so the Dirichlet map shares the same axis
    exclude_names = EXCLUDE_NAMES
    target_names = TARGET_NAMES
    target_classes = [idx for idx, name in model.names.items() if name in target_names]
    # Run inference
    results = model(np.array(controller.last_event.frame), classes=target_classes)

    # show yolo detections in a window
    # results[0].plot()

    # Make a copy to avoid modifying the original DataFrame passed in
    updated_itemDF = itemDF.copy()
    _detected_cells = []

    annotated_img = results[0].plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{save_dir}/yolo_step{step_count}.jpg", annotated_img_rgb)
    cv2.imshow("YOLO Result", annotated_img_rgb)
    cv2.waitKey(1) # Display the window briefly; adjust as needed for your environment

    # Process detections
    for box in results[0].boxes:
        confidence = box.conf[0].item()
        if confidence > confThr:
            class_id = int(box.cls[0].item())
            className = model.names[class_id]
            
            if className in exclude_names:
                continue # Skip excluded classes
            # Extract detection info
            # print(f"Detected {className}")
            x, y, w, h = box.xywh[0] # Center x, y, width, height

            x_pix, y_pix, w_pix, h_pix = round(x.item()), round(y.item()), round(w.item()), round(h.item())

            # Estimate 3D position
            if fusionMode == 'focal':
                ## Focal length method
                x_glob, y_glob, z_glob = coord23D_focal(x_pix, y_pix, w_pix, h_pix, controller)

            elif fusionMode == 'GT':
                ## Ground truth lookup
                object_info = next((obj for obj in object_metadata if obj["objectType"] == className), None)
                # print(f"Object metadata for {className}: {object_info}")
                x_glob, y_glob, z_glob = object_info['position']['x'], object_info['position']['y'], object_info['position']['z']
            else:
                print(f"Warning: Unknown fusionMode '{fusionMode}' specified. Defaulting to focal method.")
                x_glob, y_glob, z_glob = coord23D_focal(x_pix, y_pix, w_pix, h_pix, controller)


            # Skip if coord23D failed
            if x_glob == 0.0 and y_glob == 0.0 and z_glob == 0.0:
                 continue

            # --- Dirichlet object-map evidence (perfect YOLO: +1 to detected class) ---
            if beta is not None and x_points is not None and z_points is not None:
                ev = confidence if USE_CONF_AS_EVIDENCE else 1.0
                update_object_map(beta, className, x_glob, z_glob, x_points, z_points, evidence=ev)
                _detected_cells.append(tuple(int(v) for v in
                                             world_to_grid(x_glob, z_glob, x_points, z_points)))

            new_position = np.array([x_glob, y_glob, z_glob])

            updated = False
            # --- *** CHECK IF DATAFRAME IS EMPTY *** ---
            # Only try to match if the DataFrame has data and the necessary column
            if not updated_itemDF.empty and 'objectType' in updated_itemDF.columns:
                match_indices = updated_itemDF.index[updated_itemDF['objectType'] == className].tolist()

                for idx in match_indices:
                    try:
                        # Parse existing position string
                        existing_position_str = updated_itemDF.loc[idx, 'Position']
                        existing_position = np.array([float(val.strip()) for val in existing_position_str.split(',')])

                        # Check distance
                        dist = np.linalg.norm(new_position - existing_position)
                        if dist < 0.1: # don't take average: 0.1, take average: 10.0
                            # Average positions if close enough
                            avg_position = (new_position + existing_position) / 2.0
                            updated_itemDF.loc[idx, 'Position'] = f"{avg_position[0]:.2f}, {avg_position[1]:.2f}, {avg_position[2]:.2f}"
                            # Optionally update confidence
                            updated_itemDF.loc[idx, 'Conf'] = max(confidence, updated_itemDF.loc[idx, 'Conf'])
                            updated = True
                            # break # Stop checking once updated # TODO
                    except Exception as e:
                        print(f"Error processing existing position for {className} at index {idx}: {e}")
                        continue # Skip this entry if parsing fails
            # --- *** END CHECK *** ---

            # If no nearby existing object was found/updated, add as new row
            if not updated:
                new_row_data = {
                    "objectType": [className],
                    "Conf": [confidence],
                    "Position": [f"{x_glob:.2f}, {y_glob:.2f}, {z_glob:.2f}"]
                }
                new_row_df = pd.DataFrame(new_row_data)

                # Use concat, ensuring columns align even if updated_itemDF was initially empty
                updated_itemDF = pd.concat([updated_itemDF, new_row_df], ignore_index=True)


    # Everything in view that was NOT a detection is evidence of empty floor.
    # Without this the map cannot tell "looked here, nothing" from "never
    # looked", and the semantic likelihood assigns both the same prior value.
    if (free_space and beta is not None and robot_pose is not None
            and x_points is not None and z_points is not None):
        rx, rz, ryaw = robot_pose
        update_free_space(beta, x_points, z_points, rx, rz, ryaw,
                          detected_cells=_detected_cells)

    # Ensure essential columns exist before returning, even if no objects detected
    # This prevents errors later if no objects are found in the initial scan
    for col in ['objectType', 'Conf', 'Position']:
         if col not in updated_itemDF.columns:
              updated_itemDF[col] = pd.Series(dtype='object' if col != 'Conf' else 'float')

    return updated_itemDF


def initialize_envKnowledge(controller, model, itemDF, save_path, confThr=0.3, fusionMode=None,
                            beta=None, x_points=None, z_points=None, free_space=True):
    """Initializes the environment knowledge by scanning the surroundings.

    Rotates the agent 360 degrees (4 steps of 90 degrees), calling `visionBranch`
    at each step to populate the `itemDF` DataFrame with detected objects.

    Parameters
    ----------
    controller : ai2thor.controller.Controller
        The AI2-THOR controller instance.
    model : ultralytics.YOLO
        The loaded YOLOv8 model instance.
    itemDF : pd.DataFrame
        An empty DataFrame to be populated with initial object detections.
    probMap : np.ndarray
        The current Bayesian probability map (unused in this function directly,
        but might be intended for later use or passed down).
    x_points : np.ndarray
        1D array of x-coordinates defining the grid columns (unused).
    z_points : np.ndarray
        1D array of z-coordinates defining the grid rows (unused).
    confThr : float, optional
        Confidence threshold for YOLO detections passed to `visionBranch`. Defaults to 0.3.

    Returns
    -------
    pd.DataFrame
        The `itemDF` DataFrame populated with objects detected during the scan.
    """
    current_itemDF = itemDF.copy() # Start with the (presumably empty) DataFrame
    num_rotations = 4 # 360 degrees / 90 degrees per step

    print("Initializing environment knowledge by rotating...")
    for i in range(num_rotations):
        print(f"Rotation step {i+1}/{num_rotations}")
        # Forward fusionMode so visionBranch uses the requested depth method
        md = controller.last_event.metadata["agent"]
        pose = (md["position"]["x"], md["position"]["z"], md["rotation"]["y"])
        current_itemDF = visionBranch(model, current_itemDF, controller, save_path, i+100,
                                      fusionMode=fusionMode, confThr=confThr,
                                      beta=beta, x_points=x_points, z_points=z_points,
                                      robot_pose=pose, free_space=free_space)
        # Convert to dict and back to handle potential duplicate index issues if concat runs oddly
        itemDF_list = current_itemDF.to_dict(orient='records')
        current_itemDF = pd.DataFrame(itemDF_list)
        print(f"Detected items after step {i+1}:")
        print(current_itemDF.head())
        print("---")

        try:
            intialFrame = controller.last_event.cv2img  # AI2-THOR gives frame in BGR
            frame_filename = os.path.join(save_path, f"initializationFrame_{i}.png")
            cv2.imwrite(frame_filename, intialFrame)
        except Exception as e:
            print(f"Warning: Could not save initialization frame {i} to {save_path}: {e}")

        # Rotate for the next view, unless it's the last step
        if i < num_rotations - 1:
            controller.step("RotateLeft", degrees=90) # Use explicit degrees

    print("Finished initialization scan.")
    # The add_goal_similarity call was commented out, keep it that way unless needed here.
    return current_itemDF


def add_goal_similarity(itemDF, goal_phrase, probMap, x_points, z_points, alg_choice='F'):
    """Calculates and adds multimodal similarity scores to the item DataFrame.

    For each object in `itemDF`, calculates:
    - `visionSim`: Based on detection confidence (`Conf` column).
    - `langSim`: Cosine similarity between the object name embedding and the `goal_phrase` embedding.
    - `olfactionSim`: The value from the `probMap` corresponding to the object's grid location.
    - `goalSim`: A combined score based on the `alg_choice`:
        - 'f' (fusion): langSim * olfactionSim
        - 'v' (vision): langSim
        - 'o' (olfaction): olfactionSim

    The DataFrame is then sorted by `goalSim` descending.

    Parameters
    ----------
    itemDF : pd.DataFrame
        DataFrame with detected objects. Requires columns 'objectType', 'Conf', 'Position'.
    goal_phrase : str
        The textual description of the search goal (e.g., "source of smoke odor").
    probMap : np.ndarray
        The current 2D Bayesian belief map.
    x_points : np.ndarray
        1D array of x-coordinates defining the grid columns.
    z_points : np.ndarray
        1D array of z-coordinates defining the grid rows.
    alg_choice : str, optional
        Determines how `goalSim` is calculated ('f', 'v', or 'o'). Defaults to 'f'.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with added similarity columns ('visionSim', 'langSim',
        'olfactionSim', 'goalSim') and sorted by 'goalSim' descending.
    """
    if itemDF.empty:
        print("Warning: itemDF is empty in add_goal_similarity. Returning empty DataFrame.")
        # Ensure the columns exist even if empty
        for col in ["visionSim", "olfactionSim", "langSim", "goalSim"]:
             if col not in itemDF.columns:
                  itemDF[col] = np.nan
        return itemDF

    goal_embedding = model.encode(goal_phrase, convert_to_tensor=True)

    # Initialize columns if they don't exist
    for col in ["visionSim", "olfactionSim", "langSim", "goalSim"]:
        if col not in itemDF.columns:
            itemDF[col] = np.nan

    # Calculate similarities row by row
    for idx, row in itemDF.iterrows():
        object_type = row["objectType"]

        # Vision Similarity (simply confidence)
        itemDF.loc[idx, "visionSim"] = row["Conf"]

        # Language Similarity
        object_embedding = model.encode(object_type, convert_to_tensor=True)
        # Ensure embeddings are on the same device if using GPU
        # goal_embedding = goal_embedding.to(object_embedding.device)
        lang_similarity = util.pytorch_cos_sim(object_embedding, goal_embedding).item()
        itemDF.loc[idx, "langSim"] = lang_similarity

        # Olfaction Similarity
        try:
            pos_str = row["Position"]
            x_world, _, z_world = map(float, pos_str.split(','))
            grid_indices = world_to_grid(x_world, z_world, x_points, z_points)
            grid_row, grid_col = grid_indices[0], grid_indices[1] # Extract row and column

            # Ensure indices are within bounds
            if 0 <= grid_row < probMap.shape[0] and 0 <= grid_col < probMap.shape[1]:
                olf_val = probMap[grid_row, grid_col]
                itemDF.loc[idx, "olfactionSim"] = olf_val
            else:
                 print(f"Warning: Calculated grid indices ({grid_row}, {grid_col}) for object {object_type} at ({x_world:.2f}, {z_world:.2f}) are out of probMap bounds ({probMap.shape}). Setting olfactionSim to 0.")
                 itemDF.loc[idx, "olfactionSim"] = 0.0 # Assign a default value
                 olf_val = 0.0 # Use default for combined calculation
        except Exception as e:
            print(f"Error calculating olfaction similarity for object {object_type} at index {idx}: {e}")
            itemDF.loc[idx, "olfactionSim"] = 0.0 # Assign a default value on error
            olf_val = 0.0

        # Combined Goal Similarity based on alg_choice
        # Use .loc to ensure values are properly assigned back to the DataFrame slice
        vision_sim = itemDF.loc[idx, "visionSim"]
        if alg_choice == "F" or alg_choice == "G":
            combined_sim = lang_similarity * olf_val
        elif alg_choice == "V":
            combined_sim = lang_similarity
        elif alg_choice == "O":
            combined_sim = olf_val
        else: # Default or unknown mode, maybe just use fusion?
            print(f"Warning: Unknown alg_choice '{alg_choice}'. Defaulting to fusion.")
            combined_sim = lang_similarity * olf_val
        itemDF.loc[idx, "goalSim"] = combined_sim

    # Sort by combined similarity (handle potential NaNs by placing them last)
    itemDF.sort_values(by="goalSim", ascending=False, inplace=True, na_position='last')

    # Print the location of the highest belief in the probability map
    if probMap.size > 0: # Check if probMap is not empty
        max_index = np.unravel_index(np.argmax(probMap), probMap.shape)
        # Ensure indices are within bounds before accessing x_points/z_points
        if max_index[1] < len(x_points) and max_index[0] < len(z_points):
             max_x = x_points[max_index[1]]
             max_z = z_points[max_index[0]]
             print(f"Highest olfactory belief map coordinate (grid index {max_index}): world x={max_x:.2f}, z={max_z:.2f}")
        else:
             print(f"Warning: Max belief index {max_index} is out of bounds for x_points/z_points.")
    else:
         print("Warning: probMap is empty, cannot find highest belief coordinate.")


    return itemDF