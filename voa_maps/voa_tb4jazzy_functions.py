#!/usr/bin/env python3
"""
voa_TB4Jazzy_functions.py  --  platform helpers for the TurtleBot4 VAO node.

Same role as sosl_functions.py in your other project: every function here takes
the node and reads its state, so the control file stays ROS plumbing and a
state machine.

What is NOT here: any inference. The belief maps, likelihoods, hypothesis
tracking and fusion all live in voa_functions/ and are used unchanged from the
AI2-THOR experiments. This file only converts between what ROS publishes and
what those modules expect.

FRAME CONVENTION -- read before touching the auditory parts
-----------------------------------------------------------
ROS REP-103 is (x, y) with z up and yaw from +x toward +y.
The algorithm is (x, z) with y up and yaw from +z toward +x.

Mapping z := y makes them consistent, and the yaws relate by

    yaw_alg = (90 - yaw_ros) mod 360          <- ABSOLUTE bearings

but a RELATIVE bearing (a DOA measured from robot forward) needs only

    theta_alg = -theta_rel                    <- RELATIVE bearings

because the two 90 degree offsets cancel. Applying the absolute form to a
relative bearing rotates every DOA by 90 degrees AND mirrors it, and nothing
raises -- the auditory posterior just converges, confidently, on a reflected
location.
"""

import math
import numpy as np
import cv2 as cv

from .voa_functions import visionFunction as vf
from .voa_functions import soundFunctions as sf
from .voa_functions import hypotheses as hy
from .voa_functions.utils import world_to_grid


# ============================================================= MATH / FRAMES

def quaternion_to_yaw(x, y, z, w):
    """Quaternion -> yaw in radians. Same as your sosl_functions version."""
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def yaw_to_quaternion(yaw_rad):
    """Planar rotation -> (qz, qw). qx = qy = 0."""
    return math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)


def ros_yaw_to_alg_deg(yaw_ros_rad):
    """ROS yaw (rad, +x toward +y) -> algorithm yaw (deg, +z toward +x)."""
    return (90.0 - math.degrees(yaw_ros_rad)) % 360.0


def alg_deg_to_ros_yaw(yaw_alg_deg):
    """Inverse of ros_yaw_to_alg_deg."""
    return math.radians((90.0 - float(yaw_alg_deg)) % 360.0)


def doa_to_alg_relative(raw_deg, offset_deg=0.0, ccw=True):
    """Array DOA -> bearing relative to robot forward, algorithm convention.

    `offset_deg` aligns the array's zero with the robot's forward axis (the
    same quantity as CAMERA_MIC_OFFSET_DEG in your assistant controller).
    `ccw` is the sense of rotation; if the array numbers bearings clockwise and
    this is left True, every bearing is mirrored about the forward axis.

    Negation only -- see the frame note at the top of this file.
    """
    rel = float(raw_deg) - float(offset_deg)
    if not ccw:
        rel = -rel
    return (-rel) % 360.0


# ============================================================= DECODING

def decode_rgb(msg):
    """CompressedImage -> BGR array, or None."""
    arr = np.asarray(bytearray(msg.data), dtype="uint8")
    return cv.imdecode(arr, cv.IMREAD_COLOR)


def decode_depth(msg, header_bytes=12):
    """compressedDepth -> uint16 depth image in millimetres, or None.

    compressedDepth prefixes the PNG payload with a header; on this robot it is
    12 bytes, matching the msg.data[12:] slice in turtlebot_subpub_01. Decoding
    without stripping it returns None rather than raising.
    """
    raw = bytes(msg.data)[header_bytes:]
    return cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_UNCHANGED)


# ============================================================= VISION

def boxDepth(x, y, w, h, node):
    """Depth in metres at a bounding box, from node.latest_depth_image.

    Same approach as your sosl_functions.boxDepth: rescale the RGB box into
    depth pixel coordinates, drop zeros (invalid), then take the MODE of the
    survivors. The mode beats the mean here because a box that clips background
    gives a bimodal depth histogram and the mean lands between the object and
    the wall behind it.
    """
    if node.latest_depth_image is None:
        return 0.0
    depthFrame = node.latest_depth_image
    d_h, d_w = depthFrame.shape[:2]
    r_h, r_w = node.latest_rgb_shape[:2]
    if r_h == 0 or r_w == 0:
        return 0.0

    scale_x, scale_y = d_w / r_w, d_h / r_h
    x_d, y_d = int(x * scale_x), int(y * scale_y)
    w_d, h_d = int(w * scale_x), int(h * scale_y)

    vMin, vMax = max(0, y_d - h_d // 2), min(d_h, y_d + h_d // 2)
    hMin, hMax = max(0, x_d - w_d // 2), min(d_w, x_d + w_d // 2)
    if vMin >= vMax or hMin >= hMax:
        return 0.0

    depth_m = depthFrame[vMin:vMax, hMin:hMax].astype(float) / 1000.0
    valid = depth_m[depth_m > 0]
    if valid.size == 0:
        return 0.0
    vals, counts = np.unique(np.round(valid, 2), return_counts=True)
    return float(vals[int(np.argmax(counts))])


def coord23D(x, y, w, h, node):
    """Pixel box -> map (x, y, z) and depth. Returns (x, y, z, d).

    Same chain as your sosl_functions.coord23D: optical frame (X right, Y down,
    Z forward) -> base_link (X forward, Y left) -> map by the robot yaw.

    Returns d = 0.0 when the depth is unusable, so the caller can tell "no
    depth" from a legitimate detection at the origin.
    """
    image_height, image_width = node.latest_rgb_shape[:2]
    focal_h = image_width / (2.0 * np.tan(np.deg2rad(node.cam_hfov_deg) / 2.0))
    focal_v = image_height / (2.0 * np.tan(np.deg2rad(node.cam_vfov_deg) / 2.0))
    center_u, center_v = image_width / 2.0, image_height / 2.0

    d = boxDepth(x, y, w, h, node)
    if d <= 0:
        return 0.0, 0.0, 0.0, 0.0

    x_cam = (x - center_u) / focal_h * d
    y_cam = (y - center_v) / focal_v * d
    z_cam = d

    x_body, y_body, z_body = z_cam, -x_cam, -y_cam

    yaw = quaternion_to_yaw(0, 0, node.robot_map_angZ, node.robot_map_angW)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    x_global = node.robot_map_posX + (x_body * cos_yaw - y_body * sin_yaw)
    y_global = node.robot_map_posY + (x_body * sin_yaw + y_body * cos_yaw)
    z_global = node.cam_height_m + z_body

    return round(x_global, 3), round(y_global, 3), round(z_global, 3), d


def visionBranch(model, node, confThr=0.3):
    """Run YOLO, project detections into the map, update the Dirichlet object map.

    Returns (annotated_frame, n_used, n_rejected).

    Two behaviours worth knowing:

    * Detections beyond node.depth_trust_max_m are REJECTED, not clamped. The
      OAK-D overestimates systematically past roughly 2 m, and a detection
      dropped into the wrong cell is worse for the map than no detection.
    * Every cell in the frustum that did NOT receive a detection accumulates
      evidence for the empty-floor class. Without that, "looked and saw
      nothing" is indistinguishable from "never looked", and the semantic map
      assigns both the same prior value.
    """
    if node.latest_rgb_image is None or node.latest_depth_image is None:
        return None, 0, 0

    results = model(node.latest_rgb_image, verbose=False, conf=confThr)
    annotated = results[0].plot()

    det_cells, n_used, n_rejected = [], 0, 0
    for box in results[0].boxes:
        confidence = float(box.conf[0].item())
        if confidence < confThr:
            continue
        className = model.names[int(box.cls[0].item())]
        x, y, w, h = [float(v) for v in box.xywh[0]]
        x_glob, y_glob, z_glob, d = coord23D(int(x), int(y), int(w), int(h), node)
        if d <= 0.0:
            continue
        if d > node.depth_trust_max_m:
            n_rejected += 1
            continue

        vf.update_object_map(node.objectMap, className, x_glob, y_glob,
                             node.x_points, node.z_points)
        det_cells.append(tuple(int(v) for v in
                               world_to_grid(x_glob, y_glob,
                                             node.x_points, node.z_points)))
        n_used += 1

        cv.putText(annotated, f"{d:.2f}m ({x_glob:.2f},{y_glob:.2f})",
                   (int(x) - 20, int(y) + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 255), 2)

    yaw = quaternion_to_yaw(0, 0, node.robot_map_angZ, node.robot_map_angW)
    vf.update_free_space(node.objectMap, node.x_points, node.z_points,
                         node.robot_map_posX, node.robot_map_posY,
                         ros_yaw_to_alg_deg(yaw),
                         detected_cells=det_cells,
                         fov_deg=node.cam_hfov_deg,
                         evidence=node.free_evidence)
    return annotated, n_used, n_rejected


# ============================================================= OLFACTION

def olfactionBranch(node):
    """Fold one MQ3 reading into the emission-rate hypothesis tracker.

    Returns the baseline-subtracted counts, or None when not ready.

    Two things this depends on:

    * The clean-air baseline must already be set. It is an ADDITIVE offset, so
      unlike the sensor gain it does not cancel in the log-ratio and cannot be
      absorbed by the q_s hypotheses.
    * Wind is measured on this robot, so it is passed through. A plume run at
      U=0 in moving air places the estimated source upwind of the true one, and
      the error grows with wind speed.
    """
    if node.mq3_baseline is None or not node.have_olfaction:
        return None
    counts = max(float(node.olfactionChemicalConc) - node.mq3_baseline, 0.0)
    if counts <= 0.0:
        return counts

    predictor = hy.olfactory_predictor(
        node.gridX, node.gridZ, node.robot_map_posX, node.robot_map_posY,
        U=float(node.olfactionWindSpeed),
        psi_deg=(90.0 - float(node.olfactionWindDirection)) % 360.0)
    node.olfHyp.update(predictor, float(np.log(counts)), node.olf_sigma_log)
    return counts


# ============================================================= AUDITION

def auditionBranch(node):
    """Fold one stationary DOA burst into the auditory belief.

    Returns the number of samples consumed.

    ONE update from the burst mean, not one per sample. Samples inside a burst
    share a reverberation bias and the array's own calibration error, so
    treating 30 of them as independent buys no accuracy while tripling the
    confidence -- which is what makes a fused entropy trigger fire before
    vision has found anything.
    """
    with node.doa_lock:
        burst = list(node.doa_burst)
        node.doa_burst.clear()
    if not burst:
        return 0

    rel = [doa_to_alg_relative(raw, node.doa_offset_deg, node.doa_ccw)
           for raw, _ in burst]
    theta = sf.circular_mean_deg(rel)

    n_eff = max(1.0, node.doa_independent_frac * len(rel))
    sigma_eff = float(np.hypot(node.doa_sigma_deg / math.sqrt(n_eff),
                               node.doa_burst_bias_deg))

    yaw = quaternion_to_yaw(0, 0, node.robot_map_angZ, node.robot_map_angW)
    sf.update_sound_map(node.soundLog, node.x_points, node.z_points,
                        node.robot_map_posX, node.robot_map_posY,
                        ros_yaw_to_alg_deg(yaw), theta, sigma_deg=sigma_eff)

    levels = [lv for _, lv in burst if lv is not None]
    if levels and node.sndHyp is not None:
        node.sndHyp.update(
            hy.auditory_predictor(node.gridX, node.gridZ,
                                  node.robot_map_posX, node.robot_map_posY),
            float(np.mean(levels)), node.snd_sigma_db)

    if node.sound_sim is None:
        try:
            node.sound_sim = sf.class_sound_similarity(('text', node.sound_phrase),
                                                        vf.CLASSES)
        except Exception as e:
            node.get_logger().warn(f"CLAP similarity failed: {e}")
    return len(burst)


# ============================================================= GRID / BELIEF

def build_grid_from_map(node, map_msg, grid_step=0.25):
    """Grid axes, cell centres and a free-space mask from the nav2 map.

    The AI2-THOR equivalent of sceneBounds + GetReachablePositions. Unknown
    (-1) and occupied cells are excluded so belief never lands inside a wall
    and the planner never proposes a goal there.

    Returns True on success; populates node.x_points / z_points / gridX /
    gridZ / free_mask and initialises every belief structure.
    """
    info = map_msg.info
    data = np.asarray(map_msg.data, dtype=np.int8).reshape(info.height, info.width)
    free = (data >= 0) & (data < 50)
    if not free.any():
        return False

    ys, xs = np.where(free)
    x0 = info.origin.position.x + xs.min() * info.resolution
    x1 = info.origin.position.x + xs.max() * info.resolution
    y0 = info.origin.position.y + ys.min() * info.resolution
    y1 = info.origin.position.y + ys.max() * info.resolution

    r = grid_step
    node.x_points = np.arange(np.floor(x0 / r) * r, np.ceil(x1 / r) * r + r, r)
    node.z_points = np.arange(np.floor(y0 / r) * r, np.ceil(y1 / r) * r + r, r)
    node.gridX, node.gridZ = np.meshgrid(node.x_points, node.z_points)

    cols = ((node.gridX - info.origin.position.x) / info.resolution).astype(int)
    rows = ((node.gridZ - info.origin.position.y) / info.resolution).astype(int)
    ok = (rows >= 0) & (rows < info.height) & (cols >= 0) & (cols < info.width)
    mask = np.zeros(node.gridX.shape, dtype=bool)
    mask[ok] = free[rows[ok], cols[ok]]
    if not mask.any():
        return False
    node.free_mask = mask

    shape = mask.shape
    node.objectMap = vf.init_object_map(node.x_points, node.z_points)
    node.soundLog = sf.init_sound_map(node.x_points, node.z_points)
    node.olfHyp = hy.ScaleHypotheses(hy.log_space_grid(node.q_s_hypotheses),
                                     shape, label='q_s') if node.use_O else None
    node.sndHyp = hy.ScaleHypotheses(hy.db_grid(node.l0_hypotheses),
                                     shape, label='L0') if node.use_A else None

    uni = mask.astype(float)
    uni /= uni.sum()
    node.p_vis = node.p_olf = node.p_snd = node.p_fused = uni
    node.H_max = map_entropy(uni)
    return True


def map_entropy(p):
    """Shannon entropy of a belief map, in bits."""
    q = np.asarray(p, float).ravel()
    q = q[q > 1e-12]
    return float(-(q * np.log2(q)).sum())


def mask_normalise(node, p):
    """Zero out non-free cells and renormalise."""
    q = np.asarray(p, float) * node.free_mask
    s = q.sum()
    return q / s if s > 1e-12 else node.free_mask / node.free_mask.sum()


def update_belief(node):
    """Refresh every per-modality map and the log-space fusion.

    Returns the normalised fused entropy, which is the termination statistic.
    """
    if node.use_V:
        node.p_vis = mask_normalise(node, vf.visual_likelihood_multimodal(
            node.objectMap,
            goal_phrase=node.goal_phrase if node.use_O else None,
            sound_sim=node.sound_sim if node.use_A else None))
    if node.use_O and node.olfHyp is not None:
        node.p_olf = mask_normalise(node, node.olfHyp.cell_posterior())
    if node.use_A:
        total = node.soundLog + (node.sndHyp.cell_loglik()
                                 if node.sndHyp is not None else 0.0)
        node.p_snd = mask_normalise(node, sf.sound_posterior(total))

    maps, weights = [], []
    if node.use_V:
        maps.append(node.p_vis); weights.append(node.w_vision)
    if node.use_O:
        maps.append(node.p_olf); weights.append(node.w_olfact)
    if node.use_A:
        maps.append(node.p_snd); weights.append(node.w_sound)
    node.p_fused = mask_normalise(node, vf.fuse_log_evidence(maps, weights))
    return map_entropy(node.p_fused) / max(node.H_max, 1e-9)


def fused_peak(node):
    """(x, y) of the highest-belief cell in the fused map."""
    gi = np.unravel_index(int(np.argmax(node.p_fused)), node.p_fused.shape)
    return float(node.x_points[gi[1]]), float(node.z_points[gi[0]])


def peak_object_name(node):
    """What the object map believes occupies the fused peak.

    Distinguishes a class name, 'empty floor', and 'unobserved'. The last is
    the one that matters: a peak on an unobserved cell means the belief is
    pointing somewhere vision has not checked, which is a reason to keep going.
    """
    try:
        gi = np.unravel_index(int(np.argmax(node.p_fused)), node.p_fused.shape)
        if not vf.observed_mask(node.objectMap)[gi]:
            return "unobserved"
        k = int(np.argmax(vf.object_posterior(node.objectMap)[gi]))
        return vf.CLASSES[k] if k < len(vf.CLASSES) else "?"
    except Exception:
        return "?"


def pick_goal(node):
    """Next nav2 goal: toward the fused peak, capped and snapped to free space.

    Capping keeps each step short enough that the belief is refreshed on the
    way; snapping keeps nav2 from ever being handed a goal inside an obstacle.
    """
    tx, tz = fused_peak(node)
    rx, ry = node.robot_map_posX, node.robot_map_posY
    d = math.hypot(tx - rx, tz - ry)
    heading = math.atan2(tz - ry, tx - rx)
    if d <= node.nav_step_m or d < 1e-6:
        return tx, tz, heading

    f = node.nav_step_m / d
    gx, gy = rx + (tx - rx) * f, ry + (tz - ry) * f
    free_idx = np.where(node.free_mask.ravel())[0]
    fx = node.gridX.ravel()[free_idx]
    fy = node.gridZ.ravel()[free_idx]
    j = int(np.argmin((fx - gx) ** 2 + (fy - gy) ** 2))
    return float(fx[j]), float(fy[j]), heading