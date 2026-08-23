#!/usr/bin/env python3
"""
rosFunctions.py  --  platform helpers shared by every VAO ROS2 robot.

One implementation for the TurtleBot4 (Jazzy) and Robuddy (Humble). The two
robots differ only in DATA -- topic names, frame names, camera intrinsics,
which sensors exist -- and every one of those is read off the node, so there
is nothing to fork.

Contract: each function takes the node and reads its state, so the control
files stay ROS plumbing and a state machine. No inference happens here --
belief maps, likelihoods, hypothesis tracking and fusion all live in
voa_functions/ unchanged from the AI2-THOR experiments.

The node must provide: cam_hfov_deg, cam_vfov_deg, cam_height_m,
depth_trust_max_m, free_evidence, nav_step_m, olf_sigma_log, doa_offset_deg,
doa_ccw, doa_sigma_deg, doa_burst_bias_deg, doa_independent_frac,
w_vision/w_olfact/w_sound, use_V/use_O/use_A, and the belief state built by
build_grid_from_map.

WHY NOT voa_functions/visionFunction.visionBranch
-------------------------------------------------
That one takes an ai2thor Controller and reads controller.last_event.
depth_frame, and its boxDepth reads a float32 metre frame. On hardware the
depth is a uint16 millimetre image at a different resolution from the RGB,
and the pose comes from TF. Same NAME, different input -- these are the ROS
implementations of the same idea, and both call the identical
update_object_map / update_free_space underneath.

FRAME CONVENTION
----------------
ROS REP-103 is (x, y) with z up and yaw from +x toward +y. The algorithm is
(x, z) with y up and yaw from +z toward +x. Mapping z := y makes them
consistent, and the yaws relate by yaw_alg = (90 - yaw_ros) mod 360 for
ABSOLUTE bearings. A RELATIVE bearing (a DOA from robot forward) needs only
negation, because the two 90 degree offsets cancel. Applying the absolute
form to a relative bearing rotates every DOA by 90 degrees AND mirrors it,
and nothing raises.
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


def circular_mean_deg(angles_deg):
    """Mean of angles, computed on the circle.

    An arithmetic mean of 359 and 1 gives 180 -- the opposite direction. Every
    average of bearings has to go through the unit circle.

    Defined here rather than imported so this file does not depend on which
    revision of soundFunctions.py is installed.
    """
    a = np.radians(np.asarray(angles_deg, float))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0)


def doa_to_alg_relative(raw_deg, offset_deg=0.0, ccw=False):
    """Microphone-array DOA -> bearing relative to robot forward, algorithm units.

    Two calibration constants, both of which must be right or the auditory
    branch localises confidently to the wrong place with nothing raised.

    offset_deg
        Where the array's zero points relative to the robot's forward axis. If
        the array is mounted rotated 30 degrees from centre, a source dead
        ahead reads 30, and offset_deg = 30 removes that. Same quantity as
        CAMERA_MIC_OFFSET_DEG in the Robuddy assistant controller. Calibrate by
        putting the source directly in front and reading the raw value.

    ccw
        WHICH WAY THE ARRAY COUNTS. Some arrays number bearings
        counter-clockwise (raw 90 = source on the LEFT), others clockwise
        (raw 90 = source on the RIGHT). Nothing in the reading itself tells you
        which; it is a property of the hardware.

        Get it backwards and every bearing is MIRRORED about the forward axis:
        a source on the right is believed to be on the left, by exactly the
        same angle. Rays from several robot positions still cross -- just at
        the mirror image of the true source -- so the belief looks healthy and
        converges tightly onto the wrong spot.

        For the ReSpeaker XVF3800, ccw=False. Verified against
        soundFunctions.true_bearing_deg for a robot facing ROS +x:

            source ahead   raw   0  ->  alg   0   (geometric   0)
            source right   raw  90  ->  alg  90   (geometric  90)
            source behind  raw 180  ->  alg 180   (geometric 180)
            source left    raw 270  ->  alg 270   (geometric 270)

        This also matches _doa_to_relative_rad in the Robuddy assistant
        controller, which turns +CCW for raw > 180 (speaker on the left).

    HOW TO CHECK ON HARDWARE: put the source clearly to the robot's RIGHT. If
    the auditory belief builds up on the LEFT, flip this flag.

    The conversion itself is a negation, NOT the 90 degree flip used for
    absolute bearings -- for a relative bearing the two 90 degree offsets
    cancel.
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

    Returns (annotated_frame, detections, n_rejected) where `detections` is a
    list of dicts: class_name, conf, depth, x, y, sim. `sim` is the cosine
    similarity between that class name and the goal phrase -- the number that
    decides how much the detection actually contributes to the visual belief.

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

    # Per-class similarity against the goal phrase. Cached inside
    # class_goal_similarity, so this costs one SBERT encode for the whole run.
    sim_vec = None
    try:
        if getattr(node, 'goal_phrase', None):
            sim_vec = vf.class_goal_similarity(node.goal_phrase)
    except Exception as e:
        node.get_logger().warn(f"similarity lookup failed: {e}")

    det_cells, detections, n_rejected, unknown = [], [], 0, set()
    for box in results[0].boxes:
        confidence = float(box.conf[0].item())
        if confidence < confThr:
            continue
        className = model.names[int(box.cls[0].item())]
        # A class the detector reports but the object map does not know falls
        # straight through update_object_map's CLS_IDX lookup and contributes
        # NOTHING, silently. That is exactly what happens when yolo_path points
        # at a stock COCO checkpoint while LAB_CLASSES lists the fine-tuned
        # lab classes, so it is worth naming rather than swallowing.
        if className not in vf.CLS_IDX:
            unknown.add(className)
            continue
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
        detections.append(dict(
            class_name=className, conf=confidence, depth=d,
            x=x_glob, y=y_glob,
            sim=(float(sim_vec[vf.CLS_IDX[className]])
                 if sim_vec is not None else float('nan'))))

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
    if unknown:
        node.get_logger().warn(
            f"detector reported {sorted(unknown)} which are NOT in the object "
            f"map classes {vf.CLASSES[:-1]} -- these contribute nothing. "
            f"Check yolo_path matches LAB_CLASSES.")
    return annotated, detections, n_rejected


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
    counts = max(float(node.mq3_counts) - node.mq3_baseline, 0.0)
    if counts <= 0.0:
        return counts

    # NO WIND TERM. Neither robot carries an anemometer, and both the AI2-THOR
    # experiments and the lab runs assume still air, so the plume is pure
    # diffusion (U = 0). That is the same model the simulation was validated
    # against, which keeps sim and hardware comparable.
    #
    # If you later run in a draught, this is where it breaks: a plume evaluated
    # at U=0 in moving air places the estimated source UPWIND of the true one,
    # and the error grows with wind speed. Fit an anemometer and pass
    # U=wind_speed, psi_deg=(90 - wind_direction) % 360 here.
    predictor = hy.olfactory_predictor(
        node.gridX, node.gridZ, node.robot_map_posX, node.robot_map_posY,
        U=0.0, psi_deg=0.0)
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

    # A burst entry is either a bare bearing or (bearing, level_db), depending
    # on whether the array reports level. The XVF3800 does not.
    pairs = [(b if isinstance(b, (tuple, list)) else (b, None)) for b in burst]
    rel = [doa_to_alg_relative(raw, node.doa_offset_deg, node.doa_ccw)
           for raw, _ in pairs]
    theta = circular_mean_deg(rel)

    n_eff = max(1.0, node.doa_independent_frac * len(rel))
    sigma_eff = float(np.hypot(node.doa_sigma_deg / math.sqrt(n_eff),
                               node.doa_burst_bias_deg))

    yaw = quaternion_to_yaw(0, 0, node.robot_map_angZ, node.robot_map_angW)
    sf.update_sound_map(node.soundLog, node.x_points, node.z_points,
                        node.robot_map_posX, node.robot_map_posY,
                        ros_yaw_to_alg_deg(yaw), theta, sigma_deg=sigma_eff)

    levels = [lv for _, lv in pairs if lv is not None]
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
    # The range hypothesis needs a LEVEL. An array that reports bearing only
    # (the XVF3800) leaves this None, and audition is a ray rather than a spot
    # until the robot moves and rays cross.
    node.sndHyp = (hy.ScaleHypotheses(hy.db_grid(node.l0_hypotheses),
                                      shape, label='L0')
                   if (node.use_A and getattr(node, 'has_sound_level', False))
                   else None)

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
        gp = node.goal_phrase if node.use_O else None
        ss = node.sound_sim if node.use_A else None
        if gp is None and ss is None:
            # VA mode before the first sound has been heard: olfaction is off
            # so there is no odour phrase, and CLAP has not produced a sound
            # similarity vector yet (it only runs after the first successful
            # DOA burst, and falls back to SBERT if laion_clap is missing).
            # With both absent, combine_similarity has nothing to work with
            # and raises -- which crashed the very first search step.
            #
            # Fall back to the sound phrase as plain TEXT. It describes the
            # same target CLAP would have embedded, so vision still has a
            # semantic goal to project the object map onto instead of the run
            # dying before it starts.
            gp = getattr(node, 'sound_phrase', None) or node.goal_phrase
        node.p_vis = mask_normalise(node, vf.visual_likelihood_multimodal(
            node.objectMap, goal_phrase=gp, sound_sim=ss))
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


# ============================================================= RENDERING
#
# Everything below writes PNGs for a run. Kept here rather than in either node
# so both robots produce identically-formatted output, and so the simulation's
# figures and the hardware figures can be read the same way.

import os
import matplotlib
matplotlib.use('Agg')          # no display on a robot; must be set before pyplot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe


def _peak_of(m, x_points, z_points):
    """(x, z, probability) of the highest cell in a belief map."""
    gi = np.unravel_index(int(np.argmax(m)), m.shape)
    return float(x_points[gi[1]]), float(z_points[gi[0]]), float(m[gi])


def _cell_object_name(node, row, col):
    """What the object map believes occupies one cell.

    Three distinguishable answers, and the third is the one that matters:
      - a class name       : a detection has landed here
      - 'empty floor'      : looked at, nothing found
      - 'unobserved'       : no evidence of any kind has reached this cell yet
    A belief peak on an unobserved cell means the estimate is pointing
    somewhere vision has never actually checked.
    """
    if node.objectMap is None:
        return "?"
    try:
        if not vf.observed_mask(node.objectMap)[row, col]:
            return "unobserved"
        k = int(np.argmax(vf.object_posterior(node.objectMap)[row, col]))
        return vf.CLASSES[k] if k < len(vf.CLASSES) else "?"
    except Exception:
        return "?"


def _draw_trail(ax, node):
    """Straight lines through every pose the robot has occupied, plus a marker
    at the current one. Shows the actual path taken, not just where it ended.
    """
    trail = getattr(node, 'trail', None)
    if not trail:
        return
    t = np.asarray(trail, dtype=float)
    ax.plot(t[:, 0], t[:, 1], '-', lw=1.6, color='#1f77b4', alpha=0.9, zorder=4)
    ax.plot(t[:, 0], t[:, 1], '.', ms=3, color='#1f77b4', alpha=0.7, zorder=4)
    ax.plot(t[-1, 0], t[-1, 1], marker='o', ms=9, mfc='#1f77b4', mec='white',
            mew=1.5, ls='', zorder=6, label='robot')


def _label_peak(ax, node, arr, x_points, z_points, name_it):
    """Circle the map's own argmax; optionally name the object believed there."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo + 1e-12:
        return                      # flat map: argmax would be an arbitrary corner
    px, pz, pv = _peak_of(arr, x_points, z_points)
    ax.plot(px, pz, marker='o', ms=15, mfc='none', mec='#39ff14', mew=2.2,
            ls='', zorder=7)
    ax.plot(px, pz, marker='+', ms=8, mec='#39ff14', mew=1.4, ls='', zorder=7)

    if name_it:
        gi = np.unravel_index(int(np.argmax(arr)), arr.shape)
        label = f"{_cell_object_name(node, gi[0], gi[1])}\n({px:.2f}, {pz:.2f})"
    else:
        label = f"({px:.2f}, {pz:.2f})\np={pv:.3f}"

    xmid = 0.5 * (float(np.min(x_points)) + float(np.max(x_points)))
    zmid = 0.5 * (float(np.min(z_points)) + float(np.max(z_points)))
    dx = -12 if px > xmid else 12
    dz = -14 if pz > zmid else 12
    ax.annotate(label, xy=(px, pz), xytext=(dx, dz), textcoords='offset points',
                ha='right' if dx < 0 else 'left',
                va='top' if dz < 0 else 'bottom',
                fontsize=7, color='#39ff14', fontweight='bold',
                annotation_clip=False, zorder=8,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='black')])


def save_belief_maps(node, step, prefix='maps'):
    """One PNG per step: only the modalities this run actually uses, plus fused.

    A disabled branch holds a uniform map, and rendering it produces a flat
    panel that reads as a broken sensor rather than a switched-off one -- so a
    VO run gets 3 panels (olfactory, visual, fused) and a VA run gets 3
    (visual, auditory, fused), never a blank column.

    Non-free cells are drawn as blank rather than as zero, so the room's real
    shape is visible instead of a rectangle with a dark border.
    """
    try:
        xp, zp, free = node.x_points, node.z_points, node.free_mask
        panels = []
        if node.use_O and node.p_olf is not None:
            panels.append(('Olfactory', node.p_olf, False))
        if node.use_V and node.p_vis is not None:
            panels.append(('Visual (semantic)', node.p_vis, True))
        if node.use_A and node.p_snd is not None:
            panels.append(('Auditory (DOA)', node.p_snd, False))
        panels.append(('Fused', node.p_fused, True))

        extent = [float(np.min(xp)), float(np.max(xp)),
                  float(np.min(zp)), float(np.max(zp))]
        fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.6),
                                 squeeze=False)
        axes = axes[0]

        for ax, (title, arr, name_peak) in zip(axes, panels):
            a = np.asarray(arr, float)
            lo, hi = a.min(), a.max()
            norm = (a - lo) / (hi - lo) if hi > lo + 1e-12 else np.zeros_like(a)
            shown = np.where(free, norm, np.nan)     # NaN renders as blank
            im = ax.imshow(shown, origin='lower', extent=extent, cmap='hot',
                           aspect='equal', vmin=0.0, vmax=1.0)
            ax.set_title(f"{title}  H={map_entropy(a):.2f}", fontsize=11)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.grid(True, ls='--', lw=0.4, alpha=0.4)
            _draw_trail(ax, node)
            _label_peak(ax, node, a, xp, zp, name_peak)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"{node.phase}  step {step}  [{node.modalities}]", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(node.save_dir, f"{prefix}_{step:03d}.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        node.get_logger().warn(f"belief map render failed at step {step}: {e}")
        plt.close('all')


def save_object_map(node, step, prefix='objects'):
    """Flat semantic map of the environment: the most likely class per cell.

    This is the raw object map, NOT a belief about the source -- it answers
    "what has the robot seen and where", which is the thing that makes a
    wrong fused peak interpretable after the fact.
    """
    try:
        if node.objectMap is None:
            return
        xp, zp, free = node.x_points, node.z_points, node.free_mask
        observed = vf.observed_mask(node.objectMap)
        mle = np.argmax(vf.object_posterior(node.objectMap), axis=2)

        # Only label classes that actually appear, so the legend stays short
        # even with an 80-class COCO detector.
        shown_cells = observed & free
        present = sorted(set(int(k) for k in np.unique(mle[shown_cells]))) \
            if shown_cells.any() else []
        if not present:
            return
        remap = {k: i for i, k in enumerate(present)}
        img = np.full(mle.shape, np.nan)
        for k, i in remap.items():
            img[shown_cells & (mle == k)] = i

        extent = [float(np.min(xp)), float(np.max(xp)),
                  float(np.min(zp)), float(np.max(zp))]
        n = len(present)
        cmap = plt.get_cmap('tab20', max(n, 2))
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        im = ax.imshow(img, origin='lower', extent=extent, cmap=cmap,
                       vmin=-0.5, vmax=n - 0.5, aspect='equal')
        ax.set_title(f"object map -- {node.phase} step {step}", fontsize=12)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True, ls='--', lw=0.4, alpha=0.4)
        _draw_trail(ax, node)

        cb = fig.colorbar(im, ax=ax, ticks=range(n), fraction=0.046, pad=0.04)
        cb.ax.set_yticklabels([vf.CLASSES[k] if k < len(vf.CLASSES) else '?'
                               for k in present], fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(node.save_dir, f"{prefix}_{step:03d}.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        node.get_logger().warn(f"object map render failed at step {step}: {e}")
        plt.close('all')


def belief_arrays_for_saving(node):
    """Only the belief arrays this run's modalities actually produced.

    Saving a uniform placeholder for a disabled branch wastes space and, worse,
    reads later as "this modality ran and learned nothing" rather than "this
    modality was switched off".
    """
    out = dict(fused=node.p_fused, free=node.free_mask,
               x_points=node.x_points, z_points=node.z_points)
    if node.use_V:
        out['vision'] = node.p_vis
    if node.use_O:
        out['olfaction'] = node.p_olf
    if node.use_A:
        out['sound'] = node.p_snd
    return out