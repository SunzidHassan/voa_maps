#!/usr/bin/env python3
"""
voa_TB4Jazzy_node.py  --  VAO source localisation on TurtleBot4 Pro (ROS2 Jazzy).

    ros2 run voa_maps tb4_node
    ros2 run voa_maps tb4_node --ros-args -p goal_phrase:="rotten food smell"

Control only: ROS interfaces, callbacks, and the state machine. All the sensor
maths lives in voa_TB4Jazzy_functions.py, and all the inference lives in
voa_functions/ unchanged from the AI2-THOR runs.

Topics, frames and the compressedDepth header offset are taken from
turtlebot_subpub_01. Note this robot publishes map -> base_footprint, not
base_link.

CYCLE
-----
    SENSE -> [LISTEN] -> FUSE -> NAVIGATE -> SENSE ...

nav2 goals are asynchronous, so vision and olfaction keep sampling at
SENSE_PERIOD_S while the base drives. Audition cannot: a moving base is the
loudest thing in the room from the microphone's point of view, and drive
motors sit right in the band the array uses for DOA. Worse, ego-noise raises
the measured level, which the range hypothesis reads as "the source is closer
than it is" -- a systematic error that does not average out the way bearing
jitter does. So the LISTEN phase halts the base, waits for spin-down, and only
then opens the microphone.

BEFORE THE FIRST RUN
--------------------
1. Start this node BEFORE opening the odour source. The first
   MQ3_BASELINE_SAMPLES readings are taken as clean air; if the plume is
   already present the baseline absorbs it and olfaction contributes nothing.
2. Confirm map -> base_footprint exists, i.e. nav2 is running with a map and
   not just odometry. A belief accumulated in odom smears as the run goes on.
3. If a microphone array is fitted, calibrate DOA_OFFSET_DEG and confirm
   DOA_CCW before trusting any auditory result.
"""

import os
import json
import math
import time
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import cv2 as cv

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy, QoSHistoryPolicy)

from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import CompressedImage, LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ultralytics import YOLO

from .voa_functions import visionFunction as vf
from . import voa_tb4jazzy_functions as tb4

# ============================================================= CONFIG

# Lab detector classes. MUST match model.names exactly, or every detection
# falls through the CLS_IDX lookup and the object map silently never moves.
LAB_CLASSES = ['Cardboard box', 'Coke can', 'First aid box',
               'Humidifier', 'Humidifier box', 'Water bottle']

# --- topics / frames ---
RGB_TOPIC = '/oakd/rgb/image_raw/compressed'
DEPTH_TOPIC = '/oakd/stereo/image_raw/compressedDepth'
DEPTH_HEADER_BYTES = 12
OLFACTION_TOPIC = '/olfaction'      # Vector3: x=wind dir, y=wind speed, z=counts
SCAN_TOPIC = '/scan'
ODOM_TOPIC = '/odom'
MAP_TOPIC = '/map'
CMD_VEL_TOPIC = '/cmd_vel'
DOA_TOPIC = '/respeaker/doa'        # Vector3: x=deg, y=level dB, z=activity
MAP_FRAME = 'map'
BASE_FRAME = 'base_footprint'

# nav2's map_server LATCHES /map: it publishes once at startup with
# TRANSIENT_LOCAL durability. A default (VOLATILE) subscription is an
# incompatible QoS match -- it is created without any error, `ros2 topic list`
# shows the topic, and the callback simply never fires, because the message was
# published before this node existed and VOLATILE subscribers are not offered
# the cached sample. Every other subscriber on this robot (local_costmap,
# global_costmap, amcl, rviz2) uses TRANSIENT_LOCAL for exactly this reason.
MAP_QOS = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

# --- OAK-D Pro ---
CAM_HFOV_DEG = 66.0
CAM_VFOV_DEG = 54.0
CAM_HEIGHT_M = 0.25
DEPTH_TRUST_MAX_M = 2.0

# --- grid / planning ---
GRID_STEP = 0.25
NAV_STEP_M = 0.75
NAV_TIMEOUT_S = 45.0
SETTLE_S = 1.5
SENSE_PERIOD_S = 1.0
LISTEN_S = 3.0
EGO_NOISE_VEL = 0.02       # m/s or rad/s above which a DOA sample is discarded

# --- audition (TB4 Pro has no mic array by default -> run is VO) ---
ENABLE_AUDITION = False
DOA_OFFSET_DEG = 0.0
DOA_CCW = True
DOA_SIGMA_DEG = 15.0
DOA_BURST_BIAS_DEG = 10.0
DOA_INDEPENDENT_FRAC = 0.2

# --- sensor noise, already inflated (see SIGMA_INFLATION in fusion_controller) ---
OLF_SIGMA_LOG = 0.75
SND_SIGMA_DB = 7.5
MQ3_BASELINE_SAMPLES = 40

# --- source-strength hypotheses, tracked rather than assumed ---
Q_S_HYPOTHESES = tuple(10.0 ** np.linspace(3, np.log10(5000), 5))
L0_HYPOTHESES = tuple(np.linspace(45.0, 105.0, 5))

# --- fusion / termination ---
W_VISION, W_OLFACT, W_SOUND = 1.0, 0.5, 0.5
ENTROPY_FRAC = 0.7
MAX_STEPS = 40
FREE_EVIDENCE = 3.0


class VAOTurtleBot4(Node):

    def __init__(self):
        super().__init__('voa_tb4_node')

        self.declare_parameter('goal_phrase', 'rotten food smell')
        self.declare_parameter('sound_phrase', 'an alarm clock ringing')
        self.declare_parameter('yolo_path', 'voa_maps/models/YOLO/yolo26m.pt')
        self.declare_parameter('save_dir', '')
        self.declare_parameter('enable_audition', ENABLE_AUDITION)
        self.declare_parameter('entropy_frac', ENTROPY_FRAC)
        gp = lambda n: self.get_parameter(n).value

        self.goal_phrase = f"Is emitting {gp('goal_phrase')} odor:"
        self.sound_phrase = gp('sound_phrase')
        self.entropy_frac = float(gp('entropy_frac'))
        self.use_V, self.use_O = True, True
        self.use_A = bool(gp('enable_audition'))
        self.modalities = ''.join(c for c, on in
                                  (('V', self.use_V), ('A', self.use_A),
                                   ('O', self.use_O)) if on)

        # --- config exposed to the helpers via node state ---
        self.cam_hfov_deg, self.cam_vfov_deg = CAM_HFOV_DEG, CAM_VFOV_DEG
        self.cam_height_m = CAM_HEIGHT_M
        self.depth_trust_max_m = DEPTH_TRUST_MAX_M
        self.free_evidence = FREE_EVIDENCE
        self.nav_step_m = NAV_STEP_M
        self.olf_sigma_log, self.snd_sigma_db = OLF_SIGMA_LOG, SND_SIGMA_DB
        self.q_s_hypotheses, self.l0_hypotheses = Q_S_HYPOTHESES, L0_HYPOTHESES
        self.w_vision, self.w_olfact, self.w_sound = W_VISION, W_OLFACT, W_SOUND
        self.doa_offset_deg, self.doa_ccw = DOA_OFFSET_DEG, DOA_CCW
        self.doa_sigma_deg = DOA_SIGMA_DEG
        self.doa_burst_bias_deg = DOA_BURST_BIAS_DEG
        self.doa_independent_frac = DOA_INDEPENDENT_FRAC

        stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.save_dir = gp('save_dir') or os.path.join(
            os.getcwd(), f'voa_maps/save/TB4/voa_run_{stamp}')
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"modalities={self.modalities}  save_dir={self.save_dir}")

        vf.configure_classes(LAB_CLASSES)
        self.get_logger().info(f"object-map classes: {vf.CLASSES}")

        try:
            self.yoloModel = YOLO(gp('yolo_path'))
            self.get_logger().info("YOLO loaded")
        except Exception as e:
            self.get_logger().error(f"YOLO load failed: {e}")
            raise

        # --- ROS interfaces ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.create_subscription(CompressedImage, RGB_TOPIC,
                                 self.image_callback, qos_profile_sensor_data)
        self.create_subscription(CompressedImage, DEPTH_TOPIC,
                                 self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(Vector3, OLFACTION_TOPIC, self.olfactory_callback, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.laser_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, MAP_TOPIC, self.map_callback, MAP_QOS)
        if self.use_A:
            self.create_subscription(Vector3, DOA_TOPIC, self.doa_callback, 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # --- sensor state (names match sosl_functions expectations) ---
        self.latest_rgb_image = None
        self.latest_rgb_shape = (0, 0, 0)
        self.latest_rgb_stamp = None
        self.latest_depth_image = None
        self.laser = None
        self.map_msg = None
        self.robot_map_posX = 0.0
        self.robot_map_posY = 0.0
        self.robot_map_angZ = 0.0
        self.robot_map_angW = 1.0
        self.olfactionWindDirection = 0.0
        self.olfactionWindSpeed = 0.0
        self.olfactionChemicalConc = 0.0
        self.have_olfaction = False
        self.mq3_baseline = None
        self._baseline_buf = []
        self.lin_vel = 0.0
        self.ang_vel = 0.0

        self.doa_lock = threading.Lock()
        self.doa_burst = []
        self.listening = False

        # --- belief state ---
        self.grid_ready = False
        self.x_points = self.z_points = None
        self.gridX = self.gridZ = None
        self.free_mask = None
        self.objectMap = self.soundLog = None
        self.olfHyp = self.sndHyp = None
        self.sound_sim = None
        self.p_vis = self.p_olf = self.p_snd = self.p_fused = None
        self.H_max = 1.0
        self.n_depth_rejected = 0

        # --- loop state ---
        self.state = 'WAIT_MAP'
        self.step = 0
        self.rows = []
        self.t0 = time.time()
        self.nav_active = False
        self.nav_deadline = 0.0
        self.phase_until = 0.0
        self.listen_from = 0.0
        self.last_sense = 0.0
        self._pending = {}

        self.create_timer(0.1, self.control_callback)

    # ---------------------------------------------------------------- callbacks
    def image_callback(self, msg):
        img = tb4.decode_rgb(msg)
        if img is None:
            self.get_logger().error("Failed to decode RGB image")
            return
        self.latest_rgb_image = img
        self.latest_rgb_shape = img.shape
        self.latest_rgb_stamp = msg.header.stamp

    def depth_callback(self, msg):
        img = tb4.decode_depth(msg, DEPTH_HEADER_BYTES)
        if img is None:
            self.get_logger().error("Failed to decode Depth image")
            return
        self.latest_depth_image = img

    def laser_callback(self, msg):
        self.laser = msg

    def odom_callback(self, msg):
        self.lin_vel = abs(msg.twist.twist.linear.x)
        self.ang_vel = abs(msg.twist.twist.angular.z)

    def map_callback(self, msg):
        self.map_msg = msg

    def olfactory_callback(self, msg):
        # x = wind direction, y = wind speed, z = chemical concentration
        self.olfactionWindDirection = float(msg.x)
        self.olfactionWindSpeed = float(msg.y)
        self.olfactionChemicalConc = float(msg.z)
        self.have_olfaction = True

    def doa_callback(self, msg):
        """Collect a DOA sample, but only when the base is genuinely stopped.

        Dropped here rather than filtered downstream, so ego-noise never
        reaches the belief at all. The velocity check catches residual motion
        during the settle window and anything nav2 issues unexpectedly.
        """
        if not self.listening:
            return
        if self.lin_vel > EGO_NOISE_VEL or self.ang_vel > EGO_NOISE_VEL:
            return
        with self.doa_lock:
            self.doa_burst.append((float(msg.x), float(msg.y)))

    # ---------------------------------------------------------------- pose
    def update_pose(self, stamp=None):
        """Refresh robot_map_* from TF. Returns True when a pose is available.

        With a stamp, the transform is looked up AT THAT TIME. A camera frame
        processed while nav2 is still driving was captured at a pose the robot
        has already left; using the latest transform instead projects every
        detection into the wrong cell.
        """
        try:
            when = rclpy.time.Time() if stamp is None else rclpy.time.Time.from_msg(stamp)
            t = self.tf_buffer.lookup_transform(MAP_FRAME, BASE_FRAME, when)
        except TransformException:
            try:
                t = self.tf_buffer.lookup_transform(MAP_FRAME, BASE_FRAME,
                                                    rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().warn(f'no {MAP_FRAME}->{BASE_FRAME}: {ex}',
                                       throttle_duration_sec=5.0)
                return False
        tr, r = t.transform.translation, t.transform.rotation
        self.robot_map_posX = float(tr.x)
        self.robot_map_posY = float(tr.y)
        self.robot_map_angZ = float(r.z)
        self.robot_map_angW = float(r.w)
        return True

    def halt(self):
        self.cmd_vel_pub.publish(Twist())

    # ---------------------------------------------------------------- nav2
    def send_nav_goal(self, gx, gy, yaw):
        if not self.nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("nav2 navigate_to_pose unavailable")
            return False
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = MAP_FRAME
        g.pose.header.stamp = self.get_clock().now().to_msg()
        g.pose.pose.position.x = float(gx)
        g.pose.pose.position.y = float(gy)
        qz, qw = tb4.yaw_to_quaternion(yaw)
        g.pose.pose.orientation.z = qz
        g.pose.pose.orientation.w = qw
        self.nav_active = True
        self.nav_deadline = time.time() + NAV_TIMEOUT_S
        self.nav_client.send_goal_async(g).add_done_callback(self._goal_response)
        return True

    def _goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("nav2 rejected the goal")
            self.nav_active = False
            return
        handle.get_result_async().add_done_callback(
            lambda f: setattr(self, 'nav_active', False))

    # ---------------------------------------------------------------- loop
    def control_callback(self):
        now = time.time()

        if self.state == 'WAIT_MAP':
            if self.map_msg is None:
                self.get_logger().info(f"waiting for {MAP_TOPIC}...",
                                       throttle_duration_sec=5.0)
                return
            if not tb4.build_grid_from_map(self, self.map_msg, GRID_STEP):
                self.get_logger().warn("map has no free space yet",
                                       throttle_duration_sec=5.0)
                return
            self.grid_ready = True
            self.get_logger().info(
                f"grid {self.free_mask.shape[0]}x{self.free_mask.shape[1]} "
                f"({self.free_mask.size} cells, {int(self.free_mask.sum())} free), "
                f"H_max {self.H_max:.2f} bits")
            self.state = 'WAIT_SENSORS'
            return

        if self.state == 'WAIT_SENSORS':
            # Block until RGB and depth have BOTH arrived. Without this the
            # first steps run with visionBranch returning nothing, the object
            # map stays at its prior, and the run looks like vision is broken
            # when it is really just not connected yet.
            missing = []
            if self.latest_rgb_image is None:
                missing.append(RGB_TOPIC)
            if self.latest_depth_image is None:
                missing.append(DEPTH_TOPIC)
            if missing:
                self.get_logger().warn(
                    f"waiting for camera: no data yet on {', '.join(missing)}",
                    throttle_duration_sec=5.0)
                return
            self.get_logger().info(
                f"camera OK -- RGB {self.latest_rgb_shape[1]}x{self.latest_rgb_shape[0]}, "
                f"depth {self.latest_depth_image.shape[1]}x{self.latest_depth_image.shape[0]} "
                f"({self.latest_depth_image.dtype})")
            self.state = 'BASELINE' if self.use_O else 'SENSE'
            return

        if self.state == 'BASELINE':
            if not self.have_olfaction:
                self.get_logger().info(f"waiting for {OLFACTION_TOPIC}...",
                                       throttle_duration_sec=5.0)
                return
            self._baseline_buf.append(self.olfactionChemicalConc)
            if len(self._baseline_buf) >= MQ3_BASELINE_SAMPLES:
                self.mq3_baseline = float(np.median(self._baseline_buf))
                self.get_logger().info(
                    f"MQ3 clean-air baseline {self.mq3_baseline:.1f} counts "
                    f"from {len(self._baseline_buf)} samples")
                self.state = 'SENSE'
            return

        if self.state == 'SENSE':
            if not self.update_pose(self.latest_rgb_stamp):
                return
            dets, n_rej = [], 0
            if self.use_V:
                annotated, dets, n_rej = tb4.visionBranch(self.yoloModel, self)
                self.n_depth_rejected += n_rej
                if annotated is not None:
                    cv.imwrite(os.path.join(self.save_dir,
                                            f"yolo_{self.step:03d}.jpg"), annotated)
            counts = tb4.olfactionBranch(self) if self.use_O else None
            self.report_sensing(dets, n_rej, counts)
            self._pending = dict(n_det=len(dets), n_rej=n_rej, counts=counts,
                                 n_doa=0, dets=dets,
                                 rx=self.robot_map_posX, ry=self.robot_map_posY)

            if self.use_A:
                self.halt()
                self.listening = False          # stays shut until spin-down
                self.listen_from = now + SETTLE_S
                self.phase_until = now + SETTLE_S + LISTEN_S
                self.state = 'LISTEN'
            else:
                self.state = 'FUSE'
            return

        if self.state == 'LISTEN':
            if now >= self.listen_from:
                self.listening = True
            if now < self.phase_until:
                self.halt()
                return
            self.listening = False
            self._pending['n_doa'] = tb4.auditionBranch(self)
            self.state = 'FUSE'
            return

        if self.state == 'FUSE':
            trig = tb4.update_belief(self)
            p = self._pending
            self.log_step(p, trig)
            self.get_logger().info(
                f"step {self.step}  det={p['n_det']} (rej {p['n_rej']}) "
                f"doa={p['n_doa']} mq3={p['counts']}  "
                f"trigger {trig:.3f} (stop at <= {self.entropy_frac})")
            if self.olfHyp is not None and self.olfHyp.at_endpoint():
                self.get_logger().warn(
                    "q_s posterior pegged at a grid edge -- the true emission "
                    "rate is probably outside Q_S_HYPOTHESES")

            if trig <= self.entropy_frac or self.step >= MAX_STEPS:
                tx, tz = tb4.fused_peak(self)
                why = 'converged' if trig <= self.entropy_frac else 'step limit'
                self.get_logger().info(
                    f"{why}; peak is {tb4.peak_object_name(self)} at "
                    f"({tx:.2f}, {tz:.2f}); driving there and terminating")
                self.send_nav_goal(tx, tz,
                                   math.atan2(tz - p['ry'], tx - p['rx']))
                self.state = 'FINISH'
                return

            gx, gy, gyaw = tb4.pick_goal(self)
            self.state = 'NAVIGATE' if self.send_nav_goal(gx, gy, gyaw) else 'FINISH'
            return

        if self.state == 'NAVIGATE':
            if self.nav_active and now < self.nav_deadline:
                # Vision and olfaction keep sampling while driving; only the
                # microphone is blocked by ego-noise.
                if now - self.last_sense >= SENSE_PERIOD_S:
                    self.last_sense = now
                    if self.update_pose(self.latest_rgb_stamp):
                        if self.use_V:
                            _, _d, rej = tb4.visionBranch(self.yoloModel, self)
                            self.n_depth_rejected += rej
                        if self.use_O:
                            tb4.olfactionBranch(self)
                return
            if self.nav_active:
                self.get_logger().warn("nav goal timed out; continuing")
                self.nav_active = False
            self.halt()
            self.step += 1
            self.state = 'SENSE'
            return

        if self.state == 'FINISH':
            if self.nav_active and now < self.nav_deadline:
                return
            self.halt()
            self.save_data()
            self.state = 'DONE'
            return

    # ---------------------------------------------------------------- reporting
    def report_sensing(self, dets, n_rej, counts):
        """Print what the sensors actually returned this step.

        The similarity column is the one to watch: a detection with a high
        confidence but a near-zero sim contributes almost nothing to the visual
        belief, because the semantic map is the class posterior projected onto
        exactly these numbers. A run where every sim is ~0 means the goal
        phrase and the class names are not meeting, and vision is effectively
        just marking free space.
        """
        yaw = tb4.quaternion_to_yaw(0, 0, self.robot_map_angZ, self.robot_map_angW)
        self.get_logger().info(
            f"--- step {self.step} @ ({self.robot_map_posX:.2f}, "
            f"{self.robot_map_posY:.2f}) yaw {math.degrees(yaw):.0f} deg ---")

        if self.use_O:
            if counts is None:
                self.get_logger().info(
                    f"  olfaction  raw {self.olfactionChemicalConc:.1f} counts, "
                    f"baseline not set yet")
            else:
                self.get_logger().info(
                    f"  olfaction  raw {self.olfactionChemicalConc:.1f} - "
                    f"baseline {self.mq3_baseline:.1f} = {counts:.1f} counts   "
                    f"wind {self.olfactionWindSpeed:.2f} m/s @ "
                    f"{self.olfactionWindDirection:.0f} deg")

        if not self.use_V:
            return
        if not dets:
            self.get_logger().info(
                f"  vision     no usable detections"
                f"{f' ({n_rej} rejected past {DEPTH_TRUST_MAX_M} m)' if n_rej else ''}")
            return

        self.get_logger().info(
            f"  vision     {len(dets)} detection(s)"
            f"{f', {n_rej} rejected past {DEPTH_TRUST_MAX_M} m' if n_rej else ''}")
        self.get_logger().info(
            f"      {'class':<16}{'conf':>6}{'depth':>8}{'map x':>9}{'map y':>9}{'sim':>8}")
        for d in dets:
            self.get_logger().info(
                f"      {d['class_name'][:15]:<16}{d['conf']:>6.2f}"
                f"{d['depth']:>7.2f}m{d['x']:>9.2f}{d['y']:>9.2f}{d['sim']:>8.3f}")

    # ---------------------------------------------------------------- output
    def log_step(self, p, trig):
        yaw = tb4.quaternion_to_yaw(0, 0, self.robot_map_angZ, self.robot_map_angW)
        self.rows.append(dict(
            step=self.step, time=round(time.time() - self.t0, 2),
            robot_x=self.robot_map_posX, robot_y=self.robot_map_posY,
            robot_yaw=round(math.degrees(yaw), 1),
            n_detections=p['n_det'], n_depth_rejected=p['n_rej'],
            best_detection=(max(p.get('dets') or [], key=lambda d: d['sim'],
                                default={}).get('class_name')),
            best_sim=(max((d['sim'] for d in (p.get('dets') or [])),
                          default=float('nan'))),
            n_doa_samples=p['n_doa'],
            chemicalConc=p['counts'],
            wind_direction=self.olfactionWindDirection,
            wind_speed=self.olfactionWindSpeed,
            H_fused=round(tb4.map_entropy(self.p_fused), 3),
            trigger=round(trig, 4),
            peak_object=tb4.peak_object_name(self),
            # Guard on the tracker, not the modality flag: audition can be
            # enabled while the device or topic is absent.
            q_s_map=float(np.exp(self.olfHyp.map_value())) if self.olfHyp else None,
            L0_map=float(self.sndHyp.map_value()) if self.sndHyp else None,
        ))
        np.savez_compressed(
            os.path.join(self.save_dir, f"maps_{self.step:03d}.npz"),
            fused=self.p_fused, vision=self.p_vis, olfaction=self.p_olf,
            sound=self.p_snd, free=self.free_mask,
            x_points=self.x_points, z_points=self.z_points)

    def save_data(self):
        pd.DataFrame(self.rows).to_csv(
            os.path.join(self.save_dir, 'trajectory_log.csv'), index=False)
        tx, tz = tb4.fused_peak(self)
        meta = dict(platform='turtlebot4_pro', ros_distro='jazzy',
                    modalities=self.modalities, steps=self.step + 1,
                    entropy_frac=self.entropy_frac,
                    estimated_source=dict(x=tx, y=tz),
                    peak_object=tb4.peak_object_name(self),
                    mq3_baseline=self.mq3_baseline,
                    depth_rejected=self.n_depth_rejected,
                    goal_phrase=self.goal_phrase)
        if self.olfHyp is not None:
            meta['q_s_map'] = float(np.exp(self.olfHyp.map_value()))
            meta['q_s_posterior'] = [float(v) for v in self.olfHyp.hypothesis_posterior()]
            meta['q_s_pegged'] = self.olfHyp.at_endpoint()
        if self.sndHyp is not None:
            meta['L0_map'] = float(self.sndHyp.map_value())
            meta['L0_pegged'] = self.sndHyp.at_endpoint()
        with open(os.path.join(self.save_dir, 'run_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        self.get_logger().info(
            f"estimated source at map ({tx:.2f}, {tz:.2f}) -> {self.save_dir}")


def main(args=None):
    rclpy.init(args=args)
    node = VAOTurtleBot4()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.halt()
        if node.grid_ready:
            node.save_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()