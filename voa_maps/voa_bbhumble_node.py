#!/usr/bin/env python3
"""
voa_BBHumble_node.py  --  VAO source localisation on Robuddy (ROS2 Humble).

    ros2 run voa_maps bb_node
    ros2 run voa_maps bb_node --ros-args -p goal_phrase:="rotten food smell"

    # Modality ablation -- same convention as the AI2-THOR sim's MODALITY_SETS
    ros2 run voa_maps bb_node --ros-args -p modalities:=VAO   # all three senses
    ros2 run voa_maps bb_node --ros-args -p modalities:=VA    # vision + audition
    ros2 run voa_maps bb_node --ros-args -p modalities:=VO    # vision + olfaction

Sibling of voa_TB4Jazzy_node.py. Control only: ROS interfaces, callbacks and the
state machine. Sensor maths is in rosFunctions.py (shared with the TurtleBot4); inference is in
voa_functions/ unchanged from the AI2-THOR runs.

Interfaces taken from bluebotone_robuddy_09_controller.py:
    ReSpeaker XVF3800  VID 0x2886  PID 0x001A, read over a USB control
                       transfer (PARAMETERS['DOA_VALUE'] -> resid 20, cmd 18)
    nav2               navigate_to_pose and spin action servers
    /cmd_vel           Twist
    base frame         base_link

WHAT DIFFERS FROM THE TURTLEBOT4 NODE
-------------------------------------
  * modalities are VAO, not VO -- a mic array is actually fitted
  * DOA arrives by USB polling in a background thread, not on a ROS topic
  * /mq3/raw is a bare Float32: no wind, so the plume runs at U=0
  * the XVF3800's USB control interface reports no level, so the level is
    taken from the RMS of its ALSA capture stream instead -- that feeds the L0
    range hypothesis, which turns a bearing ray into a located spot

WHY THE ROBOT STOPS TO LISTEN
-----------------------------
A moving base is the loudest thing in the room from the microphone's point of
view, and drive motors sit right in the band the array uses for DOA. The LISTEN
phase halts the base, waits for spin-down, and only then opens the microphone.
A velocity gate on /odom independently discards any sample collected while the
base is still moving, so ego-noise never reaches the belief at all.

BEFORE THE FIRST RUN
--------------------
1. Confirm MQ3_BASELINE (currently 150 counts) matches this sensor's actual
   chemical-free reading before trusting olfaction. It is a fixed constant,
   not measured at startup -- if the room, sensor, or heater warm-up state has
   drifted from when it was last measured, every reading is off by a constant
   and the emission-rate hypotheses will be pegged or biased accordingly.
2. Confirm map -> base_link exists, i.e. nav2 is running with a map. The
   assistant controller reads odom -> base_link, which drifts; a belief
   accumulated in odom smears as the run goes on.
3. Calibrate DOA_OFFSET_DEG with the source dead ahead. DOA_CCW=False is
   already verified for this array -- see rosFunctions.doa_to_alg_relative.
"""

import os
import json
import math
import time
import struct
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

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose, Spin
from builtin_interfaces.msg import Duration

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ultralytics import YOLO

from .voa_functions import visionFunction as vf
from . import rosFunctions as rf

# ============================================================= CONFIG

# Class axis comes from the loaded model so it can never drift out of sync with
# what the detector reports. Set to a list only to restrict it to a subset.
# Restricted to the classes that actually matter in this environment. Two
# effects, both wanted:
#   * the Dirichlet object map has 9 real classes instead of COCO's 80, so a
#     single detection moves a cell's posterior much further;
#   * YOLO detections of anything else are dropped by visionBranch's CLS_IDX
#     check rather than being accumulated, so a stray 'bottle' cannot dilute
#     the semantic map.
# Set to None to take every class the loaded model reports instead.
OBJECT_CLASSES = ['person', 'chair', 'couch', 'toilet', 'microwave',
                  'oven', 'sink', 'refrigerator', 'clock']

# --- topics / frames ---
RGB_TOPIC = '/oak/rgb/image_raw/compressed'
DEPTH_TOPIC = '/oak/stereo/image_raw/compressedDepth'
DEPTH_HEADER_BYTES = 12
MQ3_TOPIC = '/mq3/raw'              # Float32, raw counts
SCAN_TOPIC = '/scan'
ODOM_TOPIC = '/odom'
MAP_TOPIC = '/map'
CMD_VEL_TOPIC = '/cmd_vel'
MAP_FRAME = 'map'
BASE_FRAME = 'base_link'

# nav2's map_server LATCHES /map with TRANSIENT_LOCAL durability. A default
# (VOLATILE) subscription is an incompatible QoS match: it is created without
# error, the topic shows in `ros2 topic list`, and the callback simply never
# fires because the message was published before this node existed.
MAP_QOS = QoSProfile(
    depth=1,
    history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

# --- ReSpeaker XVF3800 (from bluebotone_robuddy_09_controller) ---
RESPEAKER_VID = 0x2886
RESPEAKER_PID = 0x001A
RESPEAKER_PARAMS = {"DOA_VALUE": (20, 18, 4, "ro", "uint16")}
RESPEAKER_TIMEOUT_MS = 1000
DOA_POLL_HZ = 20.0

# --- OAK-D S2 ---
# Autofocus hunts on flat, low-texture surfaces, which shows up as a detection
# whose depth jumps frame to frame. Lock focus once at startup if you see that.
CAM_HFOV_DEG = 69.0
CAM_VFOV_DEG = 55.0
CAM_HEIGHT_M = 0.25
# Detections beyond this are rejected rather than placed at a wrong range.
# 10 m is generous for an OAK-D S2 -- stereo depth error grows roughly with
# range squared, so a detection at 10 m may be metres off. Worth watching
# n_depth_rejected in the log: if it drops to zero, the limit is no longer
# filtering anything and far detections are landing in wrong cells.
DEPTH_TRUST_MAX_M = 10.0
# Free-space ("looked, saw nothing") range. Kept separate from the detection
# limit on purpose: a detection at 8 m is a real observation with a noisy
# range, but claiming every cell out to 8 m is EMPTY on the strength of one
# frame is a much stronger assertion, and wrong wherever an object is occluded
# or simply missed. Raise it toward DEPTH_TRUST_MAX_M if coverage matters more
# than caution.
FREE_RANGE_M = 10.0

# --- grid / planning ---
GRID_STEP = 0.25
NAV_STEP_M = 0.75
# How far short of the target the robot stops. The belief peak usually lands ON
# an object, and driving to that cell means driving into it -- snapping the
# goal to a "free" cell does not help, because objects added since the map was
# built are free on paper and solid in reality. A standoff also keeps the
# object inside the camera frustum instead of against the lens.
# Should exceed the robot's footprint radius plus nav2's inflation.
STANDOFF_M = 0.5
NAV_TIMEOUT_S = 10.0
SETTLE_S = 1.5
SENSE_PERIOD_S = 1.0
LISTEN_S = 3.0
EGO_NOISE_VEL = 0.02

# --- audition ---
# 'VAO'  = all three senses processed (the default: full ablation baseline)
# 'VA'   = vision + audition, olfaction skipped entirely (no MQ3 subscription,
#          no q_s tracker)
# 'VO'   = vision + olfaction, audition skipped (no ReSpeaker probe, no DOA
#          thread, no audio stream)
# Same convention as the AI2-THOR simulation's MODALITY_SETS. Overridden by
# the `modalities` ROS parameter at runtime; this is only the default.
DEFAULT_MODALITIES = 'VAO'
DOA_OFFSET_DEG = 0.0        # array zero vs robot forward; calibrate once
DOA_CCW = False             # VERIFIED for this array -- see the functions file
DOA_SIGMA_DEG = 15.0
DOA_BURST_BIAS_DEG = 10.0
DOA_INDEPENDENT_FRAC = 0.2
DOA_REQUIRE_SPEECH = False  # the XVF3800 flag is tuned for speech, not alarms

# --- sound level, from the ALSA capture stream ---
# DOA_VALUE carries no level, so the level comes from the RMS of the audio the
# array is already streaming. That feeds the L0 range hypothesis, which is what
# turns a bearing RAY into a located SPOT.
#
# AGC IS THE RISK. If the XVF3800 is applying automatic gain control, a distant
# quiet source is amplified toward the same RMS as a near loud one and the
# range information is destroyed -- the hypothesis will then peg at a grid edge
# and the run_meta L0_pegged flag will say so. Verify by walking the robot
# toward a steady source and checking level_db rises monotonically; if it does
# not, set ENABLE_SOUND_LEVEL = False and accept bearing-only triangulation.
ENABLE_SOUND_LEVEL = True
AUDIO_DEVICE = None         # None = default ALSA input; set the index if needed
AUDIO_SAMPLE_RATE = 16000   # Hz, must match what the array's ALSA device exposes
AUDIO_CHANNELS = 2          # channels to open on the capture stream
# Neither channel is a raw mic signal -- both are XVF3800 outputs (per the
# assistant controller's own comment):
#   channel 0 = AEC + BEAMFORMING + post-process
#   channel 1 = ASR output of the auto-selected beam (also processed)
#
# CHANNEL 0, not an average of both. Beamforming's job is to steer gain toward
# wherever the array currently believes the speaker is and suppress off-axis
# energy -- that is an AGC-like distortion for level, and it breaks the
# assumption the L0 range hypothesis depends on (level falls off with distance
# in a fixed, monotone way). Averaging channels 0 and 1 mixes in channel 1's
# own independent beam-selection behaviour on top of that. Channel 1 is not a
# cleaner alternative -- it is a different processed signal, not a raw one.
AUDIO_LEVEL_CHANNEL = 0
AUDIO_BLOCK = 1600          # samples per RMS block (100 ms at 16 kHz)
LEVEL_REF_RMS = 1.0         # arbitrary: only level DIFFERENCES are identifiable

# --- sensor noise, already inflated ---
# --- plume shape ---
# c(r) = q_s / (4 pi D r) * exp(-r / lambda),   lambda = sqrt(D * tau)  [U = 0]
#
# TWO SEPARATE THINGS, and they are easy to confuse:
#   SHAPE     is set by lambda = sqrt(D*tau)  -- how fast influence falls off
#   AMPLITUDE is set by the RATIO q_s/D       -- only the ratio is identifiable
#
# The old D=10, tau=1000 gives lambda = 100 m. In a 4x4 m room exp(-r/lambda)
# is 0.96-0.99 everywhere, so the decay term did nothing at all and the model
# was a pure 1/r law with no room scale in it.
#
# For a ~4x4 m room lambda should be about half the room span, ~2 m.
# lambda = 2 needs D*tau = 4, and TAU is the safe knob to turn: changing D
# instead would rescale the amplitude by the same factor and push the required
# q_s outside Q_S_HYPOTHESES (D=0.004 would need q_s ~ 40, below the grid
# floor, so it would silently peg). Keeping D fixed leaves the working q_s
# range valid.
PLUME_D = 0.004       # diffusivity; paired with q_s, only q_s/D is identifiable
PLUME_TAU = 1000.0    # particle lifetime; D*TAU = 4 -> lambda = 2.0 m

OLF_SIGMA_LOG = 0.75

# Consecutive chemical readings are NOT independent. The MQ3 responds over
# seconds, and the six initialization views are taken from one spot ~10 cm
# apart -- measured log-noise there was 0.040, while sigma is already 0.75, so
# sigma is 19x the actual noise and raising it further is the wrong lever.
# The real problem is that each reading multiplies in a full likelihood, so six
# near-identical readings raise the evidence to the 6th power and drive the
# posterior ring to 0 and 1 within two steps.
# Tempering down-weights each update: t=1 is the old behaviour, t=1/6 treats a
# six-view scan as roughly one independent observation.
OLF_TEMPER = 0.25
SND_SIGMA_DB = 7.5

# Fixed chemical-free MQ3 reading, counts. Replaces sampling clean air at
# startup: sampling only gives a true baseline if the source is confirmed OFF
# at that moment, which removes an assumption the run otherwise depends on
# silently. Re-measure and update this if the sensor, room, or heater
# warm-up state changes -- it is not re-checked at runtime.
MQ3_BASELINE = 150.0

# Emission-rate hypotheses, in the plume model's units. THE SPAN IS THE THING
# THAT MATTERS, not the number of points.
#
# c(r) = q_s / (4 pi D r), so the largest hypothesis sets a hard CEILING on how
# far away the source can possibly be judged to be:
#       r_max = q_s_max / (4 pi D c)
# With the old ceiling of 5000 and D=10, a reading of 200 net counts could only
# be explained by a source at 0.20 m -- no cell farther away had any hypothesis
# that could produce that reading, so the posterior collapsed to a tight blob
# right next to the robot regardless of where the source actually was. That is
# a grid artefact, not evidence.
#
# 1e3..1e6 lets 200 counts be explained anywhere from 0.2 m to ~200 m, which
# comfortably covers a room. Watch the q_s_pegged flag in run_meta: if the
# posterior still piles onto the top hypothesis, widen further.
# Rescaled for D = 0.004. Only the RATIO q_s/D is identifiable, so dropping D
# by 2500x drops the q_s that explains the same reading by the same factor:
# the old 1e5 becomes ~40. Measured against this run's 320-490 counts, a
# 10..100 grid spans source ranges of roughly 0.45-4.5 m, i.e. the whole room.
# Widened slightly at both ends so that a low reading far from the source does
# not peg at the floor -- check q_s_pegged in run_meta if in doubt.
Q_S_HYPOTHESES = tuple(10.0 ** np.linspace(np.log10(5), np.log10(200), 7))

# --- fusion / termination ---
W_VISION, W_OLFACT, W_SOUND = 1.0, 0.5, 0.5
ENTROPY_FRAC = 0.7
MAX_STEPS = 3

# --- initialization phase: 5 egocentric views before search ---
# Mirrors initialize_envKnowledge in the AI2-THOR simulation, which rotates
# 4x90 degrees and calls the vision branch at each stop before search begins.
# Doing this on hardware too means the object map starts with a real look
# around the room instead of whatever the first few search steps happen to
# see, and it gives olfaction (and audition, if fitted) an initial reading
# from every direction rather than just the one the robot happened to be
# facing at startup.
# 6 headings x 60 deg. With a 69 deg HFOV that overlaps by 9 deg; 4 x 90 deg
# would leave a 21 deg blind wedge between consecutive views, wide enough to
# hide an object entirely at range.
INIT_HEADINGS = 6
INIT_SPIN_RAD = 2.0 * math.pi / INIT_HEADINGS   # closed-loop via nav2's Spin
SPIN_TIMEOUT_S = 15.0

# Frames captured at each heading AFTER the base has stopped. Several frames
# per view costs about a second and removes the single-frame failures that
# dominate real runs: residual rocking, autofocus still hunting, and YOLO
# simply missing an object in one frame that it catches in the next.
PHOTOS_PER_VIEW = 3
PHOTO_INTERVAL_S = 0.35
# Settle time before the FIRST frame at a new heading. Separate from SETTLE_S
# (which exists for the microphone) because the camera needs less: it only has
# to stop moving, not wait for drivetrain noise to die away.
CAMERA_SETTLE_S = 0.6

# Whether SEARCH also does a full rotation at each waypoint, not just
# initialization. Without it the object map only ever sees whichever direction
# the robot happened to be facing when nav2 finished driving.
SCAN_DURING_SEARCH = True
FREE_EVIDENCE = 3.0


class ReSpeaker:
    """Vendor control-transfer access to the XVF3800.

    Lifted from bluebotone_robuddy_09_controller so both nodes talk to the
    device identically. No interface is claimed, so this coexists with ALSA's
    hold on the audio interfaces.
    """

    def __init__(self, dev):
        self.dev = dev
        self._lock = threading.Lock()

    def read(self, name):
        import usb.util
        resid, cmd, length_bytes, _acc, typ = RESPEAKER_PARAMS[name]
        cmdid = 0x80 | cmd
        length = length_bytes + 1                  # +1 for the leading status byte
        with self._lock:
            resp = self.dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR
                | usb.util.CTRL_RECIPIENT_DEVICE,
                0, cmdid, resid, length, RESPEAKER_TIMEOUT_MS)
        b = resp.tobytes()
        if typ == "uint16":
            n = length_bytes // 2
            return list(struct.unpack("<" + "H" * n, b[1:1 + n * 2]))
        return list(resp)

    def read_doa(self):
        """(doa_degrees:int, is_speech:bool) or (None, False)."""
        try:
            words = self.read("DOA_VALUE")
            if not words or len(words) < 2:
                return None, False
            return int(words[0]), bool(words[1])
        except Exception:
            return None, False

    @staticmethod
    def open():
        try:
            import usb.core
            dev = usb.core.find(idVendor=RESPEAKER_VID, idProduct=RESPEAKER_PID)
            return ReSpeaker(dev) if dev is not None else None
        except Exception:
            return None


class VAORobuddy(Node):

    def __init__(self):
        super().__init__('voa_bb_node')

        self.declare_parameter('goal_phrase', 'rotten food smell')
        self.declare_parameter('sound_phrase', 'an alarm clock ringing')
        self.declare_parameter('yolo_path', 'voa_maps/models/YOLO/yolo26m.pt')
        self.declare_parameter('save_dir', '')
        self.declare_parameter('modalities', DEFAULT_MODALITIES)
        self.declare_parameter('entropy_frac', ENTROPY_FRAC)
        gp = lambda n: self.get_parameter(n).value

        self.goal_phrase = f"Is emitting {gp('goal_phrase')} odor:"
        self.sound_phrase = gp('sound_phrase')
        self.entropy_frac = float(gp('entropy_frac'))
        # Which senses this run actually processes -- same convention as the
        # AI2-THOR simulation's ablation strings ('VAO', 'VA', 'VO', ...).
        # Hardware availability can still veto A even if requested (see the
        # ReSpeaker probe below); V and O are software-only toggles.
        # Canonical V,A,O order rather than alphabetical, so the string in the
        # log reads 'VAO'/'VO' as written rather than 'AOV'/'OV'.
        _req = set(gp('modalities').upper()) & set('VAO')
        requested = ''.join(c for c in 'VAO' if c in _req)
        if not requested:
            self.get_logger().warn(
                f"modalities={gp('modalities')!r} contains none of V/A/O; "
                f"defaulting to {DEFAULT_MODALITIES}")
            requested = DEFAULT_MODALITIES
        self.use_V = 'V' in requested
        self.use_O = 'O' in requested
        self.use_A = 'A' in requested   # may still be revoked below if no device

        # --- config exposed to the helpers via node state ---
        self.cam_hfov_deg, self.cam_vfov_deg = CAM_HFOV_DEG, CAM_VFOV_DEG
        self.cam_height_m = CAM_HEIGHT_M
        self.depth_trust_max_m = DEPTH_TRUST_MAX_M
        self.free_evidence = FREE_EVIDENCE
        # How far "I looked here and saw nothing" is trusted. Previously fixed
        # at visionFunction's 2.5 m default, which is why the object map only
        # ever updated within ~3 m of the robot even with a 10 m depth limit.
        self.free_range_m = FREE_RANGE_M
        self.nav_step_m = NAV_STEP_M
        self.standoff_m = STANDOFF_M
        self.olf_sigma_log = OLF_SIGMA_LOG
        self.olf_temper = OLF_TEMPER
        self.plume_D = PLUME_D
        self.plume_tau = PLUME_TAU
        self.q_s_hypotheses = Q_S_HYPOTHESES
        # Wide, because an RMS-derived dB has an arbitrary offset: the grid has
        # to span wherever the uncalibrated scale happens to land.
        self.l0_hypotheses = tuple(np.linspace(20.0, 100.0, 5))
        self.snd_sigma_db = SND_SIGMA_DB
        self.w_vision, self.w_olfact, self.w_sound = W_VISION, W_OLFACT, W_SOUND
        self.doa_offset_deg, self.doa_ccw = DOA_OFFSET_DEG, DOA_CCW
        # Platform capability flags read by rosFunctions.
        # The XVF3800's USB control interface exposes only DOA_VALUE = (angle,
        # vad_flag) -- no level. But the array is also an ALSA capture device,
        # and the RMS of that stream IS a level. Absolute calibration does not
        # matter here: L0 is tracked as an unknown, so only DIFFERENCES between
        # positions are identifiable and any constant offset cancels.
        self.has_sound_level = ENABLE_SOUND_LEVEL
        self.doa_sigma_deg = DOA_SIGMA_DEG
        self.doa_burst_bias_deg = DOA_BURST_BIAS_DEG
        self.doa_independent_frac = DOA_INDEPENDENT_FRAC

        stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.save_dir = gp('save_dir') or os.path.join(
            os.getcwd(), f'voa_maps/save/bb/voa_run_{stamp}')
        os.makedirs(self.save_dir, exist_ok=True)

        try:
            self.yoloModel = YOLO(gp('yolo_path'))
            self.get_logger().info(f"YOLO loaded from {gp('yolo_path')}")
        except Exception as e:
            self.get_logger().error(f"YOLO load failed: {e}")
            raise

        classes = OBJECT_CLASSES or [self.yoloModel.names[i]
                                     for i in sorted(self.yoloModel.names)]
        vf.configure_classes(classes)
        self.get_logger().info(
            f"object-map classes ({len(classes)} from the detector): "
            f"{classes if len(classes) <= 12 else classes[:12] + ['...']}")

        # --- ReSpeaker: opened before announcing modalities, since a missing
        #     device downgrades the run from VAO to VO ---
        self.respeaker = ReSpeaker.open() if self.use_A else None
        if self.use_A and self.respeaker is None:
            self.get_logger().warn(
                f"no XVF3800 at VID {RESPEAKER_VID:#06x} PID {RESPEAKER_PID:#06x}; "
                f"auditory branch disabled")
            self.use_A = False
        # Actual modalities, after hardware has had a chance to veto A. This
        # can differ from `requested` (e.g. requested 'VAO' with no ReSpeaker
        # attached actually runs as 'VO') -- both are recorded in run_meta so
        # a missing device shows up in the results rather than silently
        # changing what the run claims to have tested.
        self.requested_modalities = requested
        self.modalities = ''.join(c for c, on in
                                  (('V', self.use_V), ('A', self.use_A),
                                   ('O', self.use_O)) if on)
        if self.modalities != self.requested_modalities:
            self.get_logger().warn(
                f"requested modalities={self.requested_modalities} but running "
                f"as {self.modalities} (hardware unavailable)")
        self.save_dir = os.path.join(self.save_dir, self.modalities)
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(
            f"modalities={self.modalities}  save_dir={self.save_dir}")

        # --- ROS interfaces ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.create_subscription(CompressedImage, RGB_TOPIC,
                                 self.image_callback, qos_profile_sensor_data)
        self.create_subscription(CompressedImage, DEPTH_TOPIC,
                                 self.depth_callback, qos_profile_sensor_data)
        if self.use_O:
            self.create_subscription(Float32, MQ3_TOPIC, self.mq3_callback, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.laser_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, MAP_TOPIC,
                                 self.map_callback, MAP_QOS)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')

        # --- sensor state ---
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
        self.mq3_counts = 0.0
        self.have_olfaction = False
        self.mq3_baseline = MQ3_BASELINE
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        # Every pose the robot has occupied, for the trajectory lines drawn on
        # every saved map. Appended by update_pose, so it captures init-phase
        # views and mid-drive samples too, not just one point per search step.
        self.trail = []

        self.doa_lock = threading.Lock()
        self.doa_burst = []
        self.listening = False
        self._level_lock = threading.Lock()
        self._level_db = None
        self._audio_stream = None
        self._stop_evt = threading.Event()
        self._doa_thread = None

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
        # Phase names match the AI2-THOR simulation's behavior_flag values
        # ("Initialization", "search", "goal_navigation") so runs from both
        # environments group the same way in analysis.
        self.phase = 'Initialization'
        self.init_index = 0
        self.camera_settle_until = 0.0
        self.scan_index = 0
        # Set once a waypoint's rotation is done and cleared after the next
        # navigation leg, so each search step scans exactly once instead of
        # re-entering the scan forever.
        self.scan_done = False
        self.spin_active = False
        self.spin_deadline = 0.0

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

        if self.use_A:
            if ENABLE_SOUND_LEVEL:
                self._start_audio()
            self._doa_thread = threading.Thread(target=self._doa_loop, daemon=True)
            self._doa_thread.start()

        self.create_timer(0.1, self.control_callback)

    # ---------------------------------------------------------------- callbacks
    def image_callback(self, msg):
        img = rf.decode_rgb(msg)
        if img is None:
            self.get_logger().error("Failed to decode RGB image")
            return
        self.latest_rgb_image = img
        self.latest_rgb_shape = img.shape
        self.latest_rgb_stamp = msg.header.stamp

    def depth_callback(self, msg):
        img = rf.decode_depth(msg, DEPTH_HEADER_BYTES)
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

    def mq3_callback(self, msg):
        self.mq3_counts = float(msg.data)
        self.have_olfaction = True

    # ---------------------------------------------------------------- audio level
    def _start_audio(self):
        """Open the array's ALSA capture stream purely to measure RMS.

        No interface is claimed on the USB control endpoint, so this coexists
        with the DOA control transfers -- the same arrangement the assistant
        controller uses.
        """
        try:
            import sounddevice as sd
        except Exception as e:
            self.get_logger().warn(
                f"sounddevice unavailable ({e}); sound level disabled, "
                f"audition will be bearing-only")
            self.has_sound_level = False
            return
        try:
            self._audio_stream = sd.InputStream(
                device=AUDIO_DEVICE, samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS, dtype='int16',
                blocksize=AUDIO_BLOCK, callback=self._audio_callback)
            self._audio_stream.start()
            self.get_logger().info(
                f"audio level stream open ({AUDIO_SAMPLE_RATE} Hz, "
                f"{AUDIO_CHANNELS} ch) -- L0 range hypothesis active")
        except Exception as e:
            self.get_logger().warn(
                f"could not open audio stream ({e}); sound level disabled, "
                f"audition will be bearing-only")
            self.has_sound_level = False
            self._audio_stream = None

    def _audio_callback(self, indata, frames, t, status):
        """RMS of ONE channel of the block, in dB on an arbitrary reference.

        Uses AUDIO_LEVEL_CHANNEL (0 by default) rather than averaging every
        channel together -- see the config comment for why channel 0 is the
        better of the two available, and why neither is a raw signal.

        The dB reference is arbitrary on purpose: L0 is a tracked unknown, so
        a constant offset cancels in the position-to-position differences that
        actually carry the range information.
        """
        try:
            x = np.asarray(indata, dtype=np.float64)
            if x.ndim > 1:
                ch = min(AUDIO_LEVEL_CHANNEL, x.shape[1] - 1)
                x = x[:, ch]
            rms = float(np.sqrt(np.mean(x * x)))
            if rms > 0.0:
                with self._level_lock:
                    self._level_db = 20.0 * math.log10(rms / LEVEL_REF_RMS)
        except Exception:
            pass

    def _current_level_db(self):
        with self._level_lock:
            return self._level_db

    # ---------------------------------------------------------------- DOA thread
    def _doa_loop(self):
        """Poll the XVF3800 over USB and collect samples only while stationary.

        Gated here rather than filtered downstream, so ego-noise never reaches
        the belief. The velocity check catches residual motion during the
        settle window and anything nav2 issues unexpectedly.
        """
        period = 1.0 / DOA_POLL_HZ
        while not self._stop_evt.is_set():
            if self.listening and self.lin_vel <= EGO_NOISE_VEL \
                    and self.ang_vel <= EGO_NOISE_VEL:
                angle, is_speech = self.respeaker.read_doa()
                if angle is not None and (is_speech or not DOA_REQUIRE_SPEECH):
                    # Pair each bearing with the level measured at the same
                    # moment. rosFunctions.auditionBranch accepts either a bare
                    # angle or an (angle, level) tuple.
                    lvl = self._current_level_db() if self.has_sound_level else None
                    with self.doa_lock:
                        self.doa_burst.append((float(angle), lvl))
            time.sleep(period)

    # ---------------------------------------------------------------- pose
    def update_pose(self, stamp=None):
        """Refresh robot_map_* from TF. Returns True when a pose is available.

        With a stamp the transform is looked up AT THAT TIME: a camera frame
        processed while nav2 is driving was captured at a pose the robot has
        already left, and using the latest transform projects every detection
        into the wrong cell.
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
        # Only record a new trail point once the robot has actually moved, so
        # standing still through a 3 s listening window does not add hundreds
        # of duplicate points to every plotted line.
        if (not self.trail
                or math.hypot(self.robot_map_posX - self.trail[-1][0],
                              self.robot_map_posY - self.trail[-1][1]) > 0.02):
            self.trail.append((self.robot_map_posX, self.robot_map_posY))
        return True

    def halt(self):
        self.cmd_vel_pub.publish(Twist())

    def send_spin(self, relative_yaw_rad):
        """Closed-loop in-place rotation via nav2's Spin action.

        Relative to current heading; positive is CCW per REP-103, so no frame
        conversion is needed -- this never touches the algorithm's (x, z) yaw
        convention at all, it is a pure ROS rotation command.
        """
        if not self.spin_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn("nav2 spin action unavailable")
            return False
        g = Spin.Goal()
        g.target_yaw = float(relative_yaw_rad)
        g.time_allowance = Duration(sec=int(SPIN_TIMEOUT_S))
        self.spin_active = True
        self.spin_deadline = time.time() + SPIN_TIMEOUT_S
        self.spin_client.send_goal_async(g).add_done_callback(self._spin_response)
        return True

    def _spin_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("nav2 rejected the spin goal")
            self.spin_active = False
            return
        handle.get_result_async().add_done_callback(
            lambda f: setattr(self, 'spin_active', False))

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
        qz, qw = rf.yaw_to_quaternion(yaw)
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
            if not rf.build_grid_from_map(self, self.map_msg, GRID_STEP):
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
            if not self.use_V:
                # Vision is off for this run -- nothing to wait for. All three
                # requested modes (VAO/VA/VO) include V, so this only matters
                # if you ever run 'AO' or 'A' or 'O' alone.
                self.state = 'INIT_BEGIN'
                return
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
                f"depth {self.latest_depth_image.shape[1]}x"
                f"{self.latest_depth_image.shape[0]} ({self.latest_depth_image.dtype})")
            self.state = 'INIT_BEGIN'
            return

        if self.state == 'INIT_BEGIN':
            self.init_index = 0
            self.phase = 'Initialization'
            self.get_logger().info(
                f"=== INITIALIZATION: {INIT_HEADINGS} orthogonal views ===")
            self.state = 'INIT_SENSE'
            return

        if self.state == 'INIT_SENSE':
            # Hold still and let the camera settle BEFORE capturing. Frames
            # grabbed while the base is still rocking after a spin are the
            # motion-blurred ones.
            self.halt()
            if self.camera_settle_until == 0.0:
                self.camera_settle_until = now + CAMERA_SETTLE_S
                return
            if now < self.camera_settle_until:
                return
            self.camera_settle_until = 0.0

            if not self.update_pose(self.latest_rgb_stamp):
                return
            dets, n_rej = [], 0
            if self.use_V:
                dets, n_rej = self.capture_views(f"init_{self.init_index:02d}")
            counts = rf.olfactionBranch(self) if self.use_O else None
            self.get_logger().info(
                f"--- init view {self.init_index + 1}/{INIT_HEADINGS} @ "
                f"({self.robot_map_posX:.2f}, {self.robot_map_posY:.2f}) ---")
            self.report_sensing(dets, n_rej, counts)
            self._pending = dict(n_det=len(dets), n_rej=n_rej, counts=counts, n_doa=0,
                                 dets=dets, rx=self.robot_map_posX, ry=self.robot_map_posY)

            if self.use_A:
                self.halt()
                self.listening = False
                self.listen_from = now + SETTLE_S
                self.phase_until = now + SETTLE_S + LISTEN_S
                self.state = 'INIT_LISTEN'
                return

            self.log_init_view(self._pending)
            self.state = 'INIT_NEXT'
            return

        if self.state == 'INIT_LISTEN':
            if now >= self.listen_from:
                self.listening = True
            if now < self.phase_until:
                self.halt()
                return
            self.listening = False
            n = rf.auditionBranch(self)
            self._pending['n_doa'] = n
            self.get_logger().info(
                f"  [init view {self.init_index + 1}/{INIT_HEADINGS}] audition "
                f"{n} DOA samples")
            self.log_init_view(self._pending)
            self.state = 'INIT_NEXT'
            return

        if self.state == 'INIT_NEXT':
            self.init_index += 1
            if self.init_index >= INIT_HEADINGS:
                self.get_logger().info(
                    "=== INITIALIZATION complete; entering SEARCH ===")
                self.phase = 'search'
                self.state = 'SENSE'
                return
            if self.send_spin(INIT_SPIN_RAD):
                self.state = 'INIT_ROTATE'
            else:
                # No spin server: proceed without turning rather than crash.
                # Every view will then be the same heading, which is a real
                # loss of coverage but a recoverable one -- worth fixing nav2
                # before trusting the object map, not worth stopping the run.
                self.get_logger().warn(
                    "cannot rotate for the next view; continuing at the "
                    "current heading (coverage will be incomplete)")
                self.state = 'INIT_SENSE'
            return

        if self.state == 'INIT_ROTATE':
            if self.spin_active and now < self.spin_deadline:
                return
            if self.spin_active:
                self.get_logger().warn("spin action timed out; continuing anyway")
                self.spin_active = False
            self.state = 'INIT_SENSE'
            return

        if self.state == 'SEARCH_SCAN_SENSE':
            self.halt()
            if self.camera_settle_until == 0.0:
                self.camera_settle_until = now + CAMERA_SETTLE_S
                return
            if now < self.camera_settle_until:
                return
            self.camera_settle_until = 0.0
            if not self.update_pose(self.latest_rgb_stamp):
                return
            self.capture_views(f"scan_{self.step:03d}_{self.scan_index:02d}")
            self.get_logger().info(
                f"  scan view {self.scan_index + 1}/{INIT_HEADINGS} "
                f"@ ({self.robot_map_posX:.2f}, {self.robot_map_posY:.2f})")
            self.state = 'SEARCH_SCAN_NEXT'
            return

        if self.state == 'SEARCH_SCAN_NEXT':
            self.scan_index += 1
            if self.scan_index >= INIT_HEADINGS:
                # Rotation done -- fall through to the normal sensing step,
                # which will not re-enter the scan because scan_done is set.
                self.state = 'SENSE'
                return
            if self.send_spin(INIT_SPIN_RAD):
                self.state = 'SEARCH_SCAN_ROTATE'
            else:
                self.get_logger().warn(
                    "cannot rotate mid-search; continuing at the current heading")
                self.state = 'SENSE'
            return

        if self.state == 'SEARCH_SCAN_ROTATE':
            if self.spin_active and now < self.spin_deadline:
                return
            if self.spin_active:
                self.get_logger().warn("scan spin timed out; continuing anyway")
                self.spin_active = False
            self.state = 'SEARCH_SCAN_SENSE'
            return

        if self.state == 'SENSE':
            # A full rotation at this waypoint before sensing, so the object
            # map gets all round rather than only the heading nav2 left the
            # robot facing. Reuses the initialization scan states; scan_index
            # is reset here and SCAN_RETURN sends control back to SENSE_DO.
            if SCAN_DURING_SEARCH and self.use_V and not self.scan_done:
                self.scan_index = 0
                self.scan_done = True
                self.state = 'SEARCH_SCAN_SENSE'
                return

            self.halt()
            if self.camera_settle_until == 0.0:
                self.camera_settle_until = now + CAMERA_SETTLE_S
                return
            if now < self.camera_settle_until:
                return
            self.camera_settle_until = 0.0

            if not self.update_pose(self.latest_rgb_stamp):
                return
            dets, n_rej = [], 0
            if self.use_V:
                dets, n_rej = self.capture_views(f"yolo_{self.step:03d}")
            counts = rf.olfactionBranch(self) if self.use_O else None
            self.report_sensing(dets, n_rej, counts)
            self._pending = dict(n_det=len(dets), n_rej=n_rej, counts=counts,
                                 n_doa=0, dets=dets,
                                 rx=self.robot_map_posX, ry=self.robot_map_posY)

            if self.use_A:
                self.halt()
                self.listening = False       # stays shut until spin-down
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
            n = rf.auditionBranch(self)
            self._pending['n_doa'] = n
            lvl = self._current_level_db()
            self.get_logger().info(
                f"  audition   {n} DOA samples over {LISTEN_S:.1f} s while stationary"
                + (f", level {lvl:.1f} dB" if lvl is not None else
                   ", bearing-only (no level)"))
            if self.sndHyp is not None and self.sndHyp.at_endpoint():
                self.get_logger().warn(
                    "L0 posterior pegged at a grid edge -- either the level scale "
                    "is outside l0_hypotheses, or AGC is flattening it")
            self.state = 'FUSE'
            return

        if self.state == 'FUSE':
            trig = rf.update_belief(self)
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
                tx, tz = rf.fused_peak(self)
                why = 'converged' if trig <= self.entropy_frac else 'step limit'
                self.get_logger().info(
                    f"{why}; peak is {rf.peak_object_name(self)} at "
                    f"({tx:.2f}, {tz:.2f}); driving there and terminating")
                self.phase = 'goal_navigation'
                # Approach to STANDOFF_M and face it, rather than driving onto
                # the cell. Uncapped: this is the final leg, so there is no
                # reason to stop part-way.
                gx, gy, gyaw = rf.standoff_goal(self, tx, tz)
                self.get_logger().info(
                    f"  approaching to within {self.standoff_m:.2f} m -> "
                    f"({gx:.2f}, {gy:.2f})")
                self.send_nav_goal(gx, gy, gyaw)
                self.state = 'FINISH'
                return

            gx, gy, gyaw = rf.pick_goal(self)
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
                            _, _d, rej = rf.visionBranch(self.yoloModel, self)
                            self.n_depth_rejected += rej
                        if self.use_O:
                            rf.olfactionBranch(self)
                return
            if self.nav_active:
                self.get_logger().warn("nav goal timed out; continuing")
                self.nav_active = False
            self.halt()
            self.step += 1
            self.scan_done = False      # allow one scan at the new waypoint
            self.state = 'SENSE'
            return

        if self.state == 'FINISH':
            if self.nav_active and now < self.nav_deadline:
                return
            self.halt()
            self.save_data()
            self.state = 'DONE'
            self.get_logger().info("run complete; shutting down")
            # Actually exit rather than sitting in DONE forever. rclpy.spin()
            # blocks indefinitely, and a state machine that has stopped
            # changing state gives it no reason to return -- so raise, and let
            # main() catch it. SystemExit is the conventional way to break out
            # of a spin from inside a callback: it unwinds through spin()
            # rather than being swallowed by the executor's error handling.
            raise SystemExit

    # ---------------------------------------------------------------- capture
    def capture_views(self, tag):
        """Run vision over several DISTINCT frames at the current heading.

        Each accepted frame is folded into the object map separately, so N
        frames deposit N units of Dirichlet evidence at a detected cell. That
        is the point: a detection YOLO finds in 2 of 3 frames accumulates real
        weight, while a single-frame false positive is outvoted rather than
        being treated as established fact.

        FRAME FRESHNESS IS ENFORCED. The camera callback runs on its own
        thread, so simply calling visionBranch three times in a row can process
        the SAME buffered image three times -- which would triple-count one
        observation and manufacture confidence out of nothing. Each iteration
        waits for the RGB timestamp to change, and a frame that never arrives
        is skipped rather than counted again.

        Only the last annotated frame is written to disk; writing all of them
        multiplies the image count for little extra information.

        Assumes the base is already stopped. The scan states halt and settle
        first -- capturing mid-drive is what produced the blurred frames.
        """
        all_dets, total_rej, annotated = [], 0, None
        n_used, n_stale = 0, 0
        last_stamp = None

        for i in range(max(1, PHOTOS_PER_VIEW)):
            if i > 0:
                # Wait for genuinely new pixels, not just a fixed delay.
                deadline = time.time() + PHOTO_INTERVAL_S + 0.5
                while time.time() < deadline:
                    if self._rgb_stamp_key() != last_stamp:
                        break
                    time.sleep(0.02)
                else:
                    # Timed out with no new frame. Counting the old one again
                    # would be inventing evidence, so skip this slot.
                    n_stale += 1
                    continue
                self.update_pose(self.latest_rgb_stamp)

            last_stamp = self._rgb_stamp_key()
            ann, dets, n_rej = rf.visionBranch(self.yoloModel, self)
            total_rej += n_rej
            all_dets.extend(dets)
            n_used += 1
            if ann is not None:
                annotated = ann

        self.n_depth_rejected += total_rej
        if annotated is not None:
            cv.imwrite(os.path.join(self.save_dir, f"{tag}.jpg"), annotated)
        if n_stale:
            self.get_logger().warn(
                f"{tag}: only {n_used}/{PHOTOS_PER_VIEW} frames were new "
                f"({n_stale} stale, skipped rather than double-counted) -- "
                f"the camera may be publishing slower than PHOTO_INTERVAL_S")
        return all_dets, total_rej

    def _rgb_stamp_key(self):
        """Hashable identity of the current RGB frame, for freshness checks."""
        st = self.latest_rgb_stamp
        if st is None:
            return None
        return (getattr(st, 'sec', None), getattr(st, 'nanosec', None))

    # ---------------------------------------------------------------- reporting
    def report_sensing(self, dets, n_rej, counts):
        """Print what the sensors actually returned this step.

        The similarity column is the one to watch: a detection with high
        confidence but near-zero sim contributes almost nothing to the visual
        belief, because the semantic map is the class posterior projected onto
        exactly these numbers.
        """
        yaw = rf.quaternion_to_yaw(0, 0, self.robot_map_angZ, self.robot_map_angW)
        self.get_logger().info(
            f"--- step {self.step} @ ({self.robot_map_posX:.2f}, "
            f"{self.robot_map_posY:.2f}) yaw {math.degrees(yaw):.0f} deg ---")

        if self.use_O:
            if counts is None:
                self.get_logger().info(
                    f"  olfaction  raw {self.mq3_counts:.1f} counts, "
                    f"baseline not set yet")
            else:
                self.get_logger().info(
                    f"  olfaction  raw {self.mq3_counts:.1f} - "
                    f"baseline {self.mq3_baseline:.1f} = {counts:.1f} counts")

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
    def log_init_view(self, p):
        """One row per orthogonal init view -- no trigger/entropy yet, since
        the belief has not been fused at this point."""
        yaw = rf.quaternion_to_yaw(0, 0, self.robot_map_angZ, self.robot_map_angW)
        self.rows.append(dict(
            step=-1, phase='Initialization', init_view=self.init_index,
            time=round(time.time() - self.t0, 2),
            robot_x=self.robot_map_posX, robot_y=self.robot_map_posY,
            robot_yaw=round(math.degrees(yaw), 1),
            n_detections=p['n_det'], n_depth_rejected=p['n_rej'],
            yolo_raw=(getattr(self, 'last_vision_counts', {}) or {}).get('raw'),
            det_no_depth=(getattr(self, 'last_vision_counts', {}) or {}).get('no_depth'),
            det_off_class=(getattr(self, 'last_vision_counts', {}) or {}).get('unknown_class'),
            n_doa_samples=p['n_doa'], chemicalConc=p['counts'],
        ))
        # The object map is already accumulating during initialization, so it
        # is worth rendering. The BELIEF maps are not -- update_belief has not
        # run yet, so every branch is still its uniform prior and the panels
        # would just be blank squares.
        # Full diagnostic set for this init view. The BELIEF panels are
        # included even though update_belief has not run yet -- during
        # initialization they show the priors, which is the correct baseline to
        # compare the first search step against.
        rf.save_object_map(self, self.init_index, prefix='init_objects')
        rf.save_modality_diagnostics(self, self.init_index, prefix='init_diag')
        rf.save_vision_class_maps(self, self.init_index,
                                  prefix='init_vision_classes')
        rf.save_cell_csv(self, self.init_index, prefix='init_cells')

    def log_step(self, p, trig):
        yaw = rf.quaternion_to_yaw(0, 0, self.robot_map_angZ, self.robot_map_angW)
        self.rows.append(dict(
            step=self.step, phase=self.phase, time=round(time.time() - self.t0, 2),
            robot_x=self.robot_map_posX, robot_y=self.robot_map_posY,
            robot_yaw=round(math.degrees(yaw), 1),
            n_detections=p['n_det'], n_depth_rejected=p['n_rej'],
            yolo_raw=(getattr(self, 'last_vision_counts', {}) or {}).get('raw'),
            det_no_depth=(getattr(self, 'last_vision_counts', {}) or {}).get('no_depth'),
            det_off_class=(getattr(self, 'last_vision_counts', {}) or {}).get('unknown_class'),
            best_detection=(max(p.get('dets') or [], key=lambda d: d['sim'],
                                default={}).get('class_name')),
            best_sim=(max((d['sim'] for d in (p.get('dets') or [])),
                          default=float('nan'))),
            n_doa_samples=p['n_doa'],
            chemicalConc=p['counts'],
            H_fused=round(rf.map_entropy(self.p_fused), 3),
            trigger=round(trig, 4),
            peak_object=rf.peak_object_name(self),
            q_s_map=float(np.exp(self.olfHyp.map_value())) if self.olfHyp else None,
            L0_map=float(self.sndHyp.map_value()) if self.sndHyp else None,
            level_db=self._current_level_db(),
        ))
        # Only the arrays this run's modalities actually produced. A uniform
        # placeholder for a disabled branch wastes space and, worse, reads
        # later as "this modality ran and learned nothing" rather than "this
        # modality was switched off".
        np.savez_compressed(
            os.path.join(self.save_dir, f"maps_{self.step:03d}.npz"),
            **rf.belief_arrays_for_saving(self))
        rf.save_all_diagnostics(self, self.step)

    def save_data(self):
        pd.DataFrame(self.rows).to_csv(
            os.path.join(self.save_dir, 'trajectory_log.csv'), index=False)
        tx, tz = rf.fused_peak(self)
        meta = dict(platform='robuddy', ros_distro='humble',
                    modalities=self.modalities,
                    requested_modalities=self.requested_modalities,
                    steps=self.step + 1,
                    entropy_frac=self.entropy_frac,
                    estimated_source=dict(x=tx, y=tz),
                    peak_object=rf.peak_object_name(self),
                    mq3_baseline=self.mq3_baseline,
                    depth_rejected=self.n_depth_rejected,
                    goal_phrase=self.goal_phrase,
                    final_phase=self.phase,
                    doa_offset_deg=DOA_OFFSET_DEG, doa_ccw=DOA_CCW)
        if self.olfHyp is not None:
            meta['q_s_map'] = float(np.exp(self.olfHyp.map_value()))
            meta['q_s_posterior'] = [float(v) for v in self.olfHyp.hypothesis_posterior()]
            meta['q_s_pegged'] = self.olfHyp.at_endpoint()
        if self.sndHyp is not None:
            meta['L0_map'] = float(self.sndHyp.map_value())
            meta['L0_posterior'] = [float(v) for v in self.sndHyp.hypothesis_posterior()]
            meta['L0_pegged'] = self.sndHyp.at_endpoint()
        meta['has_sound_level'] = self.has_sound_level
        with open(os.path.join(self.save_dir, 'run_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        self.get_logger().info(
            f"estimated source at map ({tx:.2f}, {tz:.2f}) -> {self.save_dir}")

    def destroy_node(self):
        self._stop_evt.set()
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop(); self._audio_stream.close()
            except Exception:
                pass
        if self._doa_thread is not None:
            self._doa_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VAORobuddy()
    try:
        rclpy.spin(node)
    except SystemExit:
        # Normal end of run -- FINISH has already halted the base and written
        # every output, so there is nothing left to do here.
        node.get_logger().info("exiting cleanly")
    except KeyboardInterrupt:
        # Ctrl-C partway through: stop the robot and salvage whatever the run
        # produced rather than losing all of it.
        node.halt()
        if node.grid_ready and node.state != 'DONE':
            node.save_data()
    finally:
        node.destroy_node()
        # shutdown() raises if the context is already torn down, which happens
        # on some rclpy versions when SystemExit propagates through spin().
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()