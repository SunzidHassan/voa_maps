"""
fusion_controller.py  --  multi-robot VAO (Visual-Auditory-Olfactory) control loop.

Belief maps are SHARED across robots and modalities. Every robot's measurement
updates the same three maps, which are then fused in log space:

    log p_fused = w_V log p_V + w_O log p_O + w_A log p_A

so a two-robot run is the sequential Bayesian combination
p(source | z_robot0, z_robot1) rather than two independent searches.

MULTI-ROBOT IMPLEMENTATION NOTE
-------------------------------
AI2-THOR is driven here with a single agent that is teleported to each robot's
pose in turn within one search step ("virtual" multi-robot). The robots share
belief but cannot collide with or occlude each other. This keeps every existing
single-agent helper (last_event.depth_frame, coord23D_focal, boxDepth) working
unchanged. If physical embodiment matters, switch the Controller to
agentCount=2 and index event.events[i] -- but those helpers all read
last_event and would each need reworking first.
"""

import json
import time
import math
import numpy as np
import pandas as pd
import os
import cv2
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable

import networkx as nx
from shapely.geometry import Polygon
from skimage.morphology import closing, disk

from ...voa_functions.visionFunction import (
    visionBranch,
    initialize_envKnowledge,
    add_goal_similarity,
    init_object_map,
    dirichlet_entropy,
    visual_likelihood_multimodal,
    fuse_log_evidence,
    object_posterior,
    observed_mask,
    CLASSES,
    EMPTY_CLASS,
)

from ...voa_functions.olfactionFunctions import simChemicalReading

# Candidate source strengths, tracked online instead of assumed. Guessing is
# not safe: measured on a 4 m grid, an emission rate wrong by 10x moved
# localisation from 0.39 m to 2.58 m, and a loudness wrong by 15 dB from
# 0.18 m to 2.83 m. Both posteriors collapse onto the truth within 2-3
# readings, so a uniform prior is enough.
# FIVE hypotheses, deliberately spread, with the model noise inflated to match.
#
# The point is not to land on the true strength. A coarse grid leaves each
# single modality imprecise, and that is fine -- fusion is what sharpens the
# estimate. What matters is that an imprecise modality stays HONEST about its
# imprecision, because sigma sets its weight in the log-space fusion.
#
# Measured (5-point grids, true strength drawn at random, k = sigma inflation):
#
#   k   V     O     A   |  VO    VA    VAO
#   1  1.30  0.91  0.95 | 0.79  0.91  0.81    <- fusion WORSE than O or A alone
#   3  1.30  0.91  0.75 | 0.41  0.42  0.32    <- best
#   4  1.30  0.90  0.75 | 0.68  0.36  0.29
#  12  1.30  1.21  0.76 | 1.00  0.39  0.38    <- too wide, O stops contributing
#
# At k=1 the coarse-grid modalities are confidently wrong and drag the fused
# estimate BELOW what audition alone achieves. Inflating sigma makes them
# appropriately humble and every fused combination then beats every single
# modality. Sigma barely moves a single modality's own argmax -- it scales the
# log-likelihood almost uniformly -- so this is purely a fusion-weight setting.
# CONSEQUENCE FOR THE STOPPING THRESHOLD. Inflation buys accuracy by making
# the posterior broader, so the fused entropy no longer falls as far:
#
#   k    VO err  VO H   |  VAO err  VAO H      (H normalised, 1.0 = uniform)
#   1     0.80   0.551  |   0.77    0.454
#   3     0.46   0.897  |   0.31    0.683
#   6     0.72   0.978  |   0.28    0.712
#
# A VO run at k=3 settles around H = 0.90, so entropy_frac = 0.5 can never
# fire and the run burns its whole step budget. Raise entropy_frac for the
# two-modality sets, or rely on the arrival test (which is the condition that
# actually means "found it"). Suggested: 0.95 for VO/VA, 0.75 for VAO.
SIGMA_INFLATION = 3.0
# 1000 .. 5000 mg/L, log-spaced: 1000, 1495, 2236, 3344, 5000.
# Spacing is 1.50x here against 3.16x over the old 100..10000 span, so
# discretisation error is much smaller -- but the range only covers 5x, and
# anything outside it cannot be represented. The posterior then pegs at an
# endpoint and still reports converged(), so watch the endpoint warning that
# ScaleHypotheses prints. Remember the MQ3 gain is absorbed here too: what is
# being estimated is the product a*q_s, not the emission rate alone, so a
# sensor with unexpected gain can push the effective value outside this window
# even when the real emission rate is inside it.
Q_S_HYPOTHESES = tuple(10.0 ** np.linspace(3, np.log10(5000), 5))   # 1000 .. 5000 mg/L
L0_HYPOTHESES = tuple(np.linspace(45.0, 105.0, 5))     # 45 .. 105 dB at 1 m
OLF_SIGMA_LOG = 0.25 * SIGMA_INFLATION        # log-concentration
SND_SIGMA_DB = 2.5 * SIGMA_INFLATION          # dB

from ...voa_functions.navigation import cell_centers

from ...voa_functions.hypotheses import (
    ScaleHypotheses, log_space_grid, db_grid,
    olfactory_predictor, auditory_predictor,
)

from ...voa_functions.soundFunctions import (
    init_sound_map,
    update_sound_map,
    sound_posterior,
    sound_belief_map,
    simSoundReading,
    class_sound_similarity,
)

from ...voa_functions.utils import (
    parse_position_string,
    find_nearest_node,
    create_graph_from_positions,
    grid_to_world,
)

from ...voa_functions.loggerFunctions import map_entropy, plot_detected_objects

LABEL_FONTSIZE = 16

# Per-modality log-evidence weights (lambda_V, lambda_O, lambda_A).
W_VISION = 1.0
W_OLFACT = 1.0
W_SOUND = 1.0

# ---------------------------------------------------------------- TRIGGER
# ONE condition ends the search:
#
#     fused entropy <= H_max * entropy_frac
#
# When it fires the robot navigates to the current highest-belief object and
# the run terminates there. Arrival is not a separate gate.
#
# dist_to_target is still written to every log, so a "succeeded within X
# metres" criterion is applied in analysis rather than at runtime -- which
# means the radius can be varied without re-running anything, and a run that
# stopped 0.6 m away is recoverable data rather than a discarded failure.
#
# The earlier 'weighted' trigger (a weighted mean of per-modality normalised
# entropies) is also gone. Its visual term never moved -- measured 1.000 ->
# 0.998 over 200 detections, because the semantic map assigned the prior floor
# to every unobserved cell -- so it acted purely as a brake, and the VA and VO
# sets could not reach any threshold below 0.49 at all.
ARRIVAL_RADIUS = 0.5     # metres; reporting threshold only, not a gate

# ---------------------------------------------------------------- NAVIGATION
# Greedy: path to the top-ranked object (fused argmax as fallback), step one
# graph node per iteration.
#
# The information-gain planner has been removed. Benchmarked on this plume
# model it did not beat greedy for olfaction -- the field spans 3.9 to 15915,
# so it is monotone and walking uphill IS the informative move -- and it only
# won for bearing-only sensing under high DOA noise, a regime the scoped VO and
# VA experiments do not exercise. navigation.py still holds the
# implementation if it is ever needed.
NAV_RADIUS = 0.75        # metres; candidate poses within this range

ROBOT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']


class RobotState:
    """Pose and per-robot bookkeeping for one agent in the team."""

    def __init__(self, robot_id, x, z, y, yaw, save_dir):
        self.id = robot_id
        self.x, self.z, self.y, self.yaw = x, z, y, yaw
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.log = []
        self.trail = [(x, z)]
        self.listen_time = 0.0
        self.color = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
        self.last_reading = dict(conc=None, snd=dict(active=False, detected=False,
                                                     doa_deg=None, level_db=None,
                                                     distance=None, listen_s=0.0))

    def sync_from(self, controller):
        md = controller.last_event.metadata["agent"]
        self.x = md["position"]["x"]
        self.z = md["position"]["z"]
        self.y = md["position"]["y"]
        self.yaw = md["rotation"]["y"]
        self.trail.append((self.x, self.z))

    def activate(self, controller):
        """Teleport the shared AI2-THOR agent onto this robot's pose."""
        controller.step(action="Teleport",
                        position=dict(x=self.x, y=self.y, z=self.z),
                        rotation=dict(x=0, y=self.yaw, z=0))


def fusion_control(controller, itemDF, yolo_model, groundTruthSourcePosition,
                   save_dir,
                   step_threshold=100, goal_phrase="",
                   bayesian_agent=None, x_points=None, z_points=None,
                   alg_choice='F', entropy_frac=None, random_action_frac=0.0,
                   fusionMode=None,
                   start_positions=None,
                   sound_source=None,
                   modalities='VAO',
                   agent_y=0.9,
                   w_vision=W_VISION, w_olfact=W_OLFACT, w_sound=W_SOUND,
                   run_meta=None):
    """Multi-robot VAO search loop.

    Parameters
    ----------
    start_positions : list[tuple[float, float]]
        One (x, z) per robot. Team size is len(start_positions).
    sound_source : SoundSource or None
        None means the scene has no audible source (odor-only condition).
    modalities : str
        Any subset of 'VAO'. 'V' = semantic vision, 'O' = olfactory plume,
        'A' = auditory. Drives both sensing and fusion weights, so this is the
        single knob for the 7-way ablation.
    goal_phrase : str
        Odor description. Pass "" when the scene has no odor source.
    """
    use_V = 'V' in modalities.upper()
    use_O = 'O' in modalities.upper()
    use_A = 'A' in modalities.upper() and sound_source is not None
    has_odor = bool(goal_phrase) and use_O

    print(f"\n=== VAO control | modalities={modalities} | "
          f"V={use_V} O={use_O} A={use_A} | nav=greedy ===")

    srcPosX, srcPosY, srcPosZ = groundTruthSourcePosition[0]

    # ---------------- scene geometry (once) ----------------
    scene_bounds = controller.last_event.metadata["sceneBounds"]
    scene_polygon = Polygon([(p[0], p[2]) for p in scene_bounds['cornerPoints']])
    min_x, min_z, max_x, max_z = scene_polygon.bounds
    scene_bounds_tuple = (min_x, max_x, min_z, max_z)

    reachable_meta = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    if reachable_meta is None:
        print("Error: could not get reachable positions.")
        return
    reachable_positions_2d = [(p["x"], p["z"]) for p in reachable_meta]

    resolution = 0.2
    xg = np.arange(min_x, max_x + resolution, resolution)
    zg = np.arange(min_z, max_z + resolution, resolution)
    mask_closed = np.zeros((len(zg), len(xg)), dtype=bool)
    if reachable_positions_2d and len(xg) and len(zg):
        Xm, Zm = np.meshgrid(xg, zg)
        rp = np.array(reachable_positions_2d)
        m = np.zeros(Xm.shape, dtype=bool)
        for i in range(Xm.shape[0]):
            for j in range(Xm.shape[1]):
                if np.min(np.hypot(rp[:, 0] - Xm[i, j], rp[:, 1] - Zm[i, j])) < resolution * 1.5:
                    m[i, j] = True
        mask_closed = closing(m, disk(1))

    graph = create_graph_from_positions(reachable_meta, threshold=0.3)

    # Flattened cell centres, shared by every information-gain evaluation.
    cellX, cellZ = cell_centers(x_points, z_points)
    node_xz = np.array([[graph.nodes[n]['pos'][0], graph.nodes[n]['pos'][2]]
                        for n in graph.nodes]) if len(graph) else np.zeros((0, 2))

    # ---------------- shared belief maps ----------------
    objectMap = init_object_map(x_points, z_points)          # Dirichlet (H, W, K)
    soundLogBelief = init_sound_map(x_points, z_points)      # bearing log-belief
    gridX, gridZ = np.meshgrid(x_points, z_points)
    # Source strength is a nuisance parameter, not a constant. Same class for
    # both branches: after the right transform they are the same equation,
    # observed = strength + predictor(r) + noise.
    olfHyp = ScaleHypotheses(log_space_grid(Q_S_HYPOTHESES),
                             (len(z_points), len(x_points)), label='q_s') \
        if has_odor else None
    sndHyp = ScaleHypotheses(db_grid(L0_HYPOTHESES),
                             (len(z_points), len(x_points)), label='L0') \
        if use_A else None
    # (robot_x, robot_z, level_db) history. The marginal range term fits a
    # single unknown source level across ALL of these, so it needs the history
    # rather than an incremental update.
    soundObs = []
    uniform = np.full((len(z_points), len(x_points)),
                      1.0 / (len(z_points) * len(x_points)))

    # BayesianAgent is no longer the olfactory estimator -- ScaleHypotheses is,
    # because BayesianAgent hardcodes q_s inside gaussian_plume. It is still
    # accepted so main.py needs no change, and its sigma_noise is read below,
    # but its prob_map is only used as the initial uniform.
    srcProbGivenOlfactory = uniform.copy()
    srcProbGivenVision = uniform.copy()
    srcProbGivenSound = uniform.copy()
    fusedProbMap = uniform.copy()

    # entropy_frac is now a threshold on the NORMALISED trigger statistic in
    # [0, 1], so the same value means the same thing across scenes of different
    # grid sizes -- previously it scaled with log2(n_cells) per scene.
    H_max = map_entropy(uniform)
    print(f"Max entropy {H_max:.2f} bits | stop when normalised fused entropy "
          f"<= {entropy_frac}, then navigate to the top-ranked object")

    olfactoryEntropy = visionEntropy = soundEntropy = fused_entropy = H_max
    semanticEntropy = float(dirichlet_entropy(objectMap).mean())
    trigger_stat, trigger_parts = 1.0, {}

    # CLAP similarity is resolved once, the first time any robot actually hears
    # the source. Before that the auditory semantic cue does not exist.
    sound_sim = None
    heard_once = False

    # ---------------- robot team ----------------
    if not start_positions:
        md = controller.last_event.metadata["agent"]["position"]
        start_positions = [(md["x"], md["z"])]
    robots = [RobotState(i, sx, sz, agent_y, 180.0,
                         os.path.join(save_dir, f"robot{i}"))
              for i, (sx, sz) in enumerate(start_positions)]
    print(f"Team size: {len(robots)}  starts: {[(round(r.x, 2), round(r.z, 2)) for r in robots]}")

    def _meta(step, flag):
        """Self-describing run record, so summaries never parse folder names."""
        m = dict(run_meta or {})
        m.update(modalities=modalities, nav_mode='greedy',
                 trigger_mode='fused', entropy_frac=entropy_frac,
                 team_size=len(robots), steps=step + 1, behavior_flag=flag,
                 terminated=(flag != "step_limit_reached"),
                 source_x=float(srcPosX), source_z=float(srcPosZ))
        if olfHyp is not None:
            m['q_s_map'] = float(np.exp(olfHyp.map_value()))
            m['q_s_posterior'] = [float(v) for v in olfHyp.hypothesis_posterior()]
            m['q_s_converged'] = olfHyp.converged()
        if sndHyp is not None:
            m['L0_map'] = float(sndHyp.map_value())
            m['L0_posterior'] = [float(v) for v in sndHyp.hypothesis_posterior()]
            m['L0_converged'] = sndHyp.converged()
        return m

    envKnowledge = itemDF.copy()
    navKnowledge = pd.DataFrame()
    step_count = 0
    behavior_flag = "Initialization"
    team_rows = []

    def _fuse():
        maps, ws = [], []
        if use_V:
            maps.append(srcProbGivenVision); ws.append(w_vision)
        if has_odor:
            maps.append(srcProbGivenOlfactory); ws.append(w_olfact)
        if use_A:
            maps.append(srcProbGivenSound); ws.append(w_sound)
        if not maps:
            return uniform.copy()
        return fuse_log_evidence(maps, ws)

    while True:
        step_start = time.time()
        print(f"\n===== Step {step_count + 1}/{step_threshold} =====")

        if step_count >= step_threshold - 1:
            behavior_flag = "step_limit_reached"
            print("Step limit reached.")
            _flush_logs(robots, team_rows, save_dir, _meta(step_count, behavior_flag))
            break

        # ================= SENSING (every robot, shared maps) =================
        step_listen_s = 0.0
        for rb in robots:
            rb.activate(controller)
            rb.sync_from(controller)

            # --- olfaction ---
            conc = None
            if has_odor:
                conc = simChemicalReading((srcPosX, srcPosZ), rb.x, rb.z)
                # BayesianAgent bakes in a fixed q_s; the tracker carries it
                # as an unknown instead. A reading at or below zero is a sensor
                # dropout, not a concentration, so it is skipped rather than
                # log-transformed.
                if conc is not None and conc > 0:
                    olfHyp.update(olfactory_predictor(gridX, gridZ, rb.x, rb.z),
                                  float(np.log(conc)), OLF_SIGMA_LOG)

            # --- audition ---
            snd = dict(active=False, detected=False, doa_deg=None,
                       level_db=None, distance=None, listen_s=0.0)
            if use_A:
                snd = simSoundReading(sound_source, rb.x, rb.z, rb.yaw, step_count)
                # A discrete clip must be heard in full, so the whole team pays
                # the dwell cost of the longest listen this step, once.
                step_listen_s = max(step_listen_s, snd['listen_s'])
                rb.listen_time += snd['listen_s']
                if snd['detected']:
                    # Pass the measured level: it pins the source to an arc
                    # rather than a full ray, which is what stops several rays
                    # crossing on empty floor and out-voting the true source.
                    # One update from the burst MEAN with its effective sigma,
                    # not one per sample. Measured: folding in all 20 samples
                    # as independent gives the same accuracy (1.28 m vs 1.23 m)
                    # but drives normalised entropy to 0.467 instead of 0.774
                    # -- pure overconfidence, and the thing that makes the
                    # fused trigger fire before vision has found anything.
                    update_sound_map(soundLogBelief, x_points, z_points,
                                     rb.x, rb.z, rb.yaw, snd['doa_deg'],
                                     sigma_deg=snd.get('doa_sigma_eff'))
                    if snd['level_db'] is not None:
                        soundObs.append((rb.x, rb.z, float(snd['level_db'])))
                        sndHyp.update(auditory_predictor(gridX, gridZ, rb.x, rb.z),
                                      float(snd['level_db']), SND_SIGMA_DB)
                    if not heard_once:
                        heard_once = True
                        sound_sim = class_sound_similarity(sound_source.query(), CLASSES)
                        top = np.argsort(-sound_sim)[:3]
                        print("[sound] first detection; CLAP top classes: "
                              f"{[(CLASSES[i], round(float(sound_sim[i]), 3)) for i in top]}")

            # --- vision ---
            if use_V:
                if step_count == 0:
                    envKnowledge = initialize_envKnowledge(
                        controller=controller, model=yolo_model, itemDF=envKnowledge,
                        save_path=rb.save_dir, confThr=0.5, fusionMode=fusionMode,
                        beta=objectMap, x_points=x_points, z_points=z_points)
                    rb.sync_from(controller)
                else:
                    envKnowledge = visionBranch(
                        yolo_model, envKnowledge, controller, rb.save_dir, step_count,
                        fusionMode=fusionMode, beta=objectMap,
                        x_points=x_points, z_points=z_points,
                        robot_pose=(rb.x, rb.z, rb.yaw))

            rb.last_reading = dict(conc=conc, snd=snd)

        # ================= BELIEF UPDATE (team-combined) =================
        if has_odor:
            srcProbGivenOlfactory = olfHyp.cell_posterior()
        if use_V:
            srcProbGivenVision = visual_likelihood_multimodal(
                objectMap,
                goal_phrase=goal_phrase if has_odor else None,
                sound_sim=sound_sim if use_A else None)
            semanticEntropy = float(dirichlet_entropy(objectMap).mean())
        if use_A:
            # Bearing (incremental) + range from the level, with L0 unknown.
            total = soundLogBelief + sndHyp.cell_loglik()
            srcProbGivenSound = sound_posterior(total)

        fusedProbMap = _fuse()

        olfactoryEntropy = map_entropy(srcProbGivenOlfactory)
        visionEntropy = map_entropy(srcProbGivenVision)
        soundEntropy = map_entropy(srcProbGivenSound)
        fused_entropy = map_entropy(fusedProbMap)

        trigger_stat = fused_entropy / max(H_max, 1e-9)
        if olfHyp is not None:
            print(f"   q_s posterior {np.round(olfHyp.hypothesis_posterior(), 3)} "
                  f"MAP {np.exp(olfHyp.map_value()):.0f} mg/L "
                  f"{'(converged)' if olfHyp.converged() else ''}"
                  f"{'  !! pegged at grid edge -- true value likely outside Q_S_HYPOTHESES' if olfHyp.at_endpoint() else ''}")
        if sndHyp is not None:
            print(f"   L0  posterior {np.round(sndHyp.hypothesis_posterior(), 3)} "
                  f"MAP {sndHyp.map_value():.0f} dB "
                  f"{'(converged)' if sndHyp.converged() else ''}"
                  f"{'  !! pegged at grid edge -- true value likely outside L0_HYPOTHESES' if sndHyp.at_endpoint() else ''}")
        print(f"H: olf={olfactoryEntropy:.2f} vis={visionEntropy:.2f} "
              f"snd={soundEntropy:.2f} fused={fused_entropy:.2f} "
              f"dir={semanticEntropy:.2f}  ->  normalised {trigger_stat:.3f} "
              f"(need <= {entropy_frac})")

        # Object ranking uses the FUSED map, so it reflects every active
        # modality and both robots rather than olfaction alone.
        # The ranking phrase must follow the ACTIVE modalities, not what the
        # scene happens to contain. In a VA run on a scene that also emits odor,
        # goal_phrase is still populated by main.py -- using it here would rank
        # objects by odor semantics in a run where olfaction is ablated, quietly
        # leaking the modality the ablation is meant to remove.
        if has_odor:
            rank_phrase = goal_phrase
        elif use_A and sound_source is not None:
            rank_phrase = sound_source.label
        else:
            rank_phrase = ""
        navKnowledge = add_goal_similarity(envKnowledge, rank_phrase, fusedProbMap,
                                           x_points, z_points, alg_choice=alg_choice)

        # ================= PLOTS =================
        _save_maps(save_dir, step_count, robots, x_points, z_points,
                   srcProbGivenOlfactory, srcProbGivenVision, srcProbGivenSound,
                   fusedProbMap, olfactoryEntropy, visionEntropy, soundEntropy,
                   fused_entropy, use_A, (srcPosX, srcPosZ),
                   navKnowledge=navKnowledge, use_O=has_odor, objectMap=objectMap)

        try:
            plot_detected_objects(itemDF=envKnowledge, mask_closed=mask_closed,
                                  scene_bounds_tuple=scene_bounds_tuple,
                                  save_path=os.path.join(
                                      save_dir, f"detected_objects_{step_count:03d}.png"))
        except Exception as e:
            print(f"Object map plot failed: {e}")

        target_error = np.nan
        if not navKnowledge.empty:
            pp = parse_position_string(navKnowledge.iloc[0]["Position"])
            target_error = float(np.linalg.norm(np.array([srcPosX, srcPosZ]) -
                                                np.array([pp[0], pp[2]])))

        step_time = time.time() - step_start + step_listen_s
        _log_step(robots, team_rows, step_count, behavior_flag, navKnowledge,
                  target_error, step_time, srcPosX, srcPosZ,
                  olfactoryEntropy, visionEntropy, soundEntropy,
                  fused_entropy, semanticEntropy, trigger_stat, trigger_parts)

        # ================= TERMINATION =================
        # TERMINATION: a single condition. Once the fused belief has
        # concentrated -- fused entropy below H_max * entropy_frac -- drive to
        # the current highest-belief object and stop.
        #
        # There is no separate arrival gate. dist_to_target is still logged so
        # a "succeeded within X metres" criterion can be applied afterwards in
        # summary; making it an analysis choice rather than a runtime one
        # means the radius can be varied without re-running anything.
        if step_count > 0 and trigger_stat <= entropy_frac:
            behavior_flag = "goal_navigation"
            name, tx, tz = robot_target(navKnowledge, fusedProbMap,
                                        x_points, z_points, 0)
            print(f"Entropy {trigger_stat:.3f} <= {entropy_frac}; "
                  f"navigating to {name} at ({tx:.2f}, {tz:.2f}) and terminating.")
            _drive_to_targets(controller, robots, graph, navKnowledge, fusedProbMap,
                              x_points, z_points, final=True)

            dists = [float(np.hypot(rb.x - tx, rb.z - tz)) for rb in robots]
            for rb, dd in zip(robots, dists):
                rb.log[-1].update(robot_x=rb.x, robot_z=rb.z, robot_yaw=rb.yaw,
                                  behavior_flag=behavior_flag,
                                  dist_to_target=round(dd, 3),
                                  gt_distance_from_source=float(
                                      np.hypot(rb.x - srcPosX, rb.z - srcPosZ)))
            team_rows[-1].update(dist_to_target=round(min(dists), 3),
                                 target_object=name)
            print(f"Closest robot ended {min(dists):.2f} m from the target, "
                  f"{min(np.hypot(rb.x - srcPosX, rb.z - srcPosZ) for rb in robots):.2f} m "
                  f"from the true source.")
            _flush_logs(robots, team_rows, save_dir, _meta(step_count, behavior_flag))
            break

        # ================= ACTION =================
        behavior_flag = "search"
        _drive_to_targets(controller, robots, graph, navKnowledge, fusedProbMap,
                          x_points, z_points, final=False)
        step_count += 1

    print("VAO control loop finished.")


def robot_target(navKnowledge, fusedProbMap, x_points, z_points, rank=0):
    """The object this robot is driving to: (name, x, z).

    SINGLE SOURCE OF TRUTH. Both the navigation step and the map annotation
    call this, so the object drawn on the fused map is by construction the one
    the robot is actually moving towards. Duplicating the rank logic would let
    the two drift apart, and the drawing would quietly become a lie.

    Robot r takes the r-th ranked object so a team spreads over hypotheses;
    with fewer objects than robots the extras fall back to the last one, and
    with no objects at all to the fused argmax.
    """
    if navKnowledge is not None and not navKnowledge.empty:
        idx = min(int(rank), len(navKnowledge) - 1)
        try:
            p = parse_position_string(navKnowledge.iloc[idx]["Position"])
            return str(navKnowledge.iloc[idx]["objectType"]), float(p[0]), float(p[2])
        except Exception:
            pass
    gi = np.unravel_index(np.argmax(fusedProbMap), fusedProbMap.shape)
    w = grid_to_world(gi, x_points, z_points)
    return "fused argmax", float(w[0]), float(w[1])


def _log_step(robots, team_rows, step_count, behavior_flag, navKnowledge,
              target_error, step_time, srcPosX, srcPosZ,
              h_olf, h_vis, h_snd, h_fused, h_sem,
              trigger_stat=np.nan, trigger_parts=None):
    trigger_parts = trigger_parts or {}
    for rb in robots:
        gt = float(np.hypot(rb.x - srcPosX, rb.z - srcPosZ))
        snd = rb.last_reading['snd']
        rb.log.append({
            "step": step_count, "robot_id": rb.id,
            "robot_x": rb.x, "robot_z": rb.z, "robot_yaw": rb.yaw,
            "step_time": step_time, "listen_s": snd['listen_s'],
            "behavior_flag": behavior_flag, "is_random": False,
            "target_object": navKnowledge.iloc[0]["objectType"] if not navKnowledge.empty else 'N/A',
            "target_coordinate": navKnowledge.iloc[0]["Position"] if not navKnowledge.empty else 'N/A',
            "target_coord_estimation_error": target_error,
            "concentration": rb.last_reading['conc'],
            "sound_active": snd['active'], "sound_detected": snd['detected'],
            "sound_doa_deg": snd['doa_deg'], "sound_level_db": snd['level_db'],
            "n_doa_samples": snd.get('n_doa_samples'),
            "doa_sigma_eff": snd.get('doa_sigma_eff'),
            "gt_distance_from_source": gt,
            "Bayesian_entropy": h_olf, "visual_entropy": h_vis,
            "sound_entropy": h_snd, "fused_entropy": h_fused,
            "semantic_entropy": h_sem, "trigger_stat": trigger_stat,
            "trigger_V": trigger_parts.get('V', np.nan),
            "trigger_O": trigger_parts.get('O', np.nan),
            "trigger_A": trigger_parts.get('A', np.nan),
        })
    team_rows.append({
        "step": step_count, "fused_entropy": h_fused,
        "Bayesian_entropy": h_olf, "visual_entropy": h_vis,
        "sound_entropy": h_snd, "semantic_entropy": h_sem,
        "target_coord_estimation_error": target_error, "step_time": step_time,
        "min_gt_distance": min(float(np.hypot(r.x - srcPosX, r.z - srcPosZ)) for r in robots),
        "any_sound_detected": any(r.last_reading['snd']['detected'] for r in robots),
        "trigger_stat": trigger_stat,
        "trigger_V": trigger_parts.get('V', np.nan),
        "trigger_O": trigger_parts.get('O', np.nan),
        "trigger_A": trigger_parts.get('A', np.nan),
    })


def _drive_to_targets(controller, robots, graph, navKnowledge, fusedProbMap,
                      x_points, z_points, final=False):
    """Move every robot one step, either greedily or by information gain.

    'greedy' keeps the original behaviour: robot r heads for the r-th ranked
    object (falling back to the fused argmax) and advances one graph node along
    the Dijkstra path.

    With final=True the robot goes all the way to the target node instead of
    advancing a single step, which is the approach phase after the belief has
    already collapsed.
    """
    for rb in robots:
        rb.activate(controller)

        _, tgx, tgz = robot_target(navKnowledge, fusedProbMap,
                                   x_points, z_points, rank=rb.id)
        target_xz = (tgx, tgz)

        start_node, _ = find_nearest_node(graph, dict(x=rb.x, y=rb.y, z=rb.z))
        end_node, _ = find_nearest_node(graph, (target_xz[0], rb.y, target_xz[1]))
        if start_node is None or end_node is None:
            continue

        try:
            path = nx.dijkstra_path(graph, start_node, end_node, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [start_node]

        if final:
            nav = graph.nodes[end_node]['pos']
        elif len(path) > 1:
            nav = graph.nodes[path[1]]['pos']
        else:
            nav = (rb.x, rb.y, rb.z)

        yaw = math.degrees(math.atan2(target_xz[0] - nav[0], target_xz[1] - nav[2])) % 360
        try:
            controller.step(action="Teleport",
                            position=dict(x=nav[0], y=rb.y, z=nav[2]),
                            rotation=dict(x=0, y=yaw, z=0), raise_for_failure=True)
            rb.sync_from(controller)
        except Exception as e:
            print(f"Robot {rb.id} teleport failed: {e}")
    return {}


def _cell_object_name(objectMap, row, col):
    """What the object map believes occupies this cell.

    Distinguishes three cases that matter when reading a peak label:
      - a class name, when a detection has landed there
      - "empty floor", when the robot looked and saw nothing
      - "unobserved", when no evidence of any kind has reached the cell
    The last is the important one: a peak on an unobserved cell means the
    modality is pointing somewhere vision has not yet checked, which is a
    reason to keep going rather than to stop.
    """
    if objectMap is None:
        return "?"
    try:
        if not observed_mask(objectMap)[row, col]:
            return "unobserved"
        k = int(np.argmax(object_posterior(objectMap)[row, col]))
        return CLASSES[k] if k < len(CLASSES) else "?"
    except Exception:
        return "?"


def _save_maps(save_dir, step_count, robots, x_points, z_points,
               p_olf, p_vis, p_snd, p_fused, h_olf, h_vis, h_snd, h_fused,
               use_A, src_xz, navKnowledge=None, use_O=True, objectMap=None):
    """Combined team belief maps: every panel aggregates both robots' evidence."""
    try:
        # Only draw panels for modalities this run actually uses. An ablated
        # branch still holds a uniform map, and rendering it produces a flat
        # panel that reads as a failed sensor rather than a disabled one.
        panels = []
        if use_O:
            panels.append((p_olf, rf'Olfactory  $H={h_olf:.2f}$'))
        panels.append((p_vis, rf'Visual (semantic)  $H={h_vis:.2f}$'))
        if use_A:
            panels.append((p_snd, rf'Auditory (DOA)  $H={h_snd:.2f}$'))
        panels.append((p_fused, rf'Fused  $H={h_fused:.2f}$'))

        extent = [min(x_points), max(x_points), min(z_points), max(z_points)]
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        for ax, (arr, title) in zip(axes, panels):
            a = np.asarray(arr, float)
            lo, hi = a.min(), a.max()
            norm = (a - lo) / (hi - lo) if hi > lo + 1e-12 else np.zeros_like(a)
            ax.imshow(norm, origin='lower', extent=extent, aspect='equal', cmap='hot')
            ax.set_title(title, fontsize=LABEL_FONTSIZE)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('z (m)')
            ax.plot(src_xz[0], src_xz[1], marker='*', ms=16,
                    mfc='white', mec='black', mew=1.0, ls='')

            # Peak of THIS map: the cell this modality alone considers most
            # likely. Skipped when the map is flat, because argmax on a uniform
            # array returns cell (0, 0) -- an inactive modality would otherwise
            # be drawn as confidently pointing at a corner.
            if hi > lo + 1e-12:
                gi = np.unravel_index(np.argmax(a), a.shape)
                px, pz = float(x_points[gi[1]]), float(z_points[gi[0]])
                pv = float(a[gi] / max(a.sum(), 1e-12))
                ax.plot(px, pz, marker='o', ms=15, mfc='none', mec='#39ff14',
                        mew=2.2, ls='', zorder=6)
                ax.plot(px, pz, marker='+', ms=8, mec='#39ff14', mew=1.4,
                        ls='', zorder=6)
                xmid = 0.5 * (min(x_points) + max(x_points))
                zmid = 0.5 * (min(z_points) + max(z_points))
                dxo = -12 if px > xmid else 12
                dzo = -14 if pz > zmid else 12
                pname = _cell_object_name(objectMap, gi[0], gi[1])
                ax.annotate(f"peak: {pname}\n({px:.2f}, {pz:.2f})  p={pv:.3f}",
                            xy=(px, pz), xytext=(dxo, dzo),
                            textcoords='offset points',
                            ha='right' if dxo < 0 else 'left',
                            va='top' if dzo < 0 else 'bottom',
                            fontsize=7, color='#39ff14', fontweight='bold',
                            annotation_clip=False,
                            path_effects=[pe.withStroke(linewidth=2.2,
                                                        foreground='black')])

            for rb in robots:
                tr = np.array(rb.trail)
                ax.plot(tr[:, 0], tr[:, 1], '-', lw=1.4, color=rb.color, alpha=0.85)
                ax.plot(rb.x, rb.z, marker='o', ms=8, mfc=rb.color,
                        mec='white', mew=1.2, ls='', label=f'R{rb.id}')
            ax.grid(True, ls='--', lw=0.4, alpha=0.5)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            sm = cm.ScalarMappable(cmap='hot', norm=mcolors.Normalize(0, 1))
            sm.set_array([])
            plt.colorbar(sm, cax=cax)

        # Mark the TARGET each robot is driving to this step, on the fused
        # panel only. Drawn via robot_target(), the same call the navigation
        # uses, so the labelled object is by construction the one being
        # approached rather than a re-derived guess.
        if navKnowledge is not None and p_fused is not None:
            ax = axes[-1]
            for rb in robots:
                name, ox, oz = robot_target(navKnowledge, p_fused,
                                            x_points, z_points, rank=rb.id)
                ax.plot(ox, oz, marker='D', ms=11, mfc=rb.color, mec='white',
                        mew=1.6, ls='', zorder=5)
                ax.plot([rb.x, ox], [rb.z, oz], ls=':', lw=1.4,
                        color=rb.color, alpha=0.9, zorder=4)
                # Flip the label inward near an edge so it does not run off the
                # axes and over the colorbar.
                xmid = 0.5 * (min(x_points) + max(x_points))
                zmid = 0.5 * (min(z_points) + max(z_points))
                dx = -10 if ox > xmid else 10
                dz = -12 if oz > zmid else 10
                ax.annotate(f"R{rb.id} -> {name}\n({ox:.2f}, {oz:.2f})",
                            xy=(ox, oz), xytext=(dx, dz),
                            textcoords='offset points',
                            ha='right' if dx < 0 else 'left',
                            va='top' if dz < 0 else 'bottom',
                            fontsize=8, fontweight='bold', color=rb.color,
                            annotation_clip=True,
                            path_effects=[pe.withStroke(linewidth=2.4,
                                                        foreground='white')])

        # Legend at figure level, not inside a panel -- an in-axes legend sat
        # in the top-right corner and covered the peak label whenever the
        # argmax landed there, which is exactly the case worth seeing.
        handles, labels = axes[0].get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        if h2:
            fig.legend(h2, l2, loc='lower center', ncol=max(len(h2), 2),
                       fontsize=9, frameon=False)
        fig.savefig(os.path.join(save_dir, f"maps_team_{step_count:03d}.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"Map plot failed at step {step_count}: {e}")
        plt.close('all')


def _flush_logs(robots, team_rows, save_dir, meta=None):
    """Write per-robot logs plus team-level logs.

    trajectory_log.csv holds robot 0 only, because calculate_total_distance
    differences consecutive robot_x/robot_z rows -- interleaving both robots
    there would report a meaningless zig-zag distance. Team totals live in
    team_log.csv and team_summary.csv.
    """
    frames = []
    for rb in robots:
        df = pd.DataFrame(rb.log)
        df.to_csv(os.path.join(save_dir, f"trajectory_log_robot{rb.id}.csv"), index=False)
        if rb.id == 0:
            df.to_csv(os.path.join(save_dir, "trajectory_log.csv"), index=False)
        frames.append(df)

    if team_rows:
        pd.DataFrame(team_rows).to_csv(os.path.join(save_dir, "team_log.csv"), index=False)

    if meta is not None:
        with open(os.path.join(save_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        return

    per_robot_dist = []
    for rb in robots:
        d = pd.DataFrame(rb.log)
        per_robot_dist.append(float(np.nansum(np.hypot(d['robot_x'].diff(),
                                                       d['robot_z'].diff()))))
    pd.DataFrame([{
        "team_size": len(robots),
        "steps": int(combined['step'].max()) + 1,
        "team_distance": round(float(np.nansum(per_robot_dist)), 3),
        "per_robot_distance": str([round(d, 3) for d in per_robot_dist]),
        "total_listen_s": round(sum(r.listen_time for r in robots), 2),
        "final_target_error": combined['target_coord_estimation_error'].iloc[-1],
        "final_min_gt_distance": combined['gt_distance_from_source'].min(),
        "final_target_object": combined['target_object'].iloc[-1],
        "terminated": (meta or {}).get("terminated", None),
    }]).to_csv(os.path.join(save_dir, "team_summary.csv"), index=False)