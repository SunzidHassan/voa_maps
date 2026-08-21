"""
main.py  --  VAO (Visual-Auditory-Olfactory) multi-robot evaluation driver.

THREE TEST CONDITIONS
---------------------
  odor_only   FloorPlan1   (kitchen)  GarbageCan  -- continuous "rotten food smell",
                                                     silent.
  sound_only  FloorPlan301 (bedroom)  Clock       -- intermittent alarm, odourless.
                                                     Semantics come from CLAP
                                                     (sound vs class names), not
                                                     from an odor phrase.
  both        FloorPlan401 (bathroom) Toilet      -- continuous "human waste smell"
                                                     plus an intermittent flush.
                                                     SBERT and CLAP similarity are
                                                     combined per class.

Each condition runs TWO robots. Robot 0 starts from row i of the first positions
CSV, robot 1 from row i of the second. Both sense every step; all belief maps are
shared and fused in log space, so the saved maps are the team's combined output.
"""

import os
import gc
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ai2thor.controller import Controller
from ultralytics import YOLO

from ControlAlgorithms.AI2Thor.ai2thor_controller import fusion_control
# from ControlAlgorithms.RandomWalk.random_controller import randomWalk_control

from voa_functions.utils import world_to_grid, get_iThor_object_centers
from voa_functions.olfactionFunctions import BayesianAgent
from voa_functions.soundFunctions import SoundSource
from voa_functions.loggerFunctions import (
    generate_trajectory_plot,
    generate_batch_summary,
)
from voa_functions.summary import generate_summary

yolo_model = YOLO("models/YOLO/yolo26mFT2.pt")

# ============================================================= CONFIG

STEP_THRESHOLD = 100
ENTROPY_FRACS = [0.7]
SIGMA_NOISE = 20
GRID_STEP = 0.25

# Full ablation: ['VAO', 'VA', 'VO', 'AO', 'V', 'A', 'O']
MODALITY_SETS = ['VAO', 'VA', 'VO']

# Set to a {condition_name: [modalities]} dict to pin specific modality sets to
# specific scenes instead of running the full grid. None = full grid.
#   e.g. {'sound_only': ['VA', 'A'], 'odor_and_sound': ['VAO']}
CONDITION_MODALITIES = {'sound_only': ['VA'], 'odor_and_sound': ['VAO'], 'odor_only': ['VO']}

# ------------------------------------------------------------- CONDITIONS
#
# `sound` is None for a silent scene. `odor` is "" for an odourless scene.
# `agent_y` is the AI2-THOR floor height for that scene type.
CONDITIONS = [
    {
        "name": "odor_only",
        "scene": "FloorPlan1",
        "agent_y": 0.9,
        "target_items": ["GarbageCan_a3dd7762"],
        "odor": "rotten food smell",
        "sound": None,
        "position_csvs": ["simulation_configs/GarbageCanPositions_1.csv",
                          "simulation_configs/GarbageCanPositions_2.csv"],
    },
    {
        "name": "sound_only",
        "scene": "FloorPlan301",
        "agent_y": 0.91,
        "target_items": ["AlarmClock_b6b90d65"],
        "odor": "",                       # odourless -> no olfactory branch
        "sound": {
            "label": "an alarm clock ringing",
            "audio_path": "assets/sounds/clock_alarm.wav",   # ESC-50 clip if present
            "interval_steps": 3,          # discrete: emits every 3rd step
            "continuous": False,
            "L0_db": 70.0,
        },
        "position_csvs": ["simulation_configs/ClockPositions_1.csv",
                          "simulation_configs/ClockPositions_2.csv"],
    },
    {
        "name": "odor_and_sound",
        "scene": "FloorPlan401",
        "agent_y": 0.91,
        "target_items": ["Toilet_5404df9d"],
        "odor": "human waste smell",
        "sound": {
            "label": "a toilet flushing",
            "audio_path": "assets/sounds/toilet_flush.wav",
            "interval_steps": 4,
            "continuous": False,
            "L0_db": 75.0,
        },
        "position_csvs": ["simulation_configs/ToiletPositions_1.csv",
                          "simulation_configs/ToiletPositions_2.csv"],
    },
]


def load_start_pairs(csv_paths):
    """Read one positions CSV per robot and zip them into per-run start pairs.

    Run i places robot 0 at row i of the first CSV and robot 1 at row i of the
    second, so the two robots always approach from different sides. The number
    of runs is bounded by the shorter file.
    """
    dfs = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"  Missing positions CSV: {p}")
            return []
        d = pd.read_csv(p)
        if not {'x', 'z'}.issubset(d.columns):
            print(f"  {p} must have 'x' and 'z' columns.")
            return []
        dfs.append(d)

    n = min(len(d) for d in dfs)
    if any(len(d) != n for d in dfs):
        print(f"  Position files differ in length; using the first {n} rows of each.")
    return [[(float(d.iloc[i]['x']), float(d.iloc[i]['z'])) for d in dfs]
            for i in range(n)]


def build_sound_source(cond, source_xz):
    """Instantiate the SoundSource at the ground-truth object position."""
    spec = cond["sound"]
    if spec is None:
        return None
    path = spec.get("audio_path")
    if path and not os.path.exists(path):
        print(f"  Audio clip {path} not found; CLAP will embed the text label instead.")
        path = None
    return SoundSource(position=source_xz,
                       label=spec["label"],
                       audio_path=path,
                       interval_steps=spec.get("interval_steps", 3),
                       continuous=spec.get("continuous", False),
                       L0_db=spec.get("L0_db", 70.0))


def build_grid(controller):
    """Grid axes from the scene bounds, snapped to GRID_STEP."""
    sb = controller.last_event.metadata["sceneBounds"]
    cp = np.floor(np.array(sb["cornerPoints"]) / GRID_STEP) * GRID_STEP
    x_min, x_max = np.round(cp[:, 0].min(), 2), np.round(cp[:, 0].max(), 2)
    z_min, z_max = np.round(cp[:, 2].min(), 2), np.round(cp[:, 2].max(), 2)
    return (np.arange(x_min, x_max + GRID_STEP, GRID_STEP),
            np.arange(z_min, z_max + GRID_STEP, GRID_STEP), sb)


def run_condition(cond, modalities):
    """Execute every entropy threshold x start-pair run for one condition."""
    print(f"\n{'=' * 70}\nCONDITION: {cond['name']}  scene={cond['scene']}  "
          f"target={cond['target_items']}  modalities={modalities}\n{'=' * 70}")

    start_pairs = load_start_pairs(cond["position_csvs"])
    if not start_pairs:
        print(f"Skipping {cond['name']}: no usable start positions.")
        return
    print(f"{len(start_pairs)} runs x {len(start_pairs[0])} robots")

    controller = Controller(
        agentMode="default", visibilityDistance=1.5, scene=cond["scene"],
        gridSize=0.25, snapToGrid=True, rotateStepDegrees=90,
        renderDepthImage=True, renderInstanceSegmentation=True,
        width=300, height=300, fieldOfView=90)

    alg_folder = f"save_{modalities}_{cond['name']}"

    try:
        for entropy_frac in ENTROPY_FRACS:
            base_save_dir = os.path.join("save/simulation", alg_folder, str(entropy_frac))
            os.makedirs(base_save_dir, exist_ok=True)

            for run_idx, starts in enumerate(start_pairs):
                run_serial = run_idx + 1
                run_dir = os.path.join(base_save_dir, str(run_serial))
                os.makedirs(run_dir, exist_ok=True)

                print(f"\n--- {cond['name']} | {modalities} | frac={entropy_frac} "
                      f"| RUN {run_serial}/{len(start_pairs)} ---")
                print(f"    starts: {starts}")

                random.seed(run_serial)
                np.random.seed(run_serial)
                controller.reset(scene=cond["scene"])

                # Place the shared agent at robot 0's start so the first
                # last_event reflects a valid in-scene pose.
                try:
                    controller.step(action="Teleport",
                                    position=dict(x=starts[0][0], y=cond["agent_y"],
                                                  z=starts[0][1]),
                                    rotation=dict(x=0, y=180, z=0),
                                    raise_for_failure=True)
                    controller.step("MoveAhead", moveMagnitude=0.01)
                except (ValueError, RuntimeError) as e:
                    print(f"    Teleport failed ({e}); skipping run.")
                    continue

                x_points, z_points, scene_bounds = build_grid(controller)

                objects = controller.last_event.metadata["objects"]
                sourcePos = get_iThor_object_centers(objects, cond["target_items"])
                if sourcePos.size == 0:
                    print(f"    Target {cond['target_items']} not in {cond['scene']}; "
                          f"skipping run.")
                    continue
                x_src, y_src, z_src = sourcePos[0]

                sound_source = build_sound_source(cond, (x_src, z_src))
                if sound_source:
                    print(f"    {sound_source}")

                reachable = controller.step(
                    action="GetReachablePositions").metadata["actionReturn"]
                if reachable is None:
                    print("    No reachable positions; skipping run.")
                    continue

                bayesian_agent = BayesianAgent(
                    pos=world_to_grid(starts[0][0], starts[0][1], x_points, z_points),
                    src_pos=world_to_grid(x_src, z_src, x_points, z_points),
                    x_points=x_points, z_points=z_points,
                    sigma_noise=SIGMA_NOISE,
                    reachable_positions=[(p["x"], p["z"]) for p in reachable])

                fusion_control(
                    controller=controller,
                    itemDF=pd.DataFrame(),
                    yolo_model=yolo_model,
                    groundTruthSourcePosition=sourcePos,
                    save_dir=run_dir,
                    step_threshold=STEP_THRESHOLD,
                    goal_phrase=(f"Is emitting {cond['odor']} odor:"
                                 if cond["odor"] else ""),
                    bayesian_agent=bayesian_agent,
                    x_points=x_points, z_points=z_points,
                    alg_choice='F',
                    entropy_frac=entropy_frac,
                    fusionMode='focal',
                    start_positions=starts,
                    sound_source=sound_source,
                    modalities=modalities,
                    agent_y=cond["agent_y"],
                    run_meta=dict(condition=cond["name"], scene=cond["scene"],
                                  target_items=cond["target_items"],
                                  run=run_serial,
                                  odor=cond["odor"] or None,
                                  sound=(cond["sound"] or {}).get("label")))

                # One trajectory plot per robot (each has its own log file).
                for r in range(len(starts)):
                    log_path = os.path.join(run_dir, f"trajectory_log_robot{r}.csv")
                    if not os.path.exists(log_path):
                        continue
                    try:
                        generate_trajectory_plot(
                            controller=controller,
                            trajectory_log_path=log_path,
                            save_path=os.path.join(run_dir, f"trajectory_robot{r}.png"),
                            odor_src=sourcePos[0],
                            odor_items=cond["target_items"],
                            scene_bounds=scene_bounds)
                    except Exception as e:
                        print(f"    Trajectory plot for robot {r} failed: {e}")

                plt.close('all')
                gc.collect()
                print(f"--- COMPLETED RUN {run_serial}/{len(start_pairs)} ---")

        try:
            generate_batch_summary(
                base_root="save", alg_folder=alg_folder,
                param_list=ENTROPY_FRACS, run_count=len(start_pairs),
                target_items=cond["target_items"], alg_choice='F')
        except Exception as e:
            print(f"Batch summary failed for {alg_folder}: {e}")

    finally:
        controller.stop()


def planned_grid():
    """Every (condition, modality) cell that will actually run, with reasons."""
    plan, skipped = [], []
    for cond in CONDITIONS:
        sets = (CONDITION_MODALITIES or {}).get(cond["name"], MODALITY_SETS) \
            if CONDITION_MODALITIES is not None else MODALITY_SETS
        for modalities in sets:
            # Drop only cells the scene cannot exercise: a sensor with no
            # stimulus. Redundant-looking cells are kept -- they are the ablation.
            if 'O' in modalities and not cond["odor"]:
                skipped.append((cond["name"], modalities, "scene has no odor"))
                continue
            if 'A' in modalities and cond["sound"] is None:
                skipped.append((cond["name"], modalities, "scene has no sound"))
                continue
            plan.append((cond, modalities))
    return plan, skipped


def main():
    plan, skipped = planned_grid()
    print(f"\n{'=' * 70}\nRUN PLAN  ({len(plan)} cells)\n{'=' * 70}")
    print(f"{'condition':>16} {'scene':>14} {'sensors':>8}   scene emits")
    for cond, m in plan:
        emits = "+".join(filter(None, ["odor" if cond["odor"] else "",
                                       "sound" if cond["sound"] else ""]))
        print(f"{cond['name']:>16} {cond['scene']:>14} {m:>8}   {emits}")
    for name, m, why in skipped:
        print(f"{name:>16} {'-':>14} {m:>8}   SKIPPED: {why}")
    print("=" * 70)

    for cond, modalities in plan:
        run_condition(cond, modalities)

    print("\nAll conditions complete.")
    try:
        generate_summary(base_path="save")
    except Exception as e:
        print(f"Overall summary failed: {e}")


if __name__ == "__main__":
    main()