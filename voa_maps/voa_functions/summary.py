"""
summary.py  --  flat summary across every VOA run.

Replaces generate_flat_summary, which could not read these runs at all. That
function decided what a folder was by matching hardcoded substrings:

    METHOD_MAP = {'f': ..., 'o': ..., 'r': ...}          # no 'VA' / 'VAO' / 'VO'
    OBJECT_MAP = {'stoveburner': ..., 'garbagecan': ...} # no 'odor_and_sound'
    thresholds = ['0.2', '0.5', '0.8', '0.9']            # no 0.4

`save_VA_odor_and_sound` matched nothing in OBJECT_MAP, so every folder was
skipped and the row list came back empty.

This version parses nothing. Each run now writes run_meta.json describing
itself -- sensors, condition, threshold, target, whether it terminated -- so
renaming a folder or adding a condition cannot break the summary. It also
reports the multi-robot quantities the old path had no concept of: team
distance, per-robot distance, and listening dwell time.
"""

import os
import json
import numpy as np
import pandas as pd


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _walk_runs(base_path):
    """Yield every directory containing a run, at any depth under base_path.

    A run is identified by the files it contains, not by where it sits, so the
    save/<sensors_condition>/<threshold>/<run>/ layout is a convention rather
    than a requirement.
    """
    for root, _dirs, files in os.walk(base_path):
        if 'run_meta.json' in files or 'team_summary.csv' in files \
                or 'trajectory_log.csv' in files:
            yield root


def _target_matches(predicted, target_items):
    """Compare a logged target to the ground-truth item, ignoring instance IDs."""
    if predicted is None or (isinstance(predicted, float) and np.isnan(predicted)):
        return None
    if not target_items:
        return None
    tgt = target_items[0] if isinstance(target_items, (list, tuple)) else target_items
    return str(predicted).split('_')[0] == str(tgt).split('_')[0]


def collect_runs(base_path="save"):
    """One row per run, assembled from run_meta.json + team_summary + team_log."""
    rows = []
    for run_dir in _walk_runs(base_path):
        meta = _read_json(os.path.join(run_dir, 'run_meta.json'))
        summary = _read_csv(os.path.join(run_dir, 'team_summary.csv'))
        team_log = _read_csv(os.path.join(run_dir, 'team_log.csv'))

        row = {
            'run_dir': os.path.relpath(run_dir, base_path),
            'condition': meta.get('condition'),
            'scene': meta.get('scene'),
            'sensors': meta.get('modalities'),
            'nav_mode': meta.get('nav_mode'),
            'trigger_mode': meta.get('trigger_mode'),
            'entropy_frac': meta.get('entropy_frac'),
            'run': meta.get('run'),
            'team_size': meta.get('team_size'),
            'steps': meta.get('steps'),
            'terminated': meta.get('terminated'),
        }

        # Fall back to folder position when a run predates run_meta.json.
        if row['sensors'] is None:
            parts = os.path.relpath(run_dir, base_path).split(os.sep)
            if parts and parts[0].startswith('save_'):
                tok = parts[0][len('save_'):].split('_')
                if tok and set(tok[0].upper()) <= set('VAO'):
                    row['sensors'] = tok[0].upper()
                    row['condition'] = '_'.join(tok[1:]) or None
            if len(parts) >= 3:
                try:
                    row['entropy_frac'] = float(parts[1])
                except ValueError:
                    pass
                row['run'] = parts[2]

        if summary is not None and not summary.empty:
            s = summary.iloc[0]
            row.update({
                'team_distance': s.get('team_distance'),
                'per_robot_distance': s.get('per_robot_distance'),
                'total_listen_s': s.get('total_listen_s'),
                'final_target_error': s.get('final_target_error'),
                'final_min_gt_distance': s.get('final_min_gt_distance'),
                'final_target_object': s.get('final_target_object'),
                'final_dist_to_target': s.get('dist_to_target'),
            })
            if row['steps'] is None:
                row['steps'] = s.get('steps')
            if row['team_size'] is None:
                row['team_size'] = s.get('team_size')

        if team_log is not None and not team_log.empty:
            last = team_log.iloc[-1]
            row.update({
                'final_fused_entropy': last.get('fused_entropy'),
                'final_trigger_stat': last.get('trigger_stat'),
                'final_trigger_V': last.get('trigger_V'),
                'final_trigger_O': last.get('trigger_O'),
                'final_trigger_A': last.get('trigger_A'),
                'total_step_time': team_log['step_time'].sum()
                if 'step_time' in team_log else np.nan,
                'sound_ever_detected': bool(team_log['any_sound_detected'].any())
                if 'any_sound_detected' in team_log else None,
            })

        row['correct_target'] = _target_matches(row.get('final_target_object'),
                                                meta.get('target_items'))
        rows.append(row)
    return rows


def generate_summary(base_path="save/simulation", out_name="flat_summary.csv"):
    """Write one flat CSV over every run, plus a grouped console report."""
    print("=" * 60)
    print(f"VOA summary from: {base_path}")
    print("=" * 60)

    if not os.path.isdir(base_path):
        print(f"'{base_path}' does not exist. Nothing to summarise.")
        return None

    rows = collect_runs(base_path)
    if not rows:
        print(f"No runs found under '{base_path}'.")
        print("Looked for directories containing run_meta.json, team_summary.csv,")
        print("or trajectory_log.csv. If runs exist, check they completed and")
        print("wrote their logs -- a crash before _flush_logs leaves none of these.")
        return None

    df = pd.DataFrame(rows).sort_values(
        ['condition', 'sensors', 'entropy_frac', 'run'], na_position='last')
    out_path = os.path.join(base_path, out_name)
    df.to_csv(out_path, index=False)
    print(f"{len(df)} runs -> {out_path}\n")

    have = [c for c in ['steps', 'team_distance', 'final_min_gt_distance',
                        'final_target_error', 'total_listen_s'] if c in df.columns]
    if have and df['sensors'].notna().any():
        agg = df.groupby(['condition', 'sensors'], dropna=False)[have].mean().round(2)
        if 'terminated' in df.columns:
            agg['terminated_%'] = (df.groupby(['condition', 'sensors'], dropna=False)
                                   ['terminated'].mean() * 100).round(0)
        if df['correct_target'].notna().any():
            agg['correct_%'] = (df.groupby(['condition', 'sensors'], dropna=False)
                                ['correct_target'].mean() * 100).round(0)
        print(agg.to_string())

    # A run that hit the step limit did not converge; its metrics are censored,
    # so flag it rather than letting it average in silently.
    # Success is decided HERE, not at runtime: the controller stops on entropy
    # alone and logs how far it ended from the target, so the radius is a
    # reporting choice and can be varied without re-running.
    for radius in (0.5, 1.0):
        col = f'success@{radius}m'
        if 'final_min_gt_distance' in df.columns:
            df[col] = df['final_min_gt_distance'] <= radius
    scols = [c for c in df.columns if c.startswith('success@')]
    if scols and df['sensors'].notna().any():
        print("\nSuccess rate by distance-to-true-source threshold:")
        print((df.groupby(['condition', 'sensors'], dropna=False)[scols].mean() * 100
               ).round(0).to_string())

    if 'terminated' in df.columns and (df['terminated'] == False).any():  # noqa: E712
        n = int((df['terminated'] == False).sum())  # noqa: E712
        print(f"\n{n} of {len(df)} runs hit the step limit without triggering.")
        print("Their steps/distance are censored at the budget, not converged values.")
        stuck = df[df['terminated'] == False]  # noqa: E712
        for (c, s), g in stuck.groupby(['condition', 'sensors'], dropna=False):
            print(f"   {c} + {s}: {len(g)} runs, "
                  f"final trigger {g['final_trigger_stat'].mean():.3f} "
                  f"vs threshold {g['entropy_frac'].iloc[0]}")
    return df


if __name__ == '__main__':
    import sys
    generate_summary(sys.argv[1] if len(sys.argv) > 1 else "save")