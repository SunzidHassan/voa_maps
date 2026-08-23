#!/usr/bin/env python3
"""
inspect_maps.py  --  read and plot the belief maps a run saves as .npz.

    python3 inspect_maps.py voa_run_2026-08-10_155537              # summary
    python3 inspect_maps.py voa_run_2026-08-10_155537 --plot        # PNG per step
    python3 inspect_maps.py voa_run_2026-08-10_155537 --step 5      # one step
    python3 inspect_maps.py voa_run_2026-08-10_155537 --dump 5      # raw values

Each maps_NNN.npz holds:

    fused, vision, olfaction, sound   (H, W) belief maps, each summing to 1
    free                              (H, W) bool, True where the cell is
                                      navigable -- belief is zero elsewhere
    x_points, z_points                1-D world coordinates of the grid axes;
                                      x_points indexes COLUMNS, z_points ROWS

The row/column order is the thing to get right: maps are indexed [row, col] =
[z, x], so map[i, j] is the cell at world (x_points[j], z_points[i]). Reading
it the other way transposes the room and every conclusion drawn from it.
"""

import os
import sys
import glob
import json
import argparse

import numpy as np


def load_run(run_dir):
    """Every maps_NNN.npz in a run directory, in step order."""
    paths = sorted(glob.glob(os.path.join(run_dir, 'VO/maps_*.npz')))
    if not paths:
        raise FileNotFoundError(
            f"no maps_*.npz under {run_dir}\n"
            f"A run only writes these once it reaches the FUSE state, so an\n"
            f"empty directory usually means it never got past waiting for /map\n"
            f"or waiting for /olfaction.")
    return paths


def entropy_bits(p):
    q = np.asarray(p, float).ravel()
    q = q[q > 1e-12]
    return float(-(q * np.log2(q)).sum())


def peak_of(m, x_points, z_points):
    """(x, z, probability) of the highest cell."""
    gi = np.unravel_index(int(np.argmax(m)), m.shape)
    return float(x_points[gi[1]]), float(z_points[gi[0]]), float(m[gi])


def summarise(run_dir):
    paths = load_run(run_dir)
    d0 = np.load(paths[0])
    xp, zp, free = d0['x_points'], d0['z_points'], d0['free']
    n_free = int(free.sum())

    print("=" * 74)
    print(f"{run_dir}   {len(paths)} steps")
    print(f"grid {free.shape[0]} x {free.shape[1]} = {free.size} cells, "
          f"{n_free} free ({100*n_free/free.size:.0f}%)")
    print(f"x {xp.min():.2f}..{xp.max():.2f} m    z {zp.min():.2f}..{zp.max():.2f} m"
          f"    cell {abs(xp[1]-xp[0]) if len(xp) > 1 else 0:.2f} m")
    print(f"H_max = {np.log2(free.size):.2f} bits over all cells, "
          f"{np.log2(max(n_free,2)):.2f} over free cells")

    meta_path = os.path.join(run_dir, 'run_meta.json')
    if os.path.exists(meta_path):
        m = json.load(open(meta_path))
        print(f"\n{m.get('modalities','?')} on {m.get('platform','?')}, "
              f"{m.get('steps','?')} steps, entropy_frac={m.get('entropy_frac','?')}")
        est = m.get('estimated_source', {})
        print(f"estimated source ({est.get('x',float('nan')):.2f}, "
              f"{est.get('y',float('nan')):.2f})  peak object: {m.get('peak_object','?')}")
        if m.get('q_s_pegged'):
            print("  !! q_s pegged at a grid edge -- true rate likely outside "
                  "Q_S_HYPOTHESES")

    keys = [k for k in ('vision', 'olfaction', 'sound', 'fused') if k in d0]
    Hmax = np.log2(free.size)
    print(f"\n{'step':>4} " + " ".join(f"{k[:5]:>7}" for k in keys) +
          f" {'fused peak (x, z)':>20} {'p':>7}")
    print(f"{'':>4} " + " ".join(f"{'H/Hmax':>7}" for _ in keys))
    print("-" * 74)
    for i, p in enumerate(paths):
        d = np.load(p)
        hs = [entropy_bits(d[k]) / Hmax for k in keys]
        px, pz, pv = peak_of(d['fused'], d['x_points'], d['z_points'])
        print(f"{i:>4} " + " ".join(f"{h:>7.3f}" for h in hs) +
              f"   ({px:>6.2f}, {pz:>6.2f}) {pv:>7.4f}")
    print("=" * 74)
    print("H/Hmax near 1.000 = flat/uninformative; near 0 = concentrated.")
    print("A branch that is not part of this run stays at exactly 1.000.")


def dump(run_dir, step):
    p = os.path.join(run_dir, f'maps_{step:03d}.npz')
    d = np.load(p)
    print(f"{p}\n")
    for k in d.files:
        a = d[k]
        print(f"{k:>12}  shape {str(a.shape):>12}  dtype {str(a.dtype):>8}", end='')
        if a.dtype == bool:
            print(f"  True in {int(a.sum())}/{a.size}")
        elif a.ndim == 2:
            print(f"  sum {a.sum():.6f}  min {a.min():.3e}  max {a.max():.3e}")
        else:
            print(f"  {a.min():.2f}..{a.max():.2f}")

    xp, zp = d['x_points'], d['z_points']
    print("\ntop 5 fused cells:")
    f = d['fused']
    flat = np.argsort(f.ravel())[::-1][:5]
    for r in flat:
        i, j = np.unravel_index(r, f.shape)
        print(f"   ({xp[j]:>6.2f}, {zp[i]:>6.2f})  p={f[i,j]:.5f}   [row {i}, col {j}]")


def plot(run_dir, only_step=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    paths = load_run(run_dir)
    out_dir = os.path.join(run_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    for p in paths:
        step = int(os.path.basename(p).split('_')[1].split('.')[0])
        if only_step is not None and step != only_step:
            continue
        d = np.load(p)
        xp, zp, free = d['x_points'], d['z_points'], d['free']
        extent = [xp.min(), xp.max(), zp.min(), zp.max()]
        Hmax = np.log2(free.size)

        # Only show branches that carry information; a modality not used in
        # this run holds a uniform map and renders as a flat panel that reads
        # as a failed sensor rather than a disabled one.
        keys = [k for k in ('olfaction', 'vision', 'sound', 'fused')
                if k in d and (d[k].max() - d[k].min()) > 1e-12]
        if not keys:
            continue

        fig, axes = plt.subplots(1, len(keys), figsize=(5*len(keys), 4.4),
                                 squeeze=False)
        axes = axes[0]
        for ax, k in zip(axes, keys):
            a = d[k].astype(float)
            shown = np.where(free, a, np.nan)     # blank out non-navigable cells
            im = ax.imshow(shown, origin='lower', extent=extent, cmap='hot',
                           aspect='equal')
            ax.set_title(f"{k}  H/Hmax={entropy_bits(a)/Hmax:.3f}")
            ax.set_xlabel('x (m)')
            ax.set_ylabel('z (m)')
            px, pz, pv = peak_of(a, xp, zp)
            ax.plot(px, pz, marker='o', ms=13, mfc='none', mec='#39ff14', mew=2.0)
            ax.annotate(f"({px:.2f}, {pz:.2f})\np={pv:.4f}", xy=(px, pz),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=7, color='#39ff14', fontweight='bold')
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{os.path.basename(run_dir)}  step {step}")
        fig.tight_layout()
        f_out = os.path.join(out_dir, f"maps_{step:03d}.png")
        fig.savefig(f_out, dpi=120)
        plt.close(fig)
        print(f"wrote {f_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--plot', action='store_true', help='write a PNG per step')
    ap.add_argument('--step', type=int, default=None, help='limit --plot to one step')
    ap.add_argument('--dump', type=int, default=None, help='raw values for one step')
    a = ap.parse_args()

    if a.dump is not None:
        dump(a.run_dir, a.dump)
    elif a.plot:
        plot(a.run_dir, a.step)
    else:
        summarise(a.run_dir)


if __name__ == '__main__':
    main()