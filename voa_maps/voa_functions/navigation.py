"""
navigation.py  --  information-driven action selection.

Replaces "path to argmax and step" with "step where the next measurement is
expected to teach you the most". The objective is the expected KL divergence
between the posterior you would hold after measuring and the belief you hold
now, which is identical to expected entropy reduction and to the mutual
information between the source location and the measurement:

    EIG(a) = H[p(s)] - E_z[ H[p(s|z,a)] ]
           = E_z[ KL( p(s|z,a) || p(s) ) ]
           = I(S; Z | a)

Each modality contributes a term, and two of the three have closed forms.

  AUDITORY   A bearing measurement that resolves the DOA bin partitions the
             grid by bearing, so the expectation collapses exactly to the
             entropy of the bearing marginal seen from the candidate pose:
                 EIG_A(a) = H[ p(theta | a) ]
             No sampling. This is zero when the whole belief lies along one
             bearing from a -- correctly telling the robot that standing on
             its own hypothesis line learns nothing, and that moving broadside
             is what triangulates.

  VISUAL     A detection resolves object identity in the cells that enter the
             field of view, so the gain is the belief mass brought into view
             weighted by how semantically uncertain those cells still are:
                 EIG_V(a) = sum_{c in FOV(a)} p(c) * Hdir(c) / log2(K)

  OLFACTORY  The measurement is a continuous concentration, so this one needs
             Monte Carlo: sample z from the predictive p(z|a), form the
             posterior each sample implies, and average the KL. A handful of
             samples is enough because the plume mean field mu(a, .) is
             computed once per candidate pose and reused across samples.

Costed objective, since pure infotaxis ignores travel:

    score(a) = sum_m w_m * EIG_m(a) - lambda_cost * dist(robot, a)

MULTI-ROBOT
-----------
Mutual information is submodular, so sequential greedy assignment is within
(1 - 1/e) of optimal: robot 0 picks, its choice is marked as claimed, robot 1
then picks against the reduced residual. Claiming is cheap here -- bearing bins
for the auditory term, FOV cells for the visual term -- which is what stops
both robots from taking the same viewpoint.
"""

import numpy as np

# ============================================================= CONFIG
N_BEARING_BINS = 24       # 15 deg resolution for the auditory marginal
N_Z_SAMPLES = 6           # Monte-Carlo samples for the olfactory term
FOV_DEG = 90.0            # camera horizontal field of view
FOV_RANGE = 4.0           # metres the camera is trusted to
LAMBDA_COST = 0.15        # bits sacrificed per metre travelled
EPS = 1e-12


def plume_field(px, pz, X, Z, q_s=2000.0, D=10.0, U=0.0, tau=1000.0, psi_deg=0.0):
    """Vectorised gaussian_plume: reading at sensor (px, pz) for EVERY hypothesised
    source cell (X, Z) at once.

    Numerically identical to olfactionFunctions.gaussian_plume called in a
    loop, but returns the whole (N,) field in one shot. eig_olfactory needs this
    field once per candidate pose, so the loop version would dominate planning
    cost on larger grids.

    Note the direction: the sensor is fixed and the SOURCE varies, which is the
    transpose of how the plume is usually evaluated. It matters only through the
    sign of the advection term, which is why psi is applied to (sensor - source)
    exactly as in the scalar version.
    """
    lambd = np.sqrt((D * tau) / (1.0 + (tau * U ** 2) / (4.0 * D)))
    psi = np.deg2rad(psi_deg)
    dx = px - np.asarray(X, float)
    dz = pz - np.asarray(Z, float)
    r = np.hypot(dx, dz)
    r = np.where(r == 0.0, 1e-3, r)
    del_z = -(dx * np.cos(psi) + dz * np.sin(psi))
    return (q_s / (4.0 * np.pi * D * r)) * np.exp((-del_z * U) / (2.0 * D) - (r / lambd))


def cell_centers(x_points, z_points):
    """Flattened (X, Z) world coordinates of every grid cell, row-major.

    Matches the [row, col] = [z, x] convention of the belief maps, so
    ravel()/reshape() round-trips align with p.
    """
    X, Z = np.meshgrid(np.asarray(x_points, float), np.asarray(z_points, float))
    return X.ravel(), Z.ravel()


def _entropy_bits(q):
    q = q[q > EPS]
    return float(-(q * np.log2(q)).sum())


# ============================================================= AUDITORY

def eig_auditory(p_flat, X, Z, rx, rz, n_bins=N_BEARING_BINS, claimed_bins=None):
    """Closed-form auditory information gain: entropy of the bearing marginal.

    Parameters
    ----------
    p_flat : np.ndarray
        Flattened belief over cells, sums to 1.
    claimed_bins : set, optional
        Bearing bins another robot already committed to observing this step.
        Their mass is removed before taking the entropy, so a second robot is
        not rewarded for re-measuring a bearing that is already covered.

    Returns
    -------
    float
        Expected information gain in bits.
    """
    bearing = np.degrees(np.arctan2(X - rx, Z - rz)) % 360.0
    idx = np.minimum((bearing / (360.0 / n_bins)).astype(int), n_bins - 1)
    marg = np.bincount(idx, weights=p_flat, minlength=n_bins)
    if claimed_bins:
        marg = marg.copy()
        for b in claimed_bins:
            marg[b] = 0.0
        s = marg.sum()
        if s <= EPS:
            return 0.0
        marg = marg / s
    return _entropy_bits(marg)


def auditory_bins_at(X, Z, rx, rz, p_flat, n_bins=N_BEARING_BINS, top_frac=0.9):
    """Bearing bins holding the bulk of the belief as seen from (rx, rz).

    Used to mark what a robot's measurement will cover so the next robot in the
    sequential-greedy pass does not duplicate it.
    """
    bearing = np.degrees(np.arctan2(X - rx, Z - rz)) % 360.0
    idx = np.minimum((bearing / (360.0 / n_bins)).astype(int), n_bins - 1)
    marg = np.bincount(idx, weights=p_flat, minlength=n_bins)
    order = np.argsort(-marg)
    keep, run = set(), 0.0
    for b in order:
        keep.add(int(b))
        run += marg[b]
        if run >= top_frac * marg.sum():
            break
    return keep


# ============================================================= VISUAL

def fov_mask(X, Z, rx, rz, yaw_deg, fov_deg=FOV_DEG, max_range=FOV_RANGE):
    """Boolean mask of cells inside the camera frustum at this pose."""
    dx, dz = X - rx, Z - rz
    rng = np.hypot(dx, dz)
    rel = (np.degrees(np.arctan2(dx, dz)) - yaw_deg + 180.0) % 360.0 - 180.0
    return (np.abs(rel) <= fov_deg / 2.0) & (rng <= max_range)


def eig_visual(p_flat, cell_entropy_flat, X, Z, rx, rz, yaw_deg,
               h_max=1.0, fov_deg=FOV_DEG, max_range=FOV_RANGE,
               claimed_mask=None):
    """Belief mass entering the FOV, weighted by its remaining class uncertainty.

    cell_entropy_flat is the per-cell Dirichlet class entropy; h_max normalises
    it to [0, 1] (pass log2(K)). A cell already resolved contributes nothing
    even if it holds belief mass, which is what stops the robot re-staring at
    an object it has already identified.
    """
    m = fov_mask(X, Z, rx, rz, yaw_deg, fov_deg, max_range)
    if claimed_mask is not None:
        m = m & ~claimed_mask
    if not m.any():
        return 0.0, m
    return float(np.sum(p_flat[m] * (cell_entropy_flat[m] / max(h_max, EPS)))), m


# ============================================================= OLFACTORY

def eig_olfactory(p_flat, mu, sigma, n_samples=N_Z_SAMPLES, rng=None):
    """Monte-Carlo expected KL for a continuous concentration measurement.

    Parameters
    ----------
    mu : np.ndarray
        Predicted mean reading at the candidate pose for every hypothesised
        source cell, i.e. mu[s] = plume(pose, s). Computed once per pose.
    sigma : float
        Sensor noise standard deviation.

    Returns
    -------
    float
        Expected KL in bits.
    """
    rng = rng or np.random.default_rng()
    # Sample z from the predictive p(z|a) = sum_s p(s) N(z; mu_s, sigma^2)
    s_idx = rng.choice(len(p_flat), size=n_samples, p=p_flat)
    z = mu[s_idx] + rng.normal(0.0, sigma, size=n_samples)

    # Likelihood matrix (n_samples, N); the constant factor cancels on
    # normalisation, so only the exponent is needed.
    d = (z[:, None] - mu[None, :]) / sigma
    logL = -0.5 * d * d
    logw = np.log(p_flat[None, :] + EPS) + logL
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    q = w / w.sum(axis=1, keepdims=True)

    kl = np.sum(q * (np.log2(q + EPS) - np.log2(p_flat[None, :] + EPS)), axis=1)
    return float(np.mean(kl))


# ============================================================= SELECTION

def select_pose(candidates, p_flat, X, Z, robot_xz,
                plume_fn=None, plume_field_fn=None, sigma=1.0,
                cell_entropy_flat=None, h_max=1.0,
                use_V=True, use_O=True, use_A=True,
                w_v=1.0, w_o=1.0, w_a=1.0,
                lambda_cost=LAMBDA_COST,
                claimed_bins=None, claimed_mask=None,
                yaw_options=(0.0, 90.0, 180.0, 270.0),
                rng=None):
    """Pick the candidate pose with the best information-per-cost score.

    Position is scored on the pose-independent terms (olfactory and auditory
    are both yaw-invariant -- a mic array resolves DOA over the full circle and
    the chemical sensor is a point sample), then yaw is chosen to maximise the
    visual term at the winning position. Decoupling this way avoids scoring
    |candidates| x |yaws| poses for no benefit.

    Parameters
    ----------
    candidates : list[tuple[float, float]]
        Reachable (x, z) positions to consider.
    plume_field_fn : callable
        plume_field_fn(px, pz) -> (N,) expected readings for every hypothesised
        source cell. Preferred when use_O; ~22x faster than the scalar form.
    plume_fn : callable
        Scalar fallback, plume_fn(px, pz, (sx, sz)) -> expected reading.

    Returns
    -------
    dict
        best pose, its yaw, the per-modality gains, and the claim sets to pass
        to the next robot in the sequential-greedy pass.
    """
    rng = rng or np.random.default_rng()
    p_flat = np.asarray(p_flat, float).ravel()
    p_flat = p_flat / max(p_flat.sum(), EPS)

    best = None
    for (cx, cz) in candidates:
        gains = {}
        if use_A:
            gains['A'] = eig_auditory(p_flat, X, Z, cx, cz, claimed_bins=claimed_bins)
        if use_O and (plume_field_fn is not None or plume_fn is not None):
            if plume_field_fn is not None:
                mu = np.asarray(plume_field_fn(cx, cz), float)
            else:
                mu = np.array([plume_fn(cx, cz, (X[i], Z[i])) for i in range(len(X))])
            gains['O'] = eig_olfactory(p_flat, mu, sigma, rng=rng)

        base = w_a * gains.get('A', 0.0) + w_o * gains.get('O', 0.0)

        best_yaw, best_v, best_m = 0.0, 0.0, None
        if use_V and cell_entropy_flat is not None:
            for yaw in yaw_options:
                v, m = eig_visual(p_flat, cell_entropy_flat, X, Z, cx, cz, yaw,
                                  h_max=h_max, claimed_mask=claimed_mask)
                if v > best_v or best_m is None:
                    best_yaw, best_v, best_m = yaw, v, m
        gains['V'] = best_v

        dist = float(np.hypot(cx - robot_xz[0], cz - robot_xz[1]))
        score = base + w_v * best_v - lambda_cost * dist

        if best is None or score > best['score']:
            best = dict(x=cx, z=cz, yaw=best_yaw, score=score, gains=gains,
                        dist=dist, fov=best_m)

    if best is None:
        return None

    best['claim_bins'] = auditory_bins_at(X, Z, best['x'], best['z'], p_flat) if use_A else set()
    best['claim_mask'] = best['fov'] if (use_V and best['fov'] is not None) else None
    return best


def plan_team(robots_xz, candidates_per_robot, p_map, X, Z, **kw):
    """Sequential-greedy assignment over the team.

    Mutual information is submodular, so assigning robots one at a time and
    marking each choice as claimed is within (1 - 1/e) of the optimal joint
    assignment, at a fraction of the cost of searching the joint action space.

    Parameters
    ----------
    X, Z : np.ndarray
        Flattened cell-centre coordinates from cell_centers(). Precomputed by
        the caller because they are constant for the whole run.

    Returns
    -------
    list[dict]
        One selection per robot, in order.
    """
    p_flat = np.asarray(p_map, float).ravel()
    claimed_bins, claimed_mask = set(), None
    out = []
    for rxz, cands in zip(robots_xz, candidates_per_robot):
        if not cands:
            out.append(None)
            continue
        sel = select_pose(cands, p_flat, X, Z, rxz,
                          claimed_bins=claimed_bins, claimed_mask=claimed_mask, **kw)
        if sel is None:
            out.append(None)
            continue
        claimed_bins = claimed_bins | sel['claim_bins']
        if sel['claim_mask'] is not None:
            claimed_mask = sel['claim_mask'] if claimed_mask is None else \
                (claimed_mask | sel['claim_mask'])
        out.append(sel)
    return out