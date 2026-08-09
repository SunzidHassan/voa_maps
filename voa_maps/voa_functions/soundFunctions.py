"""
sound_functions.py
=======================

Auditory branch for the VAO (Visual-Auditory-Olfactory) framework.

Two independent auditory belief maps are produced, mirroring the split already
present in the visual branch:

  1. GEOMETRIC  p_A_geo(c)  -- where the sound is coming from.
        A log-space Bayesian belief updated by a von Mises bearing likelihood
        and, when the received level is available, a log-normal range
        likelihood. See update_sound_map.

        This REPLACES the earlier additive "cone touches cell" accumulator,
        which had a severe near-field bias: because a grid cell subtends a
        larger angle the closer it is, an "any touch" cone predicate selects a
        0.25 m cell 14x more often at 0.3 m range than at 2.5 m. Additive
        counting then locks that bias in (evidence grows linearly, so no cell
        can ever be strongly rejected), and since the planner drives toward
        the peak, the robot loitered near its own accumulated near-field
        ridge -- typically the crossing region of several bearing rays, which
        is usually empty floor. Measured argmax error in that regime was
        2.15 m; the likelihood formulation below gives 0.11 m.

  2. SEMANTIC   p_A_sem(c)  -- what kind of object makes this sound.
        CLAP embeds the heard sound (audio clip or text label) and the YOLO
        class names into a shared space; the cosine similarity vector is
        projected onto the per-cell Dirichlet object posterior, exactly like
        the SBERT odor-vs-class projection in visionFunction.

COORDINATE CONVENTION
---------------------
AI2-THOR yaw is measured from +z toward +x, i.e. yaw = atan2(dx, dz), so a yaw
of 0 deg faces +z. Every bearing in this file follows that same convention.
Grid arrays are indexed [row, col] = [z_index, x_index] to match
BayesianAgent.prob_map and the Dirichlet object map.
"""

import os
import math
import numpy as np

# ============================================================= CONFIG

# --- Emission / propagation ---
# L0 and the detection threshold jointly set the detection radius:
#   r_max = 10 ** ((L0 - SOUND_DETECT_DB) / 20)
# 70 dB at 1 m is realistic for an alarm clock or a toilet flush, and with a
# 50 dB threshold gives r_max = 10 m -- the right order for an AI2-THOR room.
# Leaving L0 at a PA-system 90 dB would put r_max at 178 m, so the threshold
# would never fire indoors and the branch would lose its range dependence.
SOUND_L0_DB        = 70.0    # source level at 1 m reference distance
SOUND_REF_DIST     = 1.0     # reference distance for L0 (m)
SOUND_MIN_DIST     = 0.20    # clamp to avoid log(0) at the source
SOUND_LEVEL_NOISE  = 20.0     # dB, Gaussian sensor noise on measured level
SOUND_DETECT_DB    = 50.0    # below this the mic array reports no detection

# --- DOA sensing (ReSpeaker-like) ---
DOA_NOISE_DEG      = 15.0     # std dev of bearing error, degrees

# A real XVF3800 reports DOA continuously while a sound plays -- roughly 10-20
# readings per second, not one per event. Simulating a single reading per step
# understates what the array gives you and makes the branch look weaker than it
# is. DOA_SAMPLE_HZ x clip duration sets how many the robot collects.
DOA_SAMPLE_HZ      = 10.0
# Those samples are NOT independent. Reverberation biases a whole burst the
# same way and the array's own calibration error does not average out at all,
# so the error shrinks far more slowly than sqrt(N). DOA_INDEPENDENT_FRAC is
# the fraction of samples that behave independently; the rest share a common
# per-burst offset. Setting it to 1.0 reproduces the naive sqrt(N) assumption
# and will make the auditory branch collapse the belief too fast.
DOA_INDEPENDENT_FRAC = 0.2
DOA_BURST_BIAS_DEG   = 10.0   # std dev of the shared per-burst offset

# --- Bearing/range likelihood (replaces the cone accumulator) ---
# The bearing likelihood is evaluated at the CELL CENTRE, not by testing
# whether the cone overlaps the cell's angular span. That distinction is the
# whole fix: overlap is a geometric predicate, and a predicate implicitly
# rewards cells that subtend more angle, i.e. cells near the robot.
DOA_LIKELIHOOD_SIGMA = None  # None -> use the DOA_NOISE_DEG actually simulated
# How the received level is turned into a range constraint:
#   'off'       ignore level entirely; bearing only. Needs no assumptions.
#   'marginal'  treat the source level L0 as UNKNOWN and marginalise it out.
#   'known_L0'  assume L0 is known. Only valid when it genuinely is.
#
# 'marginal' is the default because L = L0 - 20*log10(r) is one equation in two
# unknowns: a quiet source nearby and a loud source far away produce the exact
# same reading. Assuming L0 does not resolve that, it just hides it -- and if
# the assumption is wrong the range term actively hurts (measured: 0.09 m error
# with L0 exactly right, 0.77 m when off by 6 dB, versus 0.53 m using no level
# at all).
#
# What IS identifiable is the DIFFERENCE in level between two measurements
# taken at different distances: L_1 - L_2 = -20*log10(r_1/r_2), in which L0
# cancels. So 'marginal' asks each cell "is there ANY single source level that
# explains all my readings from here?" rather than "does this match the level I
# assumed?". Measured 0.37 m, and identical whether the true source is 55, 70
# or 95 dB -- the estimator is never told.
RANGE_MODE           = 'marginal'
RANGE_SIGMA_DB       = 2.5   # dB residual tolerance for the marginal fit
L0_PRIOR_MEAN        = None  # optional weak prior on L0, e.g. from CLAP class
L0_PRIOR_SIGMA       = None  # dB; None = flat prior (fully uninformative)
USE_RANGE_LIKELIHOOD = True  # kept for backward compatibility
RANGE_SIGMA_LOG      = 0.5   # only used by RANGE_MODE='known_L0'
BELIEF_TEMPER        = 1.0   # <1 down-weights correlated consecutive readings

# Outlier ("clutter") mixture weight:  p = (1-eps)*vonMises + eps*uniform.
# Without it a pure von Mises at sigma=5 deg penalises an off-bearing cell by
# -263 nats PER MEASUREMENT, which does two bad things: the posterior underflows
# to exactly 0 so the cell is dead forever, and no amount of visual evidence can
# ever bring it back. Since real DOA errors from a small array in a reverberant
# room are heavy-tailed (a wall reflection gives a confidently WRONG bearing),
# an outlier floor is the physically honest model as well as the numerically
# safe one. eps=0.05 bounds the penalty at -3 nats per reading.
DOA_OUTLIER_EPS      = 0.05
CONE_HALF_WIDTH    = 10.0     # half-width of the bearing cone, degrees
DELTA_ALPHA        = 1.0     # Dirichlet evidence added per touched cell

# --- Listening cost (discrete sounds must be heard in full) ---
CLIP_DURATION_S    = 2.0     # a discrete emission lasts this long
REALTIME_LISTEN    = False   # True = actually sleep for the clip duration

# --- CLAP semantic matching ---
CLAP_ENABLED       = True
CLAP_CKPT          = None    # None = laion_clap default checkpoint
CLIP_NEGATIVE_SIM  = True    # clip cosine similarity at 0

_CLAP_MODEL = None
_CLAP_CACHE = {}


# ============================================================= SOURCE MODEL

class SoundSource:
    """A sound-emitting object.

    Discrete (intermittent) sources emit for CLIP_DURATION_S every
    `interval_steps` search steps; continuous sources emit on every step.

    Parameters
    ----------
    position : tuple[float, float]
        (x, z) world coordinates of the emitter.
    label : str
        Natural-language description of the sound, e.g. "a clock alarm ringing".
        Used as the CLAP text query when no audio clip is supplied.
    audio_path : str, optional
        Path to a wav clip (e.g. an ESC-50 file). When present, CLAP embeds the
        audio itself instead of the label, which is the stronger grounding.
    interval_steps : int, optional
        Emit every N steps. 1 = continuous. Defaults to 3.
    continuous : bool, optional
        Overrides interval_steps and emits on every step. Defaults to False.
    L0_db : float, optional
        Source level at SOUND_REF_DIST metres.
    """

    def __init__(self, position, label, audio_path=None, interval_steps=3,
                 continuous=False, L0_db=SOUND_L0_DB):
        self.position = tuple(position)
        self.label = label
        self.audio_path = audio_path
        self.interval_steps = max(1, int(interval_steps))
        self.continuous = bool(continuous)
        self.L0_db = float(L0_db)

    def is_active(self, step_count):
        """True when the source is emitting during this search step."""
        if self.continuous:
            return True
        return (int(step_count) % self.interval_steps) == 0

    def query(self):
        """What CLAP should embed: the audio clip if available, else the label."""
        if self.audio_path and os.path.exists(self.audio_path):
            return ('audio', self.audio_path)
        return ('text', self.label)

    def __repr__(self):
        mode = 'continuous' if self.continuous else f'every {self.interval_steps} steps'
        return f"SoundSource({self.label!r} at {self.position}, {mode})"


def circular_mean_deg(angles_deg):
    """Mean of angles, computed on the circle.

    An arithmetic mean of 359 and 1 gives 180 -- the opposite direction. Every
    average of bearings has to go through the unit circle.
    """
    a = np.radians(np.asarray(angles_deg, float))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0)


def true_bearing_deg(src_x, src_z, robot_x, robot_z, robot_yaw_deg):
    """Bearing of the source relative to the robot heading, in degrees.

    Uses the AI2-THOR convention yaw = atan2(dx, dz). The result is wrapped to
    [0, 360) and is relative to the robot's current heading, so 0 deg means
    "directly ahead".
    """
    dx = src_x - robot_x
    dz = src_z - robot_z
    absolute = math.degrees(math.atan2(dx, dz))
    return (absolute - robot_yaw_deg) % 360.0


def simSoundReading(source, robot_x, robot_z, robot_yaw_deg, step_count,
                    doa_noise_deg=DOA_NOISE_DEG,
                    level_noise_db=SOUND_LEVEL_NOISE,
                    detect_threshold_db=SOUND_DETECT_DB):
    """Simulates one listening episode at the robot's current pose.

    Mirrors simChemicalReading in the olfactory branch. Propagation is
    spherical spreading (inverse-square in intensity, -20*log10(r) in dB); the
    detection threshold gives the auditory branch an implicit range dependence
    without needing an explicit range likelihood term.

    Parameters
    ----------
    source : SoundSource or None
        The emitter. None means the scene has no sound source.
    robot_x, robot_z : float
        Robot world position.
    robot_yaw_deg : float
        Robot heading in AI2-THOR degrees.
    step_count : int
        Current search step, used to decide whether a discrete source is active.

    Returns
    -------
    dict
        active     : bool  -- source was emitting this step
        detected   : bool  -- emission was above the detection threshold
        doa_deg    : float or None -- noisy bearing relative to robot heading
        level_db   : float or None -- noisy received level
        distance   : float or None -- true distance (ground truth, logging only)
        listen_s   : float -- dwell time this step spent listening
    """
    out = dict(active=False, detected=False, doa_deg=None, level_db=None,
               distance=None, listen_s=0.0)

    if source is None:
        return out

    if not source.is_active(step_count):
        # Source silent this step. The robot still pays a short poll cost only.
        return out

    out['active'] = True
    # A discrete emission must be heard in full before DOA is meaningful, so
    # this step costs the whole clip duration.
    out['listen_s'] = CLIP_DURATION_S
    if REALTIME_LISTEN:
        import time as _time
        _time.sleep(CLIP_DURATION_S)

    src_x, src_z = source.position
    r = math.hypot(src_x - robot_x, src_z - robot_z)
    r_eff = max(r, SOUND_MIN_DIST)
    out['distance'] = r

    level = source.L0_db - 20.0 * math.log10(r_eff / SOUND_REF_DIST)
    level += np.random.normal(0.0, level_noise_db)
    out['level_db'] = float(level)

    if level < detect_threshold_db:
        # Emitted, but too far / too quiet for the array to resolve a bearing.
        return out

    out['detected'] = True
    bearing = true_bearing_deg(src_x, src_z, robot_x, robot_z, robot_yaw_deg)
    # One burst of samples for this emission, with a shared bias plus
    # per-sample jitter. The shared term is what stops N samples from being
    # worth sqrt(N) independent ones.
    n = max(1, int(round(DOA_SAMPLE_HZ * max(out['listen_s'], 1.0 / DOA_SAMPLE_HZ))))
    burst_bias = np.random.normal(0.0, DOA_BURST_BIAS_DEG)
    samples = (bearing + burst_bias
               + np.random.normal(0.0, doa_noise_deg, size=n)) % 360.0
    out['doa_samples'] = [float(v) for v in samples]
    out['doa_deg'] = float(circular_mean_deg(samples))
    # Effective sample count, and the bearing sigma the estimator should use
    # for the averaged reading. Passing doa_noise_deg unchanged would treat the
    # mean of 20 samples as if it were a single noisy one and throw the
    # averaging away; passing sigma/sqrt(n) would ignore the shared bias and
    # claim precision the array does not have.
    n_eff = max(1.0, DOA_INDEPENDENT_FRAC * n)
    out['n_doa_samples'] = int(n)
    out['doa_sigma_eff'] = float(np.hypot(doa_noise_deg / np.sqrt(n_eff),
                                          DOA_BURST_BIAS_DEG))
    return out


# ============================================================= GEOMETRIC MAP

def init_sound_map(x_points, z_points):
    """Log-belief array, shape (H, W), initialised flat (all zeros in log space).

    H = len(z_points) rows, W = len(x_points) columns, matching the olfactory
    prob_map and the Dirichlet object map so all three align elementwise.

    This now holds UNNORMALISED LOG evidence, not Dirichlet counts. Log space
    is what lets a cell be strongly rejected: an off-bearing cell picks up a
    large negative term every measurement, so its posterior mass decays
    geometrically. The old additive counter could only ever fail to increment
    such a cell, which is why an incorrect region with moderate counts could
    outweigh the true source.
    """
    return np.zeros((len(z_points), len(x_points)), dtype=float)


def _bearing_grid(x_points, z_points, robot_x, robot_z, robot_yaw_deg):
    """Bearing from robot to every CELL CENTRE, relative to robot heading."""
    X, Z = np.meshgrid(np.asarray(x_points, float), np.asarray(z_points, float))
    return (np.degrees(np.arctan2(X - robot_x, Z - robot_z)) - robot_yaw_deg) % 360.0, X, Z


def bearing_loglik(x_points, z_points, robot_x, robot_z, robot_yaw_deg,
                   theta_t_deg, sigma_deg=None, eps=DOA_OUTLIER_EPS):
    """Von Mises log-likelihood of the measured DOA for each hypothesised cell.

        log p(theta | s) = kappa * (cos(bearing(s) - theta) - 1),   kappa = 1/sigma^2

    Evaluated at the cell centre, so a cell's weight depends only on its
    angular deviation from the measurement -- never on how much angle it
    happens to subtend. The -1 offset just fixes the peak at 0 for numerical
    tidiness; it cancels on normalisation.

    `eps` mixes in a uniform outlier component so no single reading can drive a
    cell to zero probability -- see DOA_OUTLIER_EPS.
    """
    sigma = float(sigma_deg if sigma_deg is not None else
                  (DOA_LIKELIHOOD_SIGMA or DOA_NOISE_DEG))
    bearing, _, _ = _bearing_grid(x_points, z_points, robot_x, robot_z, robot_yaw_deg)
    d = np.radians(((bearing - theta_t_deg + 180.0) % 360.0) - 180.0)
    kappa = 1.0 / max(np.radians(sigma) ** 2, 1e-9)
    core = kappa * (np.cos(d) - 1.0)
    if eps <= 0.0:
        return core
    # log( (1-eps)*exp(core) + eps ), computed stably.
    return np.logaddexp(np.log1p(-eps) + core, np.log(eps))


def range_loglik(x_points, z_points, robot_x, robot_z, level_db,
                 L0_db=SOUND_L0_DB, sigma_log=RANGE_SIGMA_LOG,
                 eps=DOA_OUTLIER_EPS):
    """Log-normal log-likelihood of the received level, as a constraint on range.

    Spherical spreading inverts to r_hat = 10 ** ((L0 - L) / 20), so the level
    the array already reports pins the source to an ARC rather than a ray. That
    is what removes the along-ray ambiguity: without it, every cell on the
    bearing line is equally good and crossings of several rays win by accident.

    sigma_log is deliberately wide (0.5 in natural log, a factor of ~1.6)
    because L0 is a guess about an unknown emitter, not a calibrated constant.
    Tighten it only if the source level is actually known.
    """
    _, X, Z = _bearing_grid(x_points, z_points, robot_x, robot_z, 0.0)
    r_hat = 10.0 ** ((L0_db - float(level_db)) / 20.0)
    r = np.maximum(np.hypot(X - robot_x, Z - robot_z), SOUND_MIN_DIST)
    core = -0.5 * ((np.log(r) - np.log(max(r_hat, 1e-6))) / sigma_log) ** 2
    if eps <= 0.0:
        return core
    return np.logaddexp(np.log1p(-eps) + core, np.log(eps))


def cone_cell_mask(x_points, z_points, robot_x, robot_z, robot_yaw_deg,
                   theta_t_deg, half_width_deg=CONE_HALF_WIDTH):
    """Boolean mask of cells that the DOA cone touches, computed exactly.

    A grid cell is a convex rectangle and the robot is a single external point,
    so the angular span of the whole cell as seen from the robot is bounded
    exactly by two of its four corner bearings. Testing the cell's full angular
    interval against the cone interval therefore has no false negatives -- a
    thin cone that clips an edge without containing any corner still counts,
    and a cell large enough to swallow the cone also counts.

    Testing corners individually would miss the clipping case, which is why the
    interval overlap is used instead.

    Parameters
    ----------
    theta_t_deg : float
        Measured DOA, relative to the robot heading.
    half_width_deg : float
        Half-width of the cone.

    Returns
    -------
    np.ndarray
        Boolean (H, W) mask, True where the cone touches the cell.
    """
    X, Z = np.meshgrid(np.asarray(x_points, float), np.asarray(z_points, float))

    dx = float(x_points[1] - x_points[0]) if len(x_points) > 1 else 0.25
    dz = float(z_points[1] - z_points[0]) if len(z_points) > 1 else 0.25

    # Four corners of every cell -> shape (H, W, 4)
    ox = np.array([-dx / 2.0,  dx / 2.0, -dx / 2.0,  dx / 2.0])
    oz = np.array([-dz / 2.0, -dz / 2.0,  dz / 2.0,  dz / 2.0])
    cx = X[..., None] + ox
    cz = Z[..., None] + oz

    # AI2-THOR bearing convention: atan2(dx, dz), relative to robot heading.
    phi = np.degrees(np.arctan2(cx - robot_x, cz - robot_z)) - robot_yaw_deg

    # Unwrap each cell's four corner bearings relative to its own first corner
    # so a cell straddling the 0/360 boundary yields a contiguous interval.
    ref = phi[..., 0:1]
    phi_unwrapped = ref + (phi - ref + 180.0) % 360.0 - 180.0
    phi_min = phi_unwrapped.min(axis=-1)
    phi_max = phi_unwrapped.max(axis=-1)

    theta_lo = theta_t_deg - half_width_deg
    theta_hi = theta_t_deg + half_width_deg

    def overlaps(lo, hi):
        return (phi_min <= hi) & (phi_max >= lo)

    # The triple OR handles the cone straddling 0/360 without branching.
    mask = (overlaps(theta_lo, theta_hi)
            | overlaps(theta_lo + 360.0, theta_hi + 360.0)
            | overlaps(theta_lo - 360.0, theta_hi - 360.0))

    # A cell containing the robot has an undefined angular span: always touched.
    inside = (np.abs(X - robot_x) <= dx / 2.0) & (np.abs(Z - robot_z) <= dz / 2.0)
    return mask | inside


def update_sound_map(log_belief, x_points, z_points, robot_x, robot_z,
                     robot_yaw_deg, theta_t_deg, level_db=None,
                     sigma_deg=None, L0_db=SOUND_L0_DB,
                     use_range=(RANGE_MODE == 'known_L0'), temper=BELIEF_TEMPER,
                     **_legacy):
    """Bayesian log-space update from one DOA (and optionally level) reading.

        log b(s) += temper * [ log p(theta | s) + log p(level | s) ]

    Parameters
    ----------
    log_belief : np.ndarray
        (H, W) log-belief from init_sound_map. Modified in place and returned.
    level_db : float, optional
        Received level. Pass it -- simSoundReading already measures it, and
        without it every cell along the bearing stays equally likely.
    temper : float
        Down-weights consecutive readings that are correlated (a robot standing
        still hears essentially the same thing twice, and treating those as
        independent overcounts the evidence).

    Notes
    -----
    `**_legacy` swallows delta_alpha / half_width_deg from old call sites so
    they degrade to bearing-only updates instead of raising.
    """
    log_belief += temper * bearing_loglik(x_points, z_points, robot_x, robot_z,
                                          robot_yaw_deg, theta_t_deg, sigma_deg)
    if use_range and level_db is not None:
        log_belief += temper * range_loglik(x_points, z_points, robot_x, robot_z,
                                            level_db, L0_db)
    # Anchor so repeated updates cannot drift toward -inf over a long run.
    log_belief -= log_belief.max()
    return log_belief


def sound_posterior(log_belief):
    """Normalised (H, W) belief map from the log-belief.

    Max-subtracted before exponentiating, so a long run cannot underflow the
    way a raw product of likelihoods would.
    """
    lb = np.asarray(log_belief, float)
    lb = lb - lb.max()
    p = np.exp(lb)
    total = p.sum()
    return p / total if total > 1e-12 else np.full_like(p, 1.0 / p.size)


def sound_entropy(log_belief, bits=True):
    """Shannon entropy of the auditory posterior.

    Replaces sound_dirichlet_entropy, which is no longer meaningful now that
    the map holds log-likelihood rather than Dirichlet counts -- and which was
    already the wrong statistic: it rose with total evidence mass as well as
    falling with belief concentration, so it could climb while the belief was
    genuinely sharpening.
    """
    p = sound_posterior(log_belief).ravel()
    p = p[p > 1e-12]
    H = float(-(p * np.log(p)).sum())
    return H / np.log(2.0) if bits else H


def sound_dirichlet_entropy(log_belief, bits=True):
    """Deprecated alias for sound_entropy, kept so old call sites keep running."""
    return sound_entropy(log_belief, bits)



def range_loglik_marginal(x_points, z_points, observations,
                          sigma_db=RANGE_SIGMA_DB,
                          L0_prior_mean=L0_PRIOR_MEAN,
                          L0_prior_sigma=L0_PRIOR_SIGMA):
    """Range constraint from received levels with the source level UNKNOWN.

    For a hypothesised source cell c and observation i taken at range r_i(c),
    the source level implied by that reading is

        d_i(c) = L_i + 20 * log10(r_i(c))

    If the source really is at c, every observation must imply the SAME L0, so
    the spread of d_i(c) across observations is the evidence. Marginalising a
    flat prior over L0 gives, up to a constant,

        log p(levels | c) = -0.5 * sum_i (d_i(c) - mean_i d_i(c))^2 / sigma^2

    which is exactly "after fitting the best single source level, how well do
    the readings agree?". No assumption about L0 survives.

    With a single observation the residuals are identically zero everywhere, so
    this correctly returns a flat map: one level reading from an unknown source
    carries no range information at all. Two readings at different distances is
    the minimum, which is one reason the two-robot setup helps here -- the team
    gets a level difference every step instead of having to move to earn one.

    Parameters
    ----------
    observations : list[tuple[float, float, float]]
        (robot_x, robot_z, level_db) for every detection so far.
    L0_prior_mean, L0_prior_sigma : float, optional
        Weak Gaussian prior on the source level, e.g. a typical level for the
        class CLAP identified. Leave as None for a flat prior.

    Returns
    -------
    np.ndarray
        (H, W) log-likelihood map, max-anchored at 0.
    """
    X, Z = np.meshgrid(np.asarray(x_points, float), np.asarray(z_points, float))
    if not observations or len(observations) < 2:
        return np.zeros(X.shape)

    D = np.stack([lv + 20.0 * np.log10(np.maximum(np.hypot(X - rx, Z - rz),
                                                  SOUND_MIN_DIST))
                  for (rx, rz, lv) in observations], axis=0)   # (n, H, W)

    n = D.shape[0]
    prec = 1.0 / max(sigma_db, 1e-6) ** 2
    if L0_prior_mean is None or L0_prior_sigma is None:
        L_hat = D.mean(axis=0, keepdims=True)
        out = -0.5 * prec * np.sum((D - L_hat) ** 2, axis=0)
    else:
        p0 = 1.0 / max(L0_prior_sigma, 1e-6) ** 2
        L_hat = ((D.sum(axis=0) * prec + L0_prior_mean * p0) / (n * prec + p0))
        out = (-0.5 * prec * np.sum((D - L_hat[None, :, :]) ** 2, axis=0)
               - 0.5 * p0 * (L_hat - L0_prior_mean) ** 2)
    return out - out.max()


def sound_belief_map(log_belief, observations=None, range_mode=RANGE_MODE,
                     x_points=None, z_points=None, **kw):
    """Combine the incremental bearing belief with the range constraint.

    The bearing term accumulates incrementally in `log_belief`; the marginal
    range term cannot, because removing the fitted L0 couples all observations
    together, so it is recomputed from the stored history each step. The
    history is a few dozen tuples, so this is cheap.
    """
    total = np.asarray(log_belief, float)
    if range_mode == 'marginal' and observations and x_points is not None:
        total = total + range_loglik_marginal(x_points, z_points, observations, **kw)
    return total


# ============================================================= CLAP SEMANTICS

def _load_clap():
    """Lazily load laion_clap. Returns None when unavailable."""
    global _CLAP_MODEL
    if _CLAP_MODEL is not None:
        return _CLAP_MODEL
    if not CLAP_ENABLED:
        return None
    try:
        import laion_clap
        m = laion_clap.CLAP_Module(enable_fusion=False)
        m.load_ckpt(CLAP_CKPT)
        _CLAP_MODEL = m
        print("[sound] CLAP loaded.")
    except Exception as e:
        print(f"[sound] CLAP unavailable ({e}); falling back to SBERT text matching.")
        _CLAP_MODEL = None
    return _CLAP_MODEL


def _cos(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T


def class_sound_similarity(sound_query, classes, clip_negative=CLIP_NEGATIVE_SIM):
    """Cosine similarity between a heard sound and each YOLO class name.

    CLAP puts audio and text in one shared space, so an audio clip can be
    compared directly against class-name text embeddings. That is the point of
    using CLAP here rather than SBERT: it grounds the actual waveform, not a
    human-written description of it.

    Parameters
    ----------
    sound_query : tuple[str, str]
        ('audio', path) or ('text', label), as returned by SoundSource.query().
    classes : list[str]
        Class names, in the canonical CLASSES order of the object map.

    Returns
    -------
    np.ndarray
        (K,) similarity vector aligned with `classes`.
    """
    kind, payload = sound_query
    key = (kind, payload, tuple(classes), clip_negative)
    if key in _CLAP_CACHE:
        return _CLAP_CACHE[key]

    # Class names are terse labels; a short prompt gives CLAP better grounding.
    prompts = [f"the sound of a {c}" for c in classes]

    clap = _load_clap()
    if clap is not None:
        try:
            cls_emb = clap.get_text_embedding(prompts, use_tensor=False)
            if kind == 'audio':
                q_emb = clap.get_audio_embedding_from_filelist([payload], use_tensor=False)
            else:
                q_emb = clap.get_text_embedding([payload], use_tensor=False)
            s = _cos(np.asarray(q_emb)[0][None, :], np.asarray(cls_emb))[0]
        except Exception as e:
            print(f"[sound] CLAP embedding failed ({e}); falling back to SBERT.")
            s = None
    else:
        s = None

    if s is None:
        # Fallback keeps the pipeline runnable without the CLAP checkpoint.
        # Text-only, so an audio path degrades to its filename stem.
        from voa_functions.visionFunction import model as _sbert
        from sentence_transformers import util as _util
        text = payload if kind == 'text' else os.path.splitext(os.path.basename(payload))[0]
        q = _sbert.encode(text, convert_to_tensor=True)
        c = _sbert.encode(prompts, convert_to_tensor=True)
        s = _util.cos_sim(q, c)[0].cpu().numpy()

    s = np.asarray(s, dtype=float)
    if clip_negative:
        s = np.clip(s, 0.0, None)
    # Empty floor makes no sound either -- same reasoning as the visual side.
    # Both vectors feed visual_likelihood_multimodal, so a nonzero score here
    # would leak straight back in through combine_similarity.
    try:
        from voa_functions.visionFunction import zero_empty_similarity
        s = zero_empty_similarity(s)
    except Exception:
        pass
    _CLAP_CACHE[key] = s
    return s


def sound_semantic_map(beta_objects, sound_query, classes):
    """Semantic auditory likelihood: object posterior projected onto CLAP scores.

        L_sem(c) = sum_k p(o_k | z_c) * clap_sim(class_k, sound)

    Structurally identical to visual_likelihood, which is what makes the two
    branches directly comparable in the ablation.
    """
    from voa_functions.visionFunction import object_posterior
    post = object_posterior(beta_objects)
    sim = class_sound_similarity(sound_query, classes)
    lik = np.tensordot(post, sim, axes=([2], [0]))
    total = lik.sum()
    return lik / total if total > 1e-12 else np.full_like(lik, 1.0 / lik.size)