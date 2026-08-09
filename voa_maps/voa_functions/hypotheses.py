"""
hypotheses.py  --  unknown source strength, tracked as a few hypotheses.

Both the olfactory and auditory branches have the same problem: the forward
model needs a source-strength constant that nobody knows.

    olfaction   c(r) = q_s / (4 pi D r) * exp(-r/lambda)      q_s unknown
    audition    L(r) = L0 - 20 log10(r)                       L0 unknown

Both are unidentifiable from a single reading -- a weak source nearby and a
strong source far away give the same number. Worse, guessing is not safe:
measured on a 4 m grid, an emission rate wrong by 10x pushed localisation from
0.39 m to 2.58 m, and a loudness wrong by 15 dB from 0.18 m to 2.83 m.

Both also become identifiable across several readings, because the strength
cancels in differences:

    log c_i - log c_j  depends only on r_i, r_j     (q_s cancels)
    L_i - L_j          depends only on r_i, r_j     (L0 cancels)

So this module carries the strength as a nuisance parameter over a small grid
of hypotheses, updates it jointly with the source location, and marginalises it
out when the location belief is needed.

WHY ONE CLASS FOR BOTH
----------------------
After the right transform, the two models are the same equation:

    observed = strength + predictor(r) + noise

with (log concentration, log q_s, log f(r)) for olfaction and (dB level, L0,
-20 log10 r) for audition. Nothing in this file knows which modality it is
serving.

WHAT THE MEASUREMENTS SAY ABOUT PRIORS
--------------------------------------
The hypothesis posterior collapses onto the truth within 2-3 readings even when
started from a deliberately wrong prior (0.8 on a wrong hypothesis, in both
modules). The data dominates almost immediately. A semantic prior from SBERT is
therefore optional decoration, not load-bearing -- and a uniform prior is one
fewer assumption to defend. `semantic_prior` exists if you want it, but the
default is uniform on purpose.
"""

import numpy as np

EPS = 1e-300


class ScaleHypotheses:
    """Joint belief over (source location, source strength).

    Parameters
    ----------
    values : sequence of float
        Candidate strengths, already in the transformed space -- log(q_s) for
        olfaction, dB for audition. Use log_space_grid / db_grid to build them.
    shape : tuple[int, int]
        (H, W) of the belief grid.
    prior : sequence of float, optional
        Prior over `values`. Defaults to uniform, which the measurements show is
        sufficient.
    label : str
        Only used in __repr__.
    """

    def __init__(self, values, shape, prior=None, label='strength'):
        self.values = np.asarray(values, float)
        self.K = len(self.values)
        self.shape = tuple(shape)
        self.label = label
        p = np.ones(self.K) / self.K if prior is None else np.asarray(prior, float)
        p = np.clip(p, 1e-12, None)
        self.log_prior = np.log(p / p.sum())
        # log joint over (hypothesis, cell), unnormalised
        self.log_joint = np.tile(self.log_prior[:, None, None], (1,) + self.shape)
        self.n_updates = 0

    def update(self, predictor, observed, sigma):
        """Fold in one reading.

        Parameters
        ----------
        predictor : np.ndarray
            (H, W) model term for this robot pose, for every hypothesised source
            cell, EXCLUDING the strength: log f(r) for olfaction, the
            attenuation -20 log10 r for audition.
        observed : float
            The reading in the matching transformed space: log concentration for
            olfaction, dB level for audition.
        sigma : float
            Noise standard deviation in that same space.
        """
        pred = np.asarray(predictor, float)
        for k, v in enumerate(self.values):
            d = (float(observed) - (v + pred)) / max(sigma, 1e-9)
            self.log_joint[k] += -0.5 * d * d
        self.log_joint -= self.log_joint.max()
        self.n_updates += 1
        return self

    def cell_loglik(self):
        """(H, W) log-likelihood of the location, strength marginalised out."""
        m = self.log_joint.max()
        w = np.exp(self.log_joint - m)
        out = np.log(w.sum(axis=0) + EPS)
        return out - out.max()

    def cell_posterior(self):
        lp = self.cell_loglik()
        p = np.exp(lp - lp.max())
        s = p.sum()
        return p / s if s > EPS else np.full(self.shape, 1.0 / p.size)

    def hypothesis_posterior(self):
        """(K,) posterior over the strength hypotheses."""
        m = self.log_joint.max()
        w = np.exp(self.log_joint - m).sum(axis=(1, 2))
        s = w.sum()
        return w / s if s > EPS else np.ones(self.K) / self.K

    def map_value(self):
        """Most probable strength, in the transformed space."""
        return float(self.values[int(np.argmax(self.hypothesis_posterior()))])

    def at_endpoint(self, threshold=0.5):
        """True when the posterior has piled up on the lowest or highest value.

        That is the signature of a true strength OUTSIDE the grid: the tracker
        cannot represent it, so it pegs at whichever end is closest and reports
        converged() with high confidence. Check this before trusting a
        converged flag, and widen the grid if it fires.
        """
        h = self.hypothesis_posterior()
        return bool(h[0] >= threshold or h[-1] >= threshold)

    def converged(self, threshold=0.9):
        """True once one hypothesis holds most of the mass.

        Useful as a diagnostic: if this never becomes True, the robot has not
        yet visited positions at different enough ranges for the strength to be
        identifiable, and the location estimate is correspondingly soft.
        """
        return bool(self.hypothesis_posterior().max() >= threshold)

    def __repr__(self):
        h = self.hypothesis_posterior()
        pairs = ", ".join(f"{v:.3g}:{p:.2f}" for v, p in zip(self.values, h))
        return f"<ScaleHypotheses {self.label} n={self.n_updates} [{pairs}]>"


# ============================================================= GRID BUILDERS

def log_space_grid(q_s_values=None, lo=100.0, hi=10000.0, n=5):
    """Olfactory hypotheses: emission rates in mg/L -> log space.

    FIVE is enough. Measured with the true rate drawn at random so it never
    coincides with a grid point:

        3 hypotheses  (10x apart)   0.97 m mean error, 26% within 0.4 m
        5 hypotheses  (3.2x apart)  0.65 m, 46%
        9 hypotheses  (1.8x apart)  0.70 m, 39%
        17 hypotheses (1.3x apart)  0.63 m, 46%

    Only 3 is clearly bad; from 5 upward the differences are noise. Denser
    grids do not help because the residual error is no longer discretisation,
    it is the geometry of where the robot happened to sample.

    Do NOT compensate for the coarse grid by tightening sigma. Sigma should be
    INFLATED (about 3x the raw measurement noise) -- see the note in
    fusion_controller. A coarse grid with a tight sigma produces a
    confidently wrong modality that poisons the fusion.

    Note these absorb the MQ3 gain. counts_net = a * q_s * f(r), so only the
    PRODUCT a*q_s is identifiable -- which means fitting the sensor slope
    offline and tracking q_s online are two ways of pinning the same unknown.
    Tracking it online is the one that survives a new sensor, a new gas, or a
    warmer day, so the offline slope calibration can be dropped. The clean-air
    baseline still has to be subtracted, because that is an additive offset and
    does not cancel.

    Because of the grid spacing, the MAP value is the nearest hypothesis, NOT
    an estimate of the true emission rate. Report it as a nuisance parameter,
    not as a measurement.
    """
    if q_s_values is None:
        q_s_values = 10.0 ** np.linspace(np.log10(lo), np.log10(hi), n)
    return np.log(np.asarray(q_s_values, float))


def _unused_log_space_grid(q_s_values=(100.0, 1000.0, 10000.0)):
    """Olfactory hypotheses: emission rates in mg/L -> log space.

    Note these absorb the MQ3 gain. counts_net = a * q_s * f(r), so only the
    PRODUCT a*q_s is identifiable -- which means fitting the sensor slope
    offline and tracking q_s online are two ways of pinning the same unknown.
    Tracking it online is the one that survives a new sensor, a new gas, or a
    warmer day, so the offline slope calibration can be dropped. The clean-air
    baseline still has to be subtracted, because that is an additive offset and
    does not cancel.
    """
    return np.log(np.asarray(q_s_values, float))


def db_grid(l0_values=None, lo=45.0, hi=105.0, n=5):
    """Auditory hypotheses: source level at 1 m, in dB. Already log-space.

    Five points over 45-105 dB gives 15 dB spacing. That is coarse relative to
    the level noise, which is the intent: the resulting posterior is broad but
    honest, and fusion tightens it. Pair it with an inflated sigma.
    """
    if l0_values is None:
        l0_values = np.linspace(lo, hi, n)
    return np.asarray(l0_values, float)


def semantic_prior(model, source_phrase, hypothesis_phrases):
    """Optional SBERT prior over strength hypotheses.

    e.g. source_phrase = "a toilet flushing", hypothesis_phrases =
    ["a quiet sound", "a loud sound", "a very loud sound"].

    The measurements say this is decoration: both branches converge onto the
    correct hypothesis within 2-3 readings even from a prior that puts 0.8 on a
    wrong one. Whether SBERT actually encodes real-world loudness or emission
    strength is untested here, so if you use this, test it first -- otherwise
    it is an unjustified assumption buying nothing.
    """
    from sentence_transformers import util
    q = model.encode(source_phrase, convert_to_tensor=True)
    h = model.encode(list(hypothesis_phrases), convert_to_tensor=True)
    s = util.cos_sim(q, h)[0].cpu().numpy()
    s = np.clip(s, 0.0, None)
    return s / s.sum() if s.sum() > 0 else np.ones(len(s)) / len(s)


# ============================================================= PREDICTORS

def olfactory_predictor(X, Z, robot_x, robot_z, D=10.0, tau=1000.0,
                        U=0.0, psi_deg=0.0):
    """log f(r): the plume shape at unit emission rate, for every source cell.

    Keeping D and tau at their literature values and tracking only q_s is the
    right split: q_s is the one that changes with the gas, the container and
    the room, while D and tau describe the medium.
    """
    lambd = np.sqrt((D * tau) / (1.0 + (tau * U ** 2) / (4.0 * D)))
    psi = np.deg2rad(psi_deg)
    dx = robot_x - np.asarray(X, float)
    dz = robot_z - np.asarray(Z, float)
    r = np.maximum(np.hypot(dx, dz), 1e-3)
    del_z = -(dx * np.cos(psi) + dz * np.sin(psi))
    return (-np.log(4.0 * np.pi * D * r)
            + (-del_z * U) / (2.0 * D) - r / lambd)


def auditory_predictor(X, Z, robot_x, robot_z, min_r=0.2):
    """Attenuation in dB, -20 log10 r, for every hypothesised source cell."""
    r = np.maximum(np.hypot(np.asarray(X, float) - robot_x,
                            np.asarray(Z, float) - robot_z), min_r)
    return -20.0 * np.log10(r)