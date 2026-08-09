import random
import numpy as np
import math

# ==========================
# Olfactory Functions
# ==========================

# def gaussian_plume(x, z, source, q_s=200, D=10, U=0, tau=10, del_t=1, psi_deg=0):
def gaussian_plume(x, z, source, q_s=2000, D=10, U=0, tau=1000, psi_deg=0):
    """Computes the odor concentration at (x, z) based on a Gaussian plume model.

    This model accounts for diffusion (D), source strength (q_s),
    advection velocity (U) at an angle (psi_deg), and decay (tau, del_t).

    Parameters
    ----------
    x : float
        The x-coordinate where concentration is evaluated.
    z : float
        The z-coordinate where concentration is evaluated.
    source : tuple[float, float]
        The (x_s, z_s) coordinates of the odor source.
    q_s : float, optional
        Source emission rate (strength). Defaults to 2000.
    D : float, optional
        Diffusion coefficient. Defaults to 10.
    U : float, optional
        Advection velocity magnitude (wind speed). Defaults to 0.
    tau : float, optional
        Time constant related to decay or stability. Defaults to 1000.
    del_t : float, optional
        Time step factor affecting decay over distance. Defaults to 1.
    psi_deg : float, optional
        Angle of advection velocity (degrees from positive z-axis?). Defaults to 0.

    Returns
    -------
    float
        The calculated odor concentration at point (x, z). Returns a large number
        if (x, z) is exactly at the source to avoid division by zero.
    """
    # Convert psi from degrees to radians
    psi = math.radians(psi_deg)
    
    # Compute lambda; note that if U==0, lambda simplifies to sqrt(D*tau)
    lambd = math.sqrt((D * tau) / (1 + (tau * U**2) / (4 * D)))
    # print(f"Lambda: {lambd}")
    
    # Loop over each source
    x_s, z_s = source  # Unpack the source coordinates
    
    # Compute differences in x and z relative to the odor source
    delta_x = (x) - (x_s)
    delta_z = (z) - (z_s)

    # Euclidean distance in the X-Z plane
    r = math.sqrt(delta_x**2 + delta_z**2)
    # print(f"Delta_x: {delta_x}, Delta_z: {delta_z}, r: {r}")
    
    # Avoid division by zero if r==0
    if r == 0:
        r = 1e-3  # record non-zero small r

    # Compute the rotated z coordinate (this incorporates advection if U != 0)
    del_z = -(delta_x * math.cos(psi) + delta_z * math.sin(psi))
    # print(f"del_z: {del_z}")
    concentration = (q_s / (4 * math.pi * D * r))
    advection = math.exp((-del_z * U) / (2 * D) - (r / lambd))
    total = concentration * advection
    # print(f"Concentration: {concentration}, Advection: {advection}, Total: {total}")
    # return total + 1
    return total


def simChemicalReading(source, robot_x, robot_z, sigma_noise=0.1):
    """Simulates a noisy chemical sensor reading at the agent's current location.

    Calculates the expected concentration using `gaussian_plume` at the agent's
    position relative to the source and adds Gaussian noise.

    Parameters
    ----------
    source : tuple[float, float]
        The (x_s, z_s) coordinates of the odor source.
    controller : ai2thor.controller.Controller
        The AI2-THOR controller instance.
    sigma_noise : float, optional
        Standard deviation of the Gaussian noise added to the reading. Defaults to 0.5.

    Returns
    -------
    float
        The simulated noisy odor concentration reading.
    """
    # robot_x, robot_y, robot_z = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
    # Calculate true concentration at robot location
    true_concentration = gaussian_plume(robot_x, robot_z, source)
    # Add Gaussian noise (note: noise std dev is scaled by 4 here)
    noisy_reading = true_concentration + np.random.normal(0, sigma_noise)
    return noisy_reading

# ==========================
# Bayesian Functions
# ==========================

class BayesianAgent:
    """Manages the Bayesian belief map for odor source localization."""
    def __init__(self, pos, src_pos, x_points, z_points, sigma_noise, reachable_positions):
        """Initializes the Bayesian agent with grid info and model parameters.

        Parameters
        ----------
        pos : np.ndarray
            Initial grid position [row, col] of the agent. (Seems unused after init?)
        src_pos : np.ndarray
            Grid position [row, col] of the true source. (Used for evaluation/debug?)
        x_points : np.ndarray
            1D array of x-coordinates defining the grid columns.
        z_points : np.ndarray
            1D array of z-coordinates defining the grid rows.
        sigma_noise : float
            Standard deviation of sensor noise, used in the likelihood calculation.
        reachable_positions : list[tuple[float, float]]
             List of reachable (x, z) world coordinates. (Seems unused after init?)
        """
        self.pos = pos
        self.src_pos = src_pos
        self.x_points = x_points
        self.z_points = z_points
        self.window_x = len(x_points)
        self.window_z = len(z_points)
        # Start with a uniform belief over the grid
        self.prob_map = np.full((self.window_z, self.window_x), 1.0 / (self.window_z * self.window_x))
        self.sigma_noise = sigma_noise
        self.reachable_positions = reachable_positions

    def prior(self) -> np.ndarray:
        """Returns the current belief map (probability distribution) over the grid.

        Returns
        -------
        np.ndarray
            The current probability map (window_z, window_x).
        """
        return self.prob_map

    def likelihood(self, current_odor_concentration, expected, sigma_noise):
        """Calculates the likelihood of observing a concentration given an expected value.

        Assumes Gaussian sensor noise. P(measurement | source_at_expected_location).

        Parameters
        ----------
        current_odor_concentration : float
            The actual measured odor concentration.
        expected : float
            The concentration expected if the source were at a specific grid location,
            calculated using `gaussian_plume`.
        sigma_noise : float
            The standard deviation of the sensor noise.

        Returns
        -------
        float
            The likelihood value.
        """
        # Gaussian probability density function
        exponent = -((current_odor_concentration - expected) ** 2) / (2 * sigma_noise ** 2)
        denominator = (np.sqrt(2 * np.pi) * sigma_noise)
        # Add small epsilon to denominator to prevent division by zero if sigma_noise is tiny
        return np.exp(exponent) / (denominator + 1e-9)

    
    def entropy(self, prob_map: np.ndarray) -> float:
        """Calculates the Shannon entropy of a probability map.

        Parameters
        ----------
        prob_map : np.ndarray
            A 2D array representing the probability distribution over the grid.
            Should sum to 1.0.

        Returns
        -------
        float
            The entropy of the distribution in bits.
        """
        eps = 1e-12 # Small epsilon to avoid log2(0)
        # Ensure probabilities are valid (non-negative) and clip for numerical stability
        p = np.clip(prob_map, eps, 1.0)
        # Normalize just in case it's slightly off
        p = p / np.sum(p)
        # p = scaler.fit_transform(p)
        return -np.sum(p * np.log2(p))

    def posterior(self, current_odor_concentration, robot_x, robot_z, smooth_sigma=2.0):
        """Updates the belief map using Bayes' theorem based on the latest odor reading.

        Calculates the posterior P(source_location | measurement) by multiplying
        the prior P(source_location) with the likelihood P(measurement | source_location)
        for every possible source location on the grid, then normalizes. Optionally
        applies Gaussian smoothing.

        Parameters
        ----------
        current_odor_concentration : float
            The latest noisy odor reading from `simChemicalReading`.
        robot_x : float
            The current world x-coordinate of the agent.
        robot_z : float
            The current world z-coordinate of the agent.
        smooth_sigma : float, optional
            Standard deviation for the Gaussian filter applied after the update.
            If 0 or negative, no smoothing is applied. Defaults to 1.0.
        """
        # 1) Bayesian update (Prior * Likelihood)
        likelihood_map = np.zeros_like(self.prob_map)
        for iz, z_source in enumerate(self.z_points):
            for ix, x_source in enumerate(self.x_points):
                # Calculate expected reading IF source was at (x_source, z_source)
                expected = gaussian_plume(robot_x, robot_z, (x_source, z_source))
                # Calculate likelihood of current measurement given that expectation
                likelihood_map[iz, ix] = self.likelihood(current_odor_concentration, expected, self.sigma_noise)

        # Pointwise multiplication: Posterior ~ Prior * Likelihood
        self.prob_map *= likelihood_map

        # 2) Normalize to make it a valid probability distribution again
        map_sum = np.sum(self.prob_map)
        if map_sum > 1e-9: # Avoid division by zero if map becomes all zeros
            #  self.prob_map = scaler.fit_transform(self.prob_map)
             self.prob_map /= map_sum
        else:
             print("Warning: Probability map sum is close to zero after update.")
             # Reset to uniform if probability collapses
             self.prob_map = np.full((self.window_z, self.window_x), 1.0 / (self.window_z * self.window_x))