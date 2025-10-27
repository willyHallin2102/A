"""
    src/mockup/antenna.py
    ---------------------
    Antenna is the basic antenna objects which aims to replicate the behaviour of
    receiver antennas as transmitting antennas. It implements an ideal antenna 
    `Isotropic` which spread signals evenly. Furthermore, this scripts implements
    a `3GPP` antenna class which follows the behaviour of the TR 38.901 definitions.
"""
from typing import Callable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from src.math.coords import add_angles, sub_angles


ArrayLike = Union[np.ndarray, float, int]

def plot_pattern(
    pattern_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_theta: Optional[int]=None, n_phi: Optional[int]=None,
    theta: ArrayLike=0, phi: ArrayLike=0,
    plot_type: str='rect_phi', ax: Optional[plt.Axes]=None, ax_label: bool=True,
    **kwargs
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, plt.Axes, Optional[plt.Axes]]
]:
    """
    Computes and optionally plot an antenna radiation pattern.

    Args:
    -----
        pattern_fn: Function of (phi, theta) returning a scalar, the 
                    function to plot, e.g., gain.
        n_theta, n_phi: Number of theta and phi angular values.
        theta, phi: arrays of elevation and azimuth angles respectively.
        plot_type:  Plot type, how to visualize the values, e.g.,
                    `rect_phi`, `polar_phi`:
                        - Rectangular or plot of the pattern vs. phi
                          one curve per `theta` value.
                    `2d`: 2D imshow plot
        ax_label:   Boolean parameter whether to indicate ax label 
                    visualized in the plot or not.
    
    Returns:
    --------
        phi:  (nphi,) Array of azimuth angles of the plot
        theta:  (ntheta,) Array of elevation angles of the plot            
        val:  array of values of the pattern 
        ax:  Image axes 
    """
    # Uniform sampling setup
    phi = np.linspace(-180, 180, n_phi) if n_phi else np.atleast_1d(phi)
    theta = np.linspace(-90, 90, n_theta) if n_theta else np.atleast_1d(theta)

    # Vectorized mesh and evaluate pattern in a single call
    phi_mat, theta_mat = np.meshgrid(phi, theta, indexing='xy')
    v = np.asarray(pattern_fn(
        phi_mat.ravel(), theta_mat.ravel())
    ).reshape(theta_mat.shape)

    if plot_type == 'none': return phi, theta, v
    if ax is None:
        ax = plt.axes(projection='polar' if 'polar' in plot_type else None)
    im = None

    if plot_type.endswith('_phi'):
        x = np.radians(phi) if plot_type.startswith('polar') else phi
        ax.plot(x, v.T, **kwargs)
        if ax_label and not plot_type.startswith('polar'):
            ax.set_xlabel('Azimuth (deg)')
            ax.set_xlim([-180, 180])

    elif plot_type.endswith('_theta'):
        x = np.radians(theta) if plot_type.startswith('polar') else theta
        ax.plot(x, v, **kwargs)
        if ax_label and not plot_type.startswith('polar'):
            ax.set_xlabel('Elevation (deg)')
            ax.set_xlim([-90, 90])
    
    elif plot_type == '2d':
        im = ax.imshow(
            np.flipud(v), aspect='auto',
            extent=[phi.min(), phi.max(), theta.min(), theta.max()],
            **kwargs
        )
        if ax_label:
            ax.set_xlabel('Azimuth (deg)')
            ax.set_ylabel('Elevation (deg)')
    else:
        raise ValueError(f'Unknown plot type {plot_type}')
    
    return phi, theta, v, ax, im



# ---------------========== Antenna's Elements ==========--------------- #

class ElementBase(ABC):
    """
    Base antenna class, represents the requirement API
    """
    @abstractmethod
    def response(self, phi: np,ndarray, theta: np.ndarray) -> np.ndarray:
        """
        The directivity response of the element

        Args:
        -----
            phi:    Azimuth angles measured in degrees
            theta:  Elevation angles measured in degrees.
        
        Returns:
        --------
            Antenna gain measured in dBi.
        """
        raise NotImplementedError('response method not implemented')
    
    def compute_gain_mean(self, n_samples: int=1000, seed: int=42) -> float:
        """
        Compute the mean antenna gain over a channel measured in dBi. For 
        and ideal lossless antenna (isotropic antenna), this method should
        approximate around 0 dBi.

        Args:
        -----
            n_samples:  The number of samples the vector passed holds.
            seed:   For reproducibility.
        """
        rng = np.random.default_rng(seed=seed)
        phi = rng.uniform(-180, 180, n_samples)
        theta = rng.uniform(0, 180, n_samples)

        # weighting of the gain from the vector
        weights = np.sin(np.radians(theta))

        # Get the gain in linear scale and weight the mean
        gain_linear = np.power(10.0, 0.1 * self.response(phi, theta))
        gain_mean = np.average(gain_linear, weights=weights)

        return 10.0 * np.log10(gain_mean)
    
    def plot_pattern(self, **kwargs):
        """
        Plots the gain pattern.

        Args:
        -----
            **kwargs:   dictionary of additional parameters
        
        Returns:
        --------
            see `src.mockup.antenna.plot_pattern()`.
        """
    

class ElementIsotropic(ElementBase):
    """
    Isotropic antenna element model 0 dBi uniform gain. Special antenna 
    element representing an ideal radiator that emits energy equally in 
    all directions.
    """
    def __init__(self): super().__init__()

    def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return a constant zero-gain array."""
        return np.zeros_like(np.asarray(phi), dtype=float)


class Element3GPP(ElementBase):
    """
    3GPP TR 38.901 Base station antenna element and follow those definitions
    made for the wireless communication.

    Models azimuth and elevation directivity according to the 3GPP standard, 
    with configurable beamwidths and side-lobe suppression.
    """
    def __init__(self,
        phi_0: float=0.0, theta_0: float=0.0,
        phi_beamwidth: float=120.0, theta_beamwidth: float=65.0
    ):
        """
            Initialize 3GPP Element Antenna Instance
        """
        super().__init__()
        self.phi_0, self.theta_0 = float(phi_0), float(theta_0)
        self.phi_bw, self.theta_bw = float(phi_beamwidth), float(theta_beamwidth)
        
        # 3GPP parameters
        self.slav = 30.0   # vertical side-lobe attenuation limit
        self.am = 30.0     # front-back attenuation limit
        self.max_gain = 0.0  # reference gain (dBi)

        # Calibrate so that mean gain ≈ 0 dBi
        self.calibrate()
    
    # def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    #     """Computes antenna gain for angles."""
    #     # Rotate angles if needed
    #     if self.theta_0 or self.phi_0:
    #         phi_1, theta_1 = sub_angles(
    #             self.phi_0, 90 - self.theta_0, phi, 90 - theta
    #         )
    #     else:
    #         phi1, theta1 = phi, theta

    #     # Normalize phi to [-180, 180)
    #     phi1 = ((phi1 + 180) % 360) - 180

    #     # Vertical pattern
    #     av = 0 if self.theta_bw <= 0 else -np.minimum(12 * (theta1 / self.theta_bw) ** 2, self.slav)

    #     # Horizontal pattern
    #     ah = 0 if self.phi_bw <= 0 else -np.minimum(12 * (phi1 / self.phi_bw) ** 2, self.am)

    #     # Final gain
    #     return self.max_gain - np.minimum(-av - ah, self.am)

    def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Computes antenna gain for angles (vectorized or grid).
        """
        phi = np.asarray(phi)
        theta = np.asarray(theta)

        # Broadcast to common shape if needed
        phi, theta = np.broadcast_arrays(phi, theta)

        if self.theta_0 or self.phi_0:
            phi1, theta1 = sub_angles(self.phi_0, 90 - self.theta_0, phi, 90 - theta)
        else:
            phi1, theta1 = phi, theta

        # Normalize phi to [-180, 180)
        phi1 = ((phi1 + 180) % 360) - 180

        Av = -np.minimum(12 * (theta1 / self.theta_bw) ** 2, self.slav)
        Ah = -np.minimum(12 * (phi1 / self.phi_bw) ** 2, self.am)

        return self.max_gain - np.minimum(-Av - Ah, self.am)

    def calibrate(self, n_samples: int = 10000) -> None:
        """
        Calibrates the maximum antenna gain.
        """
        self.max_gain -= self.compute_gain_mean(n_samples)
