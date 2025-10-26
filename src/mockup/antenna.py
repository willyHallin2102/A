"""
    src/mockup/antenna.py
    ---------------------
    Antenna element modelling and visualization utilities for plotting various
    patterns.
    This module define the base and derived classes representing antenna elements,
    along with plotting utilities that enable visualizing radiation patterns. This 
    current implementation support and provide implementation for a 
    `Isotropic antenna` and `3GPP`-compliant element patterns, suitable for 
    wireless system simulations (for raytracing comparison).
"""
from __future__ import annotations
from abc import ABC, abstractclassmethod
from typing import Callable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from src.math.coords import add_angles, sub_angles


# ---------------========== Plotting Utilities ==========--------------- #

def plot_pattern(
    pattern_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta: Optional[Union[np.ndarray, float]]=None, 
    phi: Optional[Union[np.ndarray, float]]=None,
    n_theta: Optional[int]=None, n_phi: Optional[int]=None,
    plot_type: str="rect_phi", ax: Optional[Axes]=None, ax_label: bool=True,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Axes], Optional[AxesImage]]:
    """
    Computes and optionally plot an antenna radiation pattern.

    Args:
    ------
        pattern_fn: Function returning the antenna gain (dBi) as a function 
                    of azimuth (phi) and elevation (theta). Must accept two 
                    numpy arrays (phi, theta) and return an array of matching 
                    shape.
        theta:  Elevation angles in degrees. If None, a grid is generated.
        phi:    Azimuth angles in degrees. If None, a grid is generated.
        n_theta:    Number of elevation samples if theta is None.
        n_phi:  Number of azimuth samples if phi is None.
        plot_type : Type of plot to produce, among following,
                - "none": return computed arrays only
                - "rect_phi": rectangular azimuth cut
                - "rect_theta": rectangular elevation cut
                - "polar_phi": polar azimuth cut
                - "polar_theta": polar elevation cut
                - "2d": heatmap (azimuth vs elevation)
        ax: Axes to plot on. If None, new axes are created.
        ax_label:   Whether to add axis labels.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plot functions.

    Returns:
    --------
        phi : ndarray - Azimuth angle grid in degrees.
        theta : ndarray - Elevation angle grid in degrees.
        v : ndarray - Computed antenna gain (dBi).
        ax : Axes or None - Matplotlib axes used for plotting.
        im : AxesImage or None - Image handle if applicable (for 2D plots).
    """
    # Create angle grids
    phi = np.linspace(-180, 180, n_phi) if n_phi else np.atleast_1d(phi)
    theta = np.linspace(-90, 90, n_theta) if n_theta else np.atleast_1d(theta)
    phi, theta = np.meshgrid(phi, theta, indexing="xy")

    # Evaluate pattern
    v = np.asarray(pattern_fn(phi, theta))

    # If no plotting is requested, return angles and antenna gain (v) only
    if plot_type == "none":
        return phi, theta, v, None, None

    # Select axis type
    if ax is None:
        ax = plt.axes(projection="polar") if "polar" in plot_type else plt.gca()
    
    im: Optional[AxesImage] = None

    # Choose plotting mode 
    if plot_type.endswith("phi"):
        x = np.radians(phi[0]) if "polar" in plot_type else phi[0]
        y = v.T
        ax.plot(x, y, **kwargs)
        if ax_label and "rect" in plot_type:
            ax.set_xlabel("Azimuth (deg)")
            ax.set_xlim([-180, 180])

    elif plot_type.endswith("theta"):
        x = np.radians(theta[:, 0]) if "polar" in plot_type else theta[:, 0]
        y = v
        ax.plot(x, y, **kwargs)
        if ax_label and "rect" in plot_type:
            ax.set_xlabel("Elevation (deg)")
            ax.set_xlim([-90, 90])

    elif plot_type == "2d":
        im = ax.imshow(
            np.flipud(v),
            extent=[phi.min(), phi.max(), theta.min(), theta.max()],
            aspect="auto",
            **kwargs
        )
        if ax_label:
            ax.set_xlabel("Azimuth (deg)")
            ax.set_ylabel("Elevation (deg)")

    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'")

    return phi, theta, v, ax, im



# ---------------========== Antenna Elements ==========--------------- #

class ElementBase(ABC):
    """
    Base class for antennas elements.
    """

    @abstractclassmethod
    def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        """
        # raise NotImplementedError("Subclasses must implement `response`")
        pass
    

    def compute_mean_gain(self, n_samples: int=100, seed: int=42) -> float:
        """
        Compute the mean antenna gain (dBi). For an ideal lossless 
        antenna, this method should be approximately around 0 dBi 
        """
        rng = np.random.default_rng(seed)
        phi = np.uniform(-180.0, 180.0, n_samples)
        theta = np.uniform(-90, 90, n_samples)

        weights = np.sin(np.deg2rad(theta))
        linear_gain = np.power(10.0, 0.1 * self.response(phi, theta))
        mean_linear_gain = np.average(linear_gain, weights=weights)

        return 10.0 * np.log10(mean_linear_gain)
    

    def plot_pattern(self, **kwargs):
        """
        """
        return plot_pattern(self.response, **kwargs)




class ElementIsotropic(ElementBase):
    """
    Isotropic antenna element model 0 dBi uniform gain. Special antenna 
    element representing an ideal radiator that emits energy equally in 
    all directions.
    """
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
            Initialize Element - 3GPP Instance
        """
        # Call the constructor `ElementBase` object
        super().__init__()

        # Define parameters for the 3GPP object.
        self.phi_0, self.theta_0 = float(phi_0), float(theta_0)
        self.phi_bw, self.theta_bw = float(phi_beamwidth), float(theta_beamwidth)

        self.slav, self.am, self.max_gain = 30.0, 30.0, 0.0
        self.calibrate()
    

    def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute gain (dBi) for a given azimuth and elevation
        """
        phi = np.asarray(phi, dtype=float)
        theta = np.asarray(theta, dtype=float)

        # Coordinate rotation
        if self.theta_0 or self.phi_0:
            phi_1, theta_1 = sub_angles(self.phi_0, 90-self.theta_0, phi, 90-theta)
        else:
            phi_1, theta_1 = phi, theta
        
        # Wrap azimuth to [-180, 180)
        phi_1 = ((phi_1 + 180) % 360) - 180

        # Elevation and azimuth attenuations
        av = -np.minimum(12 * (theta_1 / self.theta_beamwidth) ** 2, self.slav)
        ah = -np.minimum(12 * (phi_1 / self.phi_beamwidth) ** 2, self.am)

        # Combined pattern
        return self.max_gain - np.minimum(-(av + ah), self.am)
    

    def calibrate(self, n_samples: int = 10_000, seed: int = 42) -> None:
        """
        Calibrate `max_gain` to achieve approximately 0 dBi mean power.
        """
        self.max_gain -= self.compute_mean_gain(n_samples, seed)
