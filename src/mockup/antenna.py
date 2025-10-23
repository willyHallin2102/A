"""
    src/mockup/antenna.py
    ---------------------
"""
import numpy as np
import matplotlib.pyplot as plt

from src.math.coords import add_angles, sub_angles
from typing import Callable, Optional, Tuple, Union


def plot_pattern(
    pattern_fn: Callable, 
    theta: Optional[np.ndarray]=None, phi: Optional[np.ndarray]=None,
    n_theta: Optional[int]=None, n_phi: Optional[int]=None,
    plot_type: str="rect_phi", ax: Optional[plt.Axes]=None, ax_label: bool=True,
    **kwargs
) -> Tuple:
    """
    Compute and optionally plot antenna pattern with optimized operations.
    """
    # Optimized grid creation
    phi = np.linspace(-180, 180, n_phi) if n_phi else np.atleast_1d(phi)
    theta = np.linspace(-90, 90, n_theta) if n_theta else np.atleast_1d(theta)

    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='xy')
    v = pattern_fn(phi_grid, theta_grid)

    if plot_type == 'none': return phi_grid, theta_grid, v

    # Optimized plotting logic
    ax = ax or (plt.axes(projection='polar') if 'polar' in plot_type else plt.gca())
    im = None

    plot_config = {
        'phi': (phi_grid[0] if 'rect' in plot_type else np.radians(phi_grid[0]),v.T,'Azimuth (deg)',[-180,180]),
        'theta': (theta_grid[:,0] if 'rect' in plot_type else np.radians(theta_grid[:,0]),v,'Elevation (deg)',[-90,90])
    }

    if plot_type.endswith(('phi', 'theta')):
        key = 'phi' if plot_type.endswith('phi') else 'theta'
        x, y, xlabel, xlim = plot_config[key]
        ax.plot(x, y, **kwargs)
        if ax_label and 'rect' in plot_type:
            ax.set_xlabel(xlabel) 
            ax.set_xlim(xlim)
    
    elif plot_type == '2d':
        im = ax.imshow(
            np.flipud(v), extent=[phi.min(),phi.max(),theta.min(),theta.max()],
            aspect='auto', **kwargs
        )
        if ax_label:
            ax.set_xlabel('Azimuth (deg)')
            ax.set_ylabel('Elevation (deg)')
    else:
        raise ValueError(f"Unknown plot_type`: '{plot_type}'")
    
    return phi_grid, theta_grid, v, ax, im



class ElementBase:
    """
    """
    def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Response method not implemented")
    

    def compute_mean_gain(self, n_samples: int=1000, seed: int=42) -> float:
        """
        """
        rng = np.random.default_rng(seed)
        
        phi = rng.uniform(-180, 180, n_samples)
        theta = rng.uniform(-90, 90, n_samples)

        # Vectorized weight and gain computation
        weights = np.sin(np.deg2rad(theta))
        linear_gain = np.power(10.0, 1.0 * self.response(phi, theta))
        linear_mean_gain = np.average(linear_gain, weights=weights)

        return 10.0 * np.log10(linear_mean_gain)
    

    def plot_pattern(self, **kwargs):
        """
        """
        return plot_pattern(self.response, **kwargs)



class ElementIsotropic(ElementBase):
    """
    """
    def __init__(self): super().__init__()

    def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        """
        return np.zeros_like(np.asarray(phi), dtype=float)



class Element3GPP(ElementBase):
    """
    """
    def __init__(self, 
        phi_0: float=0.0, theta_0: float=0.0, 
        phi_beamwidth: float=120.0, theta_beamwidth: float=65.0
    ):
        # Call constructor
        super().__init__()
        
        # define parameter features
        self.phi_0, self.theta_0 = float(phi_0), float(theta_0)
        self.phi_bw, self.theta_bw = float(phi_beamwidth), float(theta_beamwidth)

        self.slav, self.am = 30.0, 30.0

        # Precompute constants, 
        self._phi_bw2, self._theta_bw2 = self.phi_bw ** 2. self.theta_bw ** 2
        self.max_gain = 0.0
        self.calibrate()
    

    def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        """
        phi, theta = np.asarray(phi, dtype=float), np.asarray(theta, dtype=float)

        # Vectorized coordinate transform
        if self.theta_0 != 0.0 or self.phi_0 != 0.0:
            phi_1, theta_1 = sub_angles(
                self.phi_0, 90.0-self.theta_0, phi, 90.0 - theta
            )
        else:
            phi_1, theta_1 = phi, theta
        
        # Optimized wrapping
        phi_1 = ((phi_1 + 180) % 360) - 180

        # Vectorized pattern computation
        av = np.zeros_like(theta_1)
        if self.theta_bw > 0:
            av = -np.minimum(12.0 * ((theta_1 / self.theta_bw) ** 2), self.slav)
        
        ah = np.zeros_like(phi_1)
        if self.phi_bw > 0:
            ah = -np.minimum(12.0 * ((phi_1 / self.phi_bw) ** 2), self.am)
        
        return self.max_gain - np.minimum(-(av + ah), self.am)
