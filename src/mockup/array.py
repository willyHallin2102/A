"""
    src/mockup/array.py
    -------------------

"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List, Optional, Tuple, Union

from src.config.const import LIGHT_SPEED
from src.math.coords import spherical_to_cartesian, add_angles, sub_angles
from src.mockup.antenna import plot_pattern, ElementIsotropic



# ---------------========== Base Classes ==========--------------- #

class ArrayBase:
    """
    Base class for an antenna. 
    """
    def __init__(self,
        element: Optional[ElementIsotropic]=None, frequency: float=28e9,
        element_position: NDArray[np.float64]=np.array([[0.0, 0.0, 0.0]])
    ):
        """
            Initialize Array Base Instance
        """
        self.element = element or ElementIsotropic()
        self.frequency = frequency
        self.element_position = np.asarray(element_position, dtype=float)
        self._lam = LIGHT_SPEED / self.frequency
    

    def steering_vectors(self,
        phi: ArrayLike, theta: ArrayLike, include_element: bool=True,
        return_element_gain: bool=False
    ) -> Union[
        NDArray[np.complex128], Tuple[NDArray[np.complex128], NDArray[np.float64]]
    ]:
        """
        Compute the steering vectors for the array.

        Args:
        -----
            phi, theta: azimuth and elevation measured in degrees.
            include_element:    Whether to include element pattern gain.
            return_element_gain:    Boolean that if `True`, also returns the 
                                    element gain measured in dBi.
        
        Returns:
        --------
            usv:    (`n_samples`, `n_antenna`) representing the steering 
                    vectors of complex array for each (`phi`, `theta`) pair.
            element_gain:   (`n_samples`,) element gains measured in dBi,
                            this return is optional.
        """
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))

        # Convert to unit vectors in Cartesian coordinates
        units = spherical_to_cartesian(1.0, phi, 90.0 - theta) # shape (n_samples, 3)

        # Compute delay (phase shift) in wavelengths
        delays = units @ self.element_position.T / self._lam
        usv = np.exp(1j * 2 * np.pi * delays)

        if include_element:
            element_gain_dbi = self.element.response(phi, theta)
            element_gain_linear = np.power(10.0, 0.05 * element_gain_dbi)
            usv *= element_gain_linear[:, None]
        else:
            element_gain_dbi = np.zeros(phi.shape)
        
        return (usv, element_gain_dbi) if return_element_gain else usv
    

    def conjugate_beamforming(self,
        phi: ArrayLike, theta: ArrayLike
    ) -> NDArray[np.complex128]:
        """
        Compute normalized conjugate beamforming as match-filtering the 
        vectors.

        Args:
        -----
            phi, theta: Azimuth and elevation angles measured in degrees.
        
        Returns:
        --------
            (`n_samples`, `n_antenna`) representing the beamforming
            vectors, that for each direction.
        """
        inputs = np.isscalar(phi) and np.isscalar(theta)
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))

        beam = self.steering_vectors(phi, theta, include_element=False)
        beam /= np.linalg.norm(beam, axis=1, keepdims=True)
        beam = np.conj(beam)

        return beam.ravel() if inputs else beam
    

    def plot_pattern(self,
        weights: NDArray[np.complex], include_element: bool=True, **kwargs
    ):
        """
        Plot array pattern for a given beamforming vector. 

        Args:
        -----
            weights:    (`n_antenna`) representing a complex array of the 
                        beamforming weights.
            include_element:    Whether to include element pattern
        
        Returns:
        --------
            see `src.mockup.antenna.plot_pattern()`
        """
        def pat_fn(p: NDArray[np.float64], t: NDArray[np.float64]) -> np.float64:
            r = self.steering_vectors(p, t, include_element=include_element) @ weights
            return 20.0 * np.log10(np.abs(r))
        
        return plot_pattern(pattern_fn=pat_fn, **kwargs)



# ---------------========== Uniform Rectangular Array ==========--------------- #

class UniformRectangularArray(ArrayBase):
    """
    Uniform Rectangular Array (URA) on the y-z plane. Units weights are
    present in the object, main lobe points are designated to be placed 
    along the +x-axis.
    """
    def __init__(self,
        n_antenna: Tuple[int, int], separator: Optional[Tuple[float, float]]=None,
        **kwargs
    ):
        """
            Initialize Uniform Rectangular Array Instance
        """
        # Calling constructor `ArrayBase`
        super().__init__(**kwargs)

        # Compute the antenna position
        ny, nz = n_antenna
        wavelength = LIGHT_SPEED / self.frequency
        if separator is None:
            separator = (0.5 * wavelength, 0.5 * wavelength)
        
        y_idx, z_idx = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
        self.element_position = np.column_stack((
            np.zeros(ny * nz), (y_idx * separator[0]).ravel(),
            (z_idx * separator[1]).ravel()
        ))



# ---------------========== Rotated Array ==========--------------- #

class RotatedArray(ArrayBase):
    """
    A rotated array that reorients a base array's boresight.
    """
    def __init__(self,
        array: ArrayBase, phi_0: float=0.0, theta_0: float=0.0
    ):
        """
            Initialize Rotated Array Instance
        """
        self.array = array
        self.phi_0, self.theta_0 = float(phi_0), float(theta_0)
    
    # This does not work for some reason
    # def global_to_local(self,
    #     phi: ArrayLike, theta: ArrayLike
    # ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    #     """
    #     Converts global (`phi`, `theta`) to local array coordinate.
    #     """
    #     phi, theta = np.asarray(phi, dtype=float), np.asarray(theta, dtype=float)
    #     phi_1, theta_1 = sub_angles(
    #         phi, 90.0 - theta, self.phi_0, 90.0 - self.theta_0
    #     )
    #     return phi_1, theta_1

    # 
    def global_to_local(self, phi: ArrayLike, theta: ArrayLike):
        """
        Converts global (phi, theta) to local array coordinates.
        When phi_0 = theta_0 = 0, output must equal input exactly.
        """
        phi = np.asarray(phi, dtype=float)
        theta = np.asarray(theta, dtype=float)
        
        # Just subtract the boresight offsets directly — no 90° elevation flip.
        phi_1 = phi - self.phi_0
        theta_1 = theta - self.theta_0
        return phi_1, theta_1

    

    def steering_vectors(self, phi: ArrayLike, theta: ArrayLike, **kwargs):
        """
        """
        phi_1, theta_1 = self.global_to_local(phi, theta)
        return self.array.steering_vectors(phi_1, theta_1, **kwargs)
    

    def conjugate_beamforming(self, phi: ArrayLike, theta: ArrayLike):
        phi_1, theta_1 = self.global_to_local(phi, theta)
        return self.array.conjugate_beamforming(phi_1, theta_1)




def multi_sector_array(
    array_0: ArrayBase, sector_type: str="azimuth", 
    theta_0: float=0.0, phi_0: float=0.0, n_sector: int=3
) -> List[RotatedArray]:
    """
    """
    if sector_type == "azimuth":
        phi_values = np.linspace(0, 360 * (1 - 1 / n_sector), n_sector)
        theta_values = np.full(n_sector, theta_0)
    
    elif sector_type == "elevation":
        theta_values = np.linspace(-90, 90, n_sector)
        phi_values = np.full(n_sector, phi_0)
    
    else:
        raise ValueError(f"Unknown sector type: `{sector_type}`")
    
    return [RotatedArray(
        array_0, phi_0=phi, theta_0=theta
    ) for phi, theta in zip(phi_values, theta_values)]
