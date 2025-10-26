"""
    src/mockup/array.py
    -------------------
"""
import numpy as np
from abc import ABC, abstractclassmethod
from typing import Optional, Tuple, Union

from src.config.const import LIGHT_SPEED
from src.math.coords import spherical_to_cartesian, add_angles, sub_angles
from src.mockup.antenna import ElementBase, ElementIsotropic, plot_pattern




@dataclass
class ArrayConfig:
    """Configuration container for array parameters."""
    frequency: float = 28e9
    element: Optional[ElementBase] = None



class ArrayBase(ABC):
    """
    Base class for an antenna array. Represents some arbitrary geometry composed
    of one or more antenna elements e.g., isotropic (ideal radiator) or a 
    3GPP antenna object.
    """
    def __init__(self,
        element: Optional[ElementBase]=None, frequency: float=28e9,
        element_position: Optional[np.ndarray]=None
    ):
        """
            Initialize Array Base Instance
        """
        self.element = element if element is not None else ElementIsotropic()
        self.frequency = float(frequency)
        self.element_position = np.array(
            self.element_position if element_position is not None else [[0.0, 0.0, 0.0]],
            dtype=float,
        )
    

    # Optionally make this abstract if geometry must be defined by subclass
    # @abstractmethod
    # def __init__(...): ...

    def steering_vectors(self,
        phi: Union[float, np.ndarray], theta: Union[float, np.ndarray],
        include_element: bool=True, return_element_gain: bool=False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the array steering vectors for the given direction of the
        antenna array. 

        Args:
        -----
            phi, theta: azimuth and elevation angles (degrees)
            include_element:    Whether to include element pattern gain
            return_element_gain:    If `True`, return both steering vectors
                                    as well as the element gains.
        Returns:
        --------
            Either returns steering vector or if returning gains it returns
            the steering vectors as well element gains.
        """
        """Compute the array steering vectors for given directions."""
        phi_arr = np.atleast_1d(np.asarray(phi, dtype=np.float32))
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float32))

        if phi_arr.shape != theta_arr.shape:
            raise ValueError("phi and theta must have identical shapes.")

        # Convert spherical to Cartesian unit vectors
        inclination = 90.0 - theta_arr
        units = spherical_to_cartesian(1.0, phi_arr, inclination)

        # Compute normalized delay (phase progression)
        delay = (units @ self.element_position.T) * self._inv_wavelength

        # Compute steering vectors (complex exponentials)
        sv = np.exp(1j * 2.0 * np.pi * delay)

        # Optionally include element gain
        if include_element:
            element_dB_gain = self.element.response(phi_arr, theta_arr)
 
            # Correct conversion: gain(dB) → linear amplitude (sqrt of power ratio)
            sv *= np.sqrt(np.power(10.0, 0.1 * element_dB_gain))[:, None]
        else:
            element_dB_gain = np.zeros_like(phi_arr)

        return (sv, element_dB_gain) if return_element_gain else sv
    


    def conjugate_beamforming(
        self, phi: Union[float, np.ndarray], theta: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Compute normalized conjugate beamforming (matched filter) weights."""
        sv = self.steering_vectors(phi, theta, include_element=False)
        norm = np.linalg.norm(sv, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (np.conj(sv) / norm).squeeze()
    

    def plot_pattern(
        self, weights: np.ndarray, include_element: bool = True, **kwargs
    ):
        """Plot array radiation pattern for given beamforming weights."""

        def pattern_fn(phi, theta):
            sv = self.steering_vectors(phi, theta, include_element=include_element)
            pattern = np.abs(sv @ weights)
            return 20.0 * np.log10(np.maximum(pattern, 1e-12))

        return plot_pattern(pattern_fn, **kwargs)



# ---------------========== URA ==========--------------- #

class UniformRectangularArray(ArrayBase):
    """Optimized Uniform Rectangular Array (URA)."""

    def __init__(self,
        n_antennas: Tuple[int, int], separator: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if len(n_antennas) != 2:
            raise ValueError("n_antennas must be a 2-element tuple: (ny, nz)")

        ny, nz = n_antennas
        super().__init__(**kwargs)

        wavelength = LIGHT_SPEED / self.frequency
        separator = (
            np.array([0.5 * wavelength, 0.5 * wavelength], dtype=np.float32)
            if separator is None
            else np.asarray(separator, dtype=np.float32)
        )

        if separator.shape != (2,):
            raise ValueError("separator must be a 2-element array: (dy, dz)")

        dy, dz = separator

        # Efficient memory layout: vectorized antenna coordinates
        y_idx, z_idx = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
        positions = np.column_stack(
            (np.zeros(ny * nz, dtype=np.float32),
             (y_idx.ravel() * dy).astype(np.float32),
             (z_idx.ravel() * dz).astype(np.float32))
        )

        self.element_position = positions
        self.n_antennas = (ny, nz)
        self.separator = separator



# ---------------========== Rotated Array Wrapper ==========--------------- #

class RotatedArray(ArrayBase):
    """
    Wraps an existing `ArrayBase` instance and applies a 3D rotation.
    """

    def __init__(self, array: ArrayBase, phi_0: float = 0.0, theta_0: float = 0.0):
        if not isinstance(array, ArrayBase):
            raise TypeError("`array` must be an `ArrayBase` instance.")
        super().__init__(
            element=array.element,
            frequency=array.frequency,
            element_position=array.element_position.copy(),
        )
        self._base_array = array
        self.phi_0 = float(phi_0)
        self.theta_0 = float(theta_0)

    def global_to_local(self, 
        phi: np.ndarray, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert global spherical angles to local (rotated) coordinates."""
        return sub_angles(phi, 90.0 - theta, self.phi_0, 90.0 - self.theta_0)

    def steering_vectors(self, phi: np.ndarray, theta: np.ndarray, **kwargs):
        phi_local, theta_local = self.global_to_local(phi, theta)
        return self._base_array.steering_vectors(phi_local, theta_local, **kwargs)

    def conjugate_beamforming(self, phi: np.ndarray, theta: np.ndarray):
        phi_local, theta_local = self.global_to_local(phi, theta)
        return self._base_array.conjugate_beamforming(phi_local, theta_local)

