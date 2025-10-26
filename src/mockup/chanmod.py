"""
    src/mockup/channel.py
    ---------------------
    Multipath channel model mockup including a omni and directional path loss
    calculations, and multi-array evaluation.
"""
import numpy as np
from typing import Dict, Final, List, Tuple, Union

from src.config.data import LinkState
from src.mockup.array import ArrayBase


class MultiPathChannel:
    """
    Multi-path channel abstraction for link-level simulations. This class
    is used with the purpose of enact the receiving antennas for a simulated 
    UAV flight trajectory.
    """
    N_ANGLES: Final[int] = 4
    ANGLE_INDICES: Final[Dict[str, int]] = {
        "aoa_phi": 0, "aoa_theta": 1, "aod_phi": 2, "aod_theta": 3
    }
    LARGE_PATH_LOSS: Final[float] = 250.0


    def __init__(self):
        """
            Initialize Multi-Path Channel Instance.
        """
        self.path_loss = np.zeros(0, dtype=np.float32)
        self.delays = np.zeros(0, dtype=np.float32)
        self.angles = np.zeros((0, self.N_ANGLES), dtype=np.float32)
        self.link_state = LinkState.NO_LINK
    

    # ------------======== Omni-Directional Computations ========------------ #

    def compute_omni_path_loss(self) -> float:
        """
        Compute omni-directional effective path loss (dB) 
        """
        if self.link_state == LinkState.NO_LINK or len(self.path_loss) == 0:
            return np.inf
        return _compute_effective_path_loss(self.path_loss)
    

    def rms_delays(self) -> float:
        """
        Compute RMS delay spread (seconds).
        """
        if self.link_state == LinkState.NO_LINK or len(self.delays) == 0:
            return 0.0
        return _compute_rms_delays(self.delays, self.path_loss)
    


# ------------======== Directional Path Loss Computations ========------------ #

def directional_path_loss(
    tx_array: ArrayBase, rx_array: ArrayBase, channel: MultiPathChannel,
    return_element_gain: bool=True, return_beamforming_gain: bool=True
) -> Union[float, List]:
    """
    Compute the effective directional path loss between the receiver antenna
    array (Rx) and the transmitting object array (Tx).

    Args:
    -----
        tx_array, rx_array: transmit and receive array objects.
        channel:    Channel model containing multipath data.
        return_element_gain:    Whether to return per-element gain values
        return_beamforming_gain:    Whether to return beamforming gains.
    
    Returns:
    -------
        Union[float, List]
            If `return_element_gain` and/or `return_beamforming_gain` are True,
            returns a list: 
            [
                effective_pathloss, tx_element_gain, rx_element_gain, 
                tx_bf_gain, rx_bf_gain
            ]
            Otherwise, returns a single float (effective path loss in dB).
    """
    if channel.link_state == LinkState.NO_LINK:
        return _handle_no_link(return_element_gain, return_beamforming_gain)
    
    angles = channel.angles
    aoa_phi, aoa_theta = angles[:, 0], 90 - angles[:, 1]
    aod_phi, aod_theta = angles[:, 2], 90 - angles[:, 3]

    # Vectorized steering vector computations
    rx_sv, rx_element_gain = rx_array.steering_vectors(aoa_phi, aoa_theta, return_element_gain=True)
    tx_sv, tx_element_gain = tx_array.steering_vectors(aod_phi, aod_theta, return_element_gain=True)

    # Element - Level path loss
    best_idx = np.argmin(channel.path_loss - tx_element_gain - rx_element_gain)

    # Beamforming weights
    rx_weights = _normalize_vector(np.conj(rx_sv[best_idx]))
    tx_weights = _normalize_vector(np.conj(tx_sv[best_idx]))

    # Beamforming gains
    rx_bf_gain = _compute_beamforming_gain(rx_sv, rx_weights, rx_element_gain)
    tx_bf_gain = _compute_beamforming_gain(tx_sv, tx_weights, tx_element_gain)

    # Effective directional path loss
    path_loss_bf = channel.path_loss - tx_bf_gain - rx_bf_gain
    effective_pl = _compute_effective_path_loss(path_loss_bf)

    return _format_output(
        effective_pl, tx_element_gain, rx_element_gain, 
        tx_bf_gain, rx_bf_gain,
        return_element_gain, return_beamforming_gain
    )



def directional_path_loss_multi_array(
    tx_array_list: List[ArrayBase], rx_array_list: List[ArrayBase],
    channel: MultiPathChannel, return_element_gain: bool=True,
    return_beamforming_gain: bool=True, return_array_indices: bool=True
) -> Union[float, List]:
    """
    Compute the minimum effective directional path loss across multiple
    Tx/Rx array pairs (multi-antenna sites).

    Returns:
    --------
        [
            effective_pathloss, tx_idx, rx_idx, 
            tx_element_gain, rx_element_gain,
            tx_bf_gain, rx_bf_gain
         ] if outputs requested.
    """
    if channel.link_state == LinkState.NO_LINK:
        return _handle_no_link(
            return_element_gain or return_array_indices, return_beamforming_gain
        )
    
    angles = channel.angles
    aoa_phi, aoa_theta = angles[:, 0], 90 - angles[:, 1]
    aod_phi, aod_theta = angles[:, 2], 90 - angles[:, 3]

    best = {
        "path_loss": MultiPathChannel.LARGE_PATH_LOSS,
        "tx_idx": -1, "rx_idx": -1,
    }

    for irx, rx_array in enumerate(rx_array_list):
        for itx, tx_array in enumerate(tx_array_list):
            rx_sv, rx_element = rx_array.steering_vectors(aoa_phi, aoa_theta, return_element_gain=True)
            tx_sv, tx_element = tx_array.steering_vectors(aod_phi, aod_theta, return_element_gain=True)

            element_path_loss = channel.path_loss - tx_element - tx_element
            idx = np.argmin(element_path_loss)
            current_min = element_path_loss[idx]

            if current_min < best["path_loss"]:
                best.update({
                    "path_loss": current_min, "tx_idx": itx, "rx_idx": irx,
                    "idx": idx,
                    "tx_sv": tx_sv, "rx_sv": rx_sv,
                    "tx_element": tx_element, "rx_element": rx_element
                })
    
    # Compute beamforming on best pair
    tx_weights = _normalize_vector(np.conj(best["tx_sv"][best["idx"]]))
    rx_weights = _normalize_vector(np.conj(best["rx_sv"][best["idx"]]))

    tx_bf_gain = _compute_beamforming_gain(best["tx_sv"], tx_weights, best["tx_element"])
    rx_bf_gain = _compute_beamforming_gain(best["rx_sv"], rx_weights, best["rx_element"])

    path_loss_bf = channel.pathloss - tx_bf_gain - rx_bf_gain
    effective_pl = _compute_effective_pathloss(path_loss_bf)

    return _format_multi_output(
        effective_pl, best, tx_bf_gain, rx_bf_gain,
        return_array_indices, return_element_gain, return_beamforming_gain
    )





# ===============================================================
# Helper Functions (pure NumPy, vectorized)
# ===============================================================

def _compute_effective_pathloss(path_loss: np.ndarray) -> float:
    """Stable computation of effective path loss using log-sum-exp trick."""
    if len(path_loss) == 0:
        return np.inf
    min_pl = np.min(path_loss)
    exponent = -0.1 * (path_loss - min_pl)
    max_exp = np.max(exponent)
    linear_sum = np.sum(np.exp(exponent - max_exp))
    return min_pl - 10.0 * (np.log10(linear_sum) + max_exp / np.log(10))


def _compute_rms_delays(delays: np.ndarray, path_loss: np.ndarray) -> float:
    """Compute RMS delay spread from path loss and delays."""
    min_pl = np.min(path_loss)
    weights = np.power(10.0, -0.1 * (path_loss - min_pl))
    
    s = np.sum(weights)
    if s == 0: return 0.0
    weights /= s
    
    mean_delay = np.dot(weights, delays)
    variance = np.dot(weights, (delays - mean_delay) ** 2)
    
    return np.sqrt(variance)


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize complex vector safely."""
    norm = np.linalg.norm(vec)
    return np.conj(vec) / norm if norm > 0 else vec


def _compute_beamforming_gain(
    steering_vectors: np.ndarray, weights: np.ndarray, element_gain: np.ndarray
) -> np.ndarray:
    """Compute beamforming gain (in dB) excluding element gain."""
    gain_linear = np.abs(steering_vectors @ weights)
    return 20.0 * np.log10(np.maximum(gain_linear, 1e-12)) - element_gain


def _handle_no_link_case(
    return_element_gain: bool, return_beamforming_gain: bool
) -> Union[float, List]:
    """Return consistent default values when no link exists."""
    base_result = MultiPathChannel.LARGE_PATH_LOSS
    if not (return_element_gain or return_beamforming_gain):
        return base_result
    return [base_result, np.array(0), np.array(0), np.array(0), np.array(0)]


def _format_output(
    pathloss: float,
    tx_element_gain: np.ndarray, rx_element_gain: np.ndarray,
    tx_bf_gain: np.ndarray, rx_bf_gain: np.ndarray,
    return_element: bool, return_bf: bool
) -> Union[float, List]:
    """Format return tuple for single-array case."""
    if not (return_element or return_bf):
        return pathloss
    return [pathloss, tx_element_gain, rx_element_gain, tx_bf_gain, rx_bf_gain]


def _format_multi_output(
    path_loss: float, best: dict,
    tx_bf_gain: np.ndarray, rx_bf_gain: np.ndarray,
    return_idx: bool, return_element: bool, return_bf: bool
) -> List:
    """Format return list for multi-array case."""
    result = [path_loss]
    if return_idx:
        result += [best["tx_idx"], best["rx_idx"]]
    if return_element:
        result += [best["tx_element"], best["rx_element"]]
    if return_bf:
        result += [tx_bf_gain, rx_bf_gain]
    return result

