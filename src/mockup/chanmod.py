"""
    src/mockup/chanmod.py
    ---------------------

"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple, Union

from src.config.data import LinkState
from src.mockup.array import ArrayBase


# ---------------========== MultiPath Channel ==========--------------- #

class MultiPathMChannel:
    """
    """
    N_ANGLES:   int = 4

    AOA_PHI:    int = 0
    AOA_THETA:  int = 1
    AOD_PHI:    int = 2
    AOD_THETA:  int = 3

    ANGLE_NAMES:    List[str]=[
        'AoA_phi', 'AoA_theta', 'AoD_phi', 'AoD_theta'
    ]
    PATH_LOSS:  float=250.0


    def __init__(self):
        """
            Initialize an empty channel instance.
        """
        self.path_loss: np.ndarray = np.empty(0, dtype=np.float32)
        self.delays: np.ndarray = np.empty(0, dtype=np.float32)
        self.angles: np.ndarray = np.empty((0, MultiPathMChannel.N_ANGLES), dtype=np.float32)

        self.link_state: LinkState = LinkState.NO_LINK
    

    def compute_omni_path_loss(self) -> float:
        """
        """
        if self.link_state == LinkState.NO_LINK or self.path_loss.size == 0:
            return np.inf
        
        # Compute minimum path-loss present, then compute then relative 
        # linear weights and sum in linear domain.
        min_path_loss = np.min(self.path_loss)
        sum_path_loss = np.sum(np.power(10.0, -0.1 * (self.path_loss - min_path_loss)))
        return min_path_loss - 10.0 * np.log10(sum_path_loss)
    

    def compute_rms_delay(self) -> float:
        """
        """
        if self.link_state == LinkState.NO_LINK or self.path_loss.size == 0:
            return 0.0
        
        min_path_loss = np.min(self.path_loss)
        weights = np.power(10.0, -0.1 * (self.path_loss - min_path_loss))
        weights /= np.sum(weights)

        mean_delay = np.sum(weights * self.delays)
        return np.sqrt(np.sum(weights * np.power(self.delays - mean_delay, 2)))
    

# ---------------========== Directional Path Loss ==========--------------- #

def _get_angles(channel: MultiPathMChannel) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    """
    aod_theta = 90 - channel.angles[:, MultiPathMChannel.AOD_THETA]
    aod_phi = channel.angles[:, MultiPathMChannel.AOD_PHI]
    aoa_theta = 90 - channel.angles[:, MultiPathMChannel.AOA_THETA]
    aoa_phi = channel.angles[:, MultiPathMChannel.AOA_PHI]

    return aod_phi, aod_theta, aoa_phi, aoa_theta


def directional_path_loss(
    tx_array: ArrayBase, rx_array: ArrayBase, channel: MultiPathMChannel,
    *, return_element_gain: bool=True, return_beamforming_gain: bool=False
) -> Union[float, Tuple]:
    """
    """
    if channel.link_state == LinkState.NO_LINK or channel.path_loss.size == 0:
        path_loss_effective = MultiPathMChannel.PATH_LOSS
        return (path_loss_effective,) if (return_element_gain or return_beamforming_gain) else path_loss_effective
    
    # Angles
    aod_phi, aod_theta, aoa_phi, aoa_theta = _get_angles(channel)

    # Steering vectors and element gains
    tx_steering_vectors, tx_element_gain = tx_array.steering_vectors(aod_phi, aod_theta, return_element_gain=True)
    rx_steering_vectors, rx_element_gain = rx_array.steering_vectors(aoa_phi, aoa_theta, return_element_gain=True)

    # Path loss adjusted for element gains
    path_loss_element = channel.path_loss - tx_element_gain - rx_element_gain
    im = np.argmin(path_loss_element)

    # Beamforming vectors 
    tx_weights = np.conj(tx_steering_vectors[im]) / np.linalg.norm(tx_steering_vectors[im])
    rx_weights = np.conj(rx_steering_vectors[im]) / np.linalg.norm(rx_steering_vectors[im])

    # Array beamforming gains measured in [dB]
    tx_beamforming = 20.0 * np.log10(np.abs(tx_steering_vectors @ tx_weights))
    rx_beamforming = 20.0 * np.log10(np.abs(rx_steering_vectors @ rx_weights))

    # Effective per-path loss
    beamforming_path_loss = channel.path_loss - tx_beamforming - rx_beamforming
    min_path_loss = np.min(beamforming_path_loss)
    path_loss_effective = min_path_loss - 10.0 * np.log10(
        np.sum(np.power(10.0, -0.1 * (beamforming_path_loss - min_path_loss)))
    )

    if not (return_element_gain or return_beamforming_gain):
        return path_loss_effective
    
    results: List[np.ndarray | float] = [path_loss_effective]
    if return_element_gain:
        results += [tx_element_gain, rx_element_gain]
    if return_beamforming_gain:
        results += [tx_beamforming-tx_element_gain, rx_beamforming-rx_element_gain]

    return tuple(results)


# ------------======== Multi-Sector Directional Path Loss ========------------ #

def directional_path_loss_multi_sector(
    tx_array_list: List[ArrayBase], rx_array_list: List[ArrayBase],
    channel: MultiPathMChannel, *, return_element_gain: bool=True,
    return_beamforming_gain: bool=True, return_array_indices: bool=True
) -> Union[float, Tuple]:
    """
    """
    if channel.link_state == LinkState.NO_LINK or channel.path_loss.size == 0:
        path_loss_effective = MultiPathMChannel.PATH_LOSS
        return (path_loss_effective,) if (return_element_gain or return_beamforming_gain) else path_loss_effective
    
    aod_phi, aod_theta, aoa_phi, aoa_theta = _get_angles(channel)
    best = {
        "min_pathloss": MultiPathMChannel.PATH_LOSS,
        "im": 0, "itx": 0, "irx": 0,
        "tx_steering_vectors": None, "rx_steering_vectors": None,
        "tx_element_gain": None, "rx_element_gain": None,
    }

    # Elevation for all Tx/Rx combinations
    for itx, tx_array in enumerate(tx_array_list):
        tx_steering_vectors, tx_element_gain = tx_array.steering_vectors(
            aod_phi, aod_theta, return_element_gain=True
        )
        for irx, rx_array in enumerate(rx_array_list):
            rx_steering_vectors, rx_element_gain = rx_array.steering_vectors(
                aoa_phi, aoa_theta, return_element_gain=True
            )
            path_loss_element = channel.path_loss - tx_element_gain - rx_element_gain
            im = np.argmin(path_loss_element)
            min_path_loss_i = path_loss_element[im]
            if min_path_loss_i < best["min_pathloss"]:
                best.update(dict(
                    min_pathloss=min_path_loss_i, im=im, itx=itx, irx=irx,
                    tx_steering_vectors=tx_steering_vectors,
                    rx_steering_vectors=rx_steering_vectors,
                    tx_element_gain=tx_element_gain, rx_element_gain=rx_element_gain
                ))
    
    tx_steering_vectors = best["tx_steering_vectors"]
    rx_steering_vectors = best["rx_steering_vectors"]

    tx_element_gain = best["tx_element_gain"]
    rx_element_gain = best["rx_element_gain"]

    im = best["im"]

    # Beamforming
    tx_weights = np.conj(tx_steering_vectors[im])/np.linalg.norm(tx_steering_vectors[im])
    rx_weights = np.conj(rx_steering_vectors[im])/np.linalg.norm(rx_steering_vectors[im])

    # Effective path loss
    tx_beamforming = 20.0 * np.log10(np.abs(tx_steering_vectors @ tx_weights))
    rx_beamforming = 20.0 * np.log10(np.abs(rx_steering_vectors @ rx_weights))

    # Effective path loss
    beamforming_path_loss = channel.path_loss - tx_beamforming - rx_beamforming
    min_path_loss = np.min(beamforming_path_loss)
    path_loss_effective = min_path_loss - 10.0 * np.log10(
        np.sum(np.power(10.0, -0.1 * (beamforming_path_loss-min_path_loss)))
    )

    if not (return_element_gain or return_beamforming_gain or return_array_indices):
        return path_loss_effective

    results: List[np.ndarray | float | int] = [path_loss_effective]
    if return_array_indices:
        results += [best["itx"], best["irx"]]
    if return_element_gain:
        results += [tx_element_gain, rx_element_gain]
    if return_beamforming_gain:
        results += [tx_beamforming - tx_element_gain, rx_beamforming - rx_element_gain]
    return tuple(results)
