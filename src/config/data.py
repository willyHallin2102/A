"""
    src/config/data.py
    ------------------
    Contains immutable configurations of environmental
    and simulation parameters. Provides a safe access 
    to constants used across multiple modules.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Final



# ----------======= Angle Indexing =======---------- #
# Common access AoA/AoD

class AngleIndex:
    """
    Index mapping for `Angle-of-Arrival (AoD)` and 
    `Angle-of-Departure (AoD)`.
    """
    AOA_PHI     : Final[int]=0
    AOA_THETA   : Final[int]=1
    AOD_PHI     : Final[int]=2
    AOD_THETA   : Final[int]=3

    N_ANGLES    : Final[int]=4


# ----------======= Link State =======---------- #
# Indexing communication link status

class LinkState:
    """
    Enumeration of the different link states any
    communication link possible, those are 
    `Non-line-of-sight (NLoS)`, `Line-of-Sight (LoS)`
    or if non available `NO_LINK`. 
    """
    NO_LINK     : Final[int]=0
    NLOS        : Final[int]=1
    LOS         : Final[int]=2

    N_STATES    : Final[int]=3


# ----------======= Data Configurations =======---------- #
# Environmental parameter definitions

@dataclass(frozen=True, slots=True)
class DataConfig:
    """
    Immutable configuration for simulation and data parameters.
    Altering any environmental parameters is required to be made 
    within this object directly.
    """
    frequency       : float=28e9
    data_created    : str = field(default_factory=lambda: dt.datetime.now().isoformat())
    description     : str = "dataset"
    rx_types        : tuple[str, ...] = ("Rx0", "Rx1")
    max_path_loss   : float = 200.0
    tx_power_dbm    : float = 16.0
    n_max_paths     : int   = 20
    n_unit_links    : tuple[int, ...] = (50, 25, 10)
    add_zero_los_frac   : float = 0.1
    n_dimensions    : int   = 3
    dropout_rate    : float = 0.2 # Convenience

