"""
    src/config/config.py
    --------------------
    Constants stored in this file.
"""
from typing import Final

# ----------===== Physical Constants =====---------- #

LIGHT_SPEED         : Final[float]  = 2.99792458e8   # m/s
THERMAL_NOISE       : Final[float]  = -174.0         # dBm

# ----------===== File Constants =====---------- #

PREPROC_FN          : Final[str]    = "preproc.pkl"
WEIGHTS_FN          : Final[str]    = "model.weights.h5"
CONFIG_FN           : Final[str]    = "model_config.json"