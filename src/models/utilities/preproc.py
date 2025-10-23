"""
    src/models/utilities/preproc.py
    -------------------------------
    Serialization / Deserialization of `sklearn.preprocessing` objects,
    allowing storage of their fitted state without required to save the 
    entire objects.
"""
from functools import singledispatch
from typing import Any, Dict, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

import numpy as np

# ---------------========== Type Alias ==========--------------- #

Preproc = Union[StandardScaler, OneHotEncoder, MinMaxScaler]


# ---------------========== Serialization ==========--------------- #

@singledispatch
def preproc_to_param(proc: Any) -> Dict[str, Any]:
    """
    Serialize a fitted scikit-learn preprocessing object into a JSON/pickle dict.

    Parameters
    ----------
    proc:   A fitted preprocessing instance (e.g. `StandardScaler`, `OneHotEncoder`).

    Returns
    -------
        A dictionary representation of the fitted parameters.

    Raises
    ------
    TypeError:  If the provided object type is unsupported.
    """
    raise TypeError(f"Unsupported preprocessing type: {type(proc).__name__}")


@preproc_to_param.register
def _(proc: StandardScaler) -> Dict[str, Any]:
    """ Serialize a fitter `StandardScaler` into a dictionary. """
    return {
        "mean": proc.mean_.tolist(),
        "scale": proc.scale_.tolist(),
        "var": getattr(proc, "var_", (proc.scale_ ** 2)).tolist(),
        "n_samples_seen": getattr(proc, "n_samples_seen_", None),
    }


@preproc_to_param.register
def _(proc: OneHotEncoder) -> Dict[str, Any]:
    """ Serialize a fitted `OneHotEncoder` into a dictionary. """
    params: Dict[str, Any] = {
        "categories": [category.tolist() for category in proc.categories_],
        "handle_unknown": proc.handle_unknown,
        "drop": proc.drop,
        "sparse_output": getattr(proc, "sparse_output", False),
    }
    # Add optional attributes if present
    for attribute in ("min_frequency", "max_categories"):
        if hasattr(proc, attribute):
            params[attribute] = getattr(proc, attribute)
    return params


@preproc_to_param.register
def _(proc: MinMaxScaler) -> Dict[str, Any]:
    """ Serialize a fitted `MinMaxScaler` into a dictionary. """
    return {
        "data_min": proc.data_min_.tolist(),
        "data_max": proc.data_max_.tolist(),
        "data_range": proc.data_range_.tolist(),
        "scale": proc.scale_.tolist(),
        "min": proc.min_.tolist(),
        "feature_range": getattr(proc, "feature_range", (0, 1)),
    }

# ---------------========== Deserialization ==========--------------- #
# Note: `singledispatch` works on the first argument, so we dispatch on class.

@singledispatch
def param_to_preproc(cls: Any, param: Dict[str, Any]) -> Preproc:
    """
    Deserialize a JSON-safe dictionary back into a fitted preprocessing object.

    Args:
    -----
    cls:    The class of the preprocessing object (e.g. `StandardScaler`).
    param:  The serialized parameter dictionary.

    Returns
    -------
        A reconstructed, "fitted" preprocessing instance.

    Raises
    ------
    TypeError:  If the class type is unsupported.
    """
    raise TypeError(f"Unsupported preprocessor class: {cls.__name__}")


# @param_to_preproc.register
# def _(cls: type(StandardScaler), param: Dict[str, Any]) -> StandardScaler:
#     """ Serialize a fitted `StandardScaler` into a dictionary. """
#     proc = StandardScaler()
#     proc.mean_ = np.array(param["mean"])
#     proc.scale_ = np.array(param["scale"])
#     proc.var_ = np.array(param.get("var", np.square(proc.scale_)))
#     proc.n_samples_seen_ = param.get("n_samples_seen")
#     proc.n_features_in_ = proc.mean_.shape[0]
#     return proc


# @param_to_preproc.register
# def _(cls: type(OneHotEncoder), param: Dict[str, Any]) -> OneHotEncoder:
#     """ Serialize a fitted `OneHotEncoder` into a dictionary. """
#     proc = OneHotEncoder(
#         categories=[np.array(category) for category in param["categories"]],
#         handle_unknown=param.get("handle_unknown", "ignore"),
#         drop=param.get("drop"),
#         sparse_output=param.get("sparse_output", False),
#         min_frequency=param.get("min_frequency"),
#         max_categories=param.get("max_categories"),
#     )
#     # Pretend fitted
#     proc.categories_ = [np.array(category) for category in param["categories"]]
#     proc.n_features_in_ = len(proc.categories_)
#     return proc


# @param_to_preproc.register
# def _(cls: type(MinMaxScaler), param: Dict[str, Any]) -> MinMaxScaler:
#     """ Serialize a fitted `MinMaxScaler` into a dictionary. """
#     proc = MinMaxScaler(feature_range=param.get("feature_range", (0, 1)))
#     proc.data_min_ = np.array(param["data_min"])
#     proc.data_max_ = np.array(param["data_max"])
#     proc.data_range_ = np.array(param["data_range"])
#     proc.scale_ = np.array(param["scale"])
#     proc.min_ = np.array(param["min"])
#     proc.n_features_in_ = proc.data_min_.shape[0]
#     return proc

@param_to_preproc.register(StandardScaler)
def _(cls, param: Dict[str, Any]) -> StandardScaler:
    proc = StandardScaler()
    proc.mean_ = np.array(param["mean"])
    proc.scale_ = np.array(param["scale"])
    proc.var_ = np.array(param.get("var", np.square(proc.scale_)))
    proc.n_samples_seen_ = param.get("n_samples_seen")
    proc.n_features_in_ = proc.mean_.shape[0]
    return proc


@param_to_preproc.register(OneHotEncoder)
def _(cls, param: Dict[str, Any]) -> OneHotEncoder:
    proc = OneHotEncoder(
        categories=[np.array(category) for category in param["categories"]],
        handle_unknown=param.get("handle_unknown", "ignore"),
        drop=param.get("drop"),
        sparse_output=param.get("sparse_output", False),
        min_frequency=param.get("min_frequency"),
        max_categories=param.get("max_categories"),
    )
    proc.categories_ = [np.array(category) for category in param["categories"]]
    proc.n_features_in_ = len(proc.categories_)
    return proc


@param_to_preproc.register(MinMaxScaler)
def _(cls, param: Dict[str, Any]) -> MinMaxScaler:
    proc = MinMaxScaler(feature_range=param.get("feature_range", (0, 1)))
    proc.data_min_ = np.array(param["data_min"])
    proc.data_max_ = np.array(param["data_max"])
    proc.data_range_ = np.array(param["data_range"])
    proc.scale_ = np.array(param["scale"])
    proc.min_ = np.array(param["min"])
    proc.n_features_in_ = proc.data_min_.shape[0]
    return proc



# ---------- Public API ---------- #

def serialize_preproc(proc: Preproc) -> Dict[str, Any]:
    """
    Serializing a fitted preprocessor to a dictionary.

    Args:
    -----
    proc:   Fitted preprocessing object (`StandardScaler`, 
            `OneHotEncoder`, or `MinMaxScaler`).

    Returns:
    --------
        A dictionary with keys:
            - "type": name of the preprocessor class
            - "params": serialized fitted parameters
    """
    return {
        "type": type(proc).__name__,
        "params": preproc_to_param(proc),
    }


def deserialize_preproc(data: Dict[str, Any]) -> Preproc:
    """
    Reconstructing a preprocessor from its serialized dictionary form.

    Args:
    -----
    data:   Dictionary containing the preprocessor metadata and parameters.
    Returns:
    -------
        Reconstructed fitted preprocessing object.

    Raises:
    ------
    ValueError: If the `type` key does not correspond to a supported 
                preprocessor.
    """
    cls_map = {
        "StandardScaler": StandardScaler,
        "OneHotEncoder": OneHotEncoder,
        "MinMaxScaler": MinMaxScaler,
    }

    cls = cls_map.get(data["type"])
    if cls is None:
        raise ValueError(f"Unknown preprocessor type: {data['type']}")
    return param_to_preproc(cls(), data["params"])
