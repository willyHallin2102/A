"""
    src/models/utilities/common.py
    ------------------------------
    Common TensorFlow/Keras utility shared across multiple modules are being
    stored in this script.
"""
from __future__ import annotations

import warnings
from typing import Dict, Literal, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
tfk, tfkl = tf.keras, tf.keras.layers


class SortLayer(tfkl.Layer):
    """
    A custom Keras layer that sorts the first `n_sort` features in each sample
    of the input tensor along the last dimension.

    Example:
    --------
        >>> layer = SortLayer(n_sort=3)
        >>> x = tf.constant([
                [3., 1., 2., 5.], [7., 4., 9., 6.]
            ])
        >>> layer(x)
    Attributes
    ----------
    n_sort : int
        Number of leading elements in each row to sort.
    """
    def __init__(self, n_sort: int, **kwargs):
        """
            Initialize sort-layer instance
        """
        super().__init__(**kwargs)
        if not isinstance(n_sort, int) or n_sort <= 0:
            raise ValueError("`n_sort` must be a positive integer")
        self.n_sort = n_sort
    

    def call(self,
        inputs: tf.Tensor, direction: Literal["ASCENDING", "DESCENDING"]="DESCENDING"
    ) -> tf.Tensor:
        """
        Forward pass: sorts the first `n_sort` elements in each sample.

        Args:
        -----
        inputs: Input tensor of shape `(batch, features)`.
        direction:  Sort direction. Defaults to "DESCENDING".
        
        Returns:
        --------
            Output tensor with first `n_sort` elements sorted per sample.
        """
        inputs = tf.convert_to_tensor(inputs)
        tf.debugging.assert_rank_at_least(
            inputs, 2, message="Sort - layer expects rank >= 2 (batch, features)"
        )
        
        head = tf.sort(inputs[:, :self.n_sort], direction=direction)
        tail = x[:, self.n_sort:]

        return tf.concat([head, tail], axis=-1)
    

    def get_config(self):
        """
            Return layer configuration for serialization.
        """
        config = super().get_config()
        config.update({"n_sort": self.n_sort})
        return config



class SplitSortLayer(tfkl.Layer):
    """
    Similar to `SortLayer`, but uses a fixed descending sort and can be used
    to explicitly split and recombine tensor parts.

    Example:
    --------
        >>> layer = SplitSortLayer(n_sort=2)
        >>> x = tf.constant([[1., 3., 2.],
        ...                  [9., 7., 4.]])
        >>> layer(x)
    Split Sort Layer is essential descending split-layer. 
    """
    def __init__(self, n_sort: int, **kwargs):
        """
            Initialize Split Sort Layer Instance
        """
        super().__init__(**kwargs)

        if not isinstance(n_sort, int) or n_sort <= 0:
            raise ValueError("`n_sort` is required as positive integer")
        self.n_sort = n_sort
    

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass: sorts the first `n_sort` elements descendingly.

        Args:
        -----
        x:  Input tensor of shape `(batch, features)`.

        Returns:
        --------
            Tensor with first `n_sort` elements sorted in a descending
                order. \\
        """
        x = tf.convert_to_tensor(x)

        head = tf.sort(x[:, :self.n_sort], direction="DESCENDING")
        tail = x[:, self.n_sort:]

        return tf.concat([head, tail], axis=-1)
    

    def get_config(self):
        """
            Return layer configuration for serialization.
        """
        config = super().get_config()
        config.update({"n_sort": self.n_sort})

        return config




def extract_inputs(
    inputs: Union[
        Tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]],
        Dict[str, Union[np.ndarray, tf.Tensor]], tf.Tensor, 
        Sequence[Union[np.ndarray, tf.Tensor]]
]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Standardize various input formats into two TensorFlow tensors `(x, cond)`.

    formats:
    --------
        1. Tuple or list: `(x, cond)`
        2. Dict with keys `'x'` and `'cond'`
        3. Sequence of two tensors/arrays

    Args:
    ----- 
    inputs: Input data structure containing feature and condition tensors.

    Returns
    -------
    A tuple `(x, cond)` both converted to `tf.Tensor`.

    Raises
    ------
    ValueError
        If input format is unsupported or missing required keys.
    """
    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
        x, cond = inputs
        if isinstance(x, (list, tuple)):
            x = tf.concat([tf.convert_to_tensor(part) for part in x], axis=-1)
        else:
            x = tf.convert_to_tensor(x)
        return x, tf.convert_to_tensor(cond)
    
    elif isinstance(inputs, dict):
        if "x" not in inputs or "cond" not in inputs:
            raise ValueError("Dictionary is requred to contain `x` and `cond`")
        x = tf.convert_to_tensor(inputs["x"], dtype=tf.float32)
        cond = tf.convert_to_tensor(inputs["cond"], dtype=tf.float32)
        return x, cond
    
    raise ValueError(f"Unsupported input type: '{type(inputs)}'")



def set_initialization(
    model: tfk.Model, names: Sequence[str], kernel_init: float = 1.0,
    bias_init: float = 1.0, noise_type: str = "gaussian",
):
    """
    Reinitialize the weights of selected Dense layers in a model.

    Args:
    -----
        model:  Model instance containing layers to reinitialize.
        names:  List of layer names to reset.
        kernel_init:    Scaling factor for kernel initialization. Default is 1.0.
        bias_init:  Scaling factor for bias initialization. Default is 1.0.
        noise_type: Distribution type used to sample new weights.

    Notes
    -----
    - Only built `Dense` layers are reinitialized.
    - If a specified layer does not exist, a warning is raised.
    """
    for name in names:
        try: layer = model.get_layer(name)
        except Exception:
            warnings.warn(f"Layer {name} not found; skipping initialization")
            continue

        if not isinstance(layer, tfkl.Dense) or not layer.built:
            warnings.warn(f"Layer {name} is not a built Dense layer; " 
                          f"skipping initialization")
            continue

        input_dim, output_dim = layer.kernel.shape
        if noise_type.lower() == "gaussian":
            kernel = tf.random.normal(
                shape=(int(input_dim), int(output_dim)),
                mean=0.0, stddev=kernel_init / np.sqrt(float(input_dim)),
                dtype=tf.float32,
            )
            bias = tf.random.normal(
                shape=(int(output_dim),), mean=0.0, stddev=bias_init,
                dtype=tf.float32,
            )

        elif noise_type.lower() == "uniform":
            limit = kernel_init / np.sqrt(float(input_dim))
            kernel = tf.random.uniform((int(input_dim), int(output_dim)), 
                                        -limit, limit, dtype=tf.float32)
            bias = tf.random.uniform((int(output_dim),), 
                                     -bias_init, bias_init, 
                                     dtype=tf.float32)
        else: raise ValueError(f"Unsupported noise type: {noise}")

        try: layer.set_weights([kernel.numpy(), bias.numpy()])
        except Exception as e:
            warnings.warn(f"Could not set weights for layer {name}: {e}")
