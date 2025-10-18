"""
    src/models/link.py
    ------------------

"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
tfk, tfkl = tf.keras, tf.keras.layers

from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.data import LinkState
from logs.logger import Logger, LogLevel

AUTOTUNE = tf.data.AUTOTUNE



class LinkStatePredictor:
    """
    Object with the purpose of predicting the UAV-receiver
    link state from any given position. These states are
    (NLoS/LoS) given displacement vectors and the receiver
    types.
    """
    def __init__(self,
        rx_types: List[Union[str, int]], n_unit_links: Tuple[int, ...],
        n_dimensions: int=3, add_zero_los_frac: float=0.1, dropout_rate: float=0.2,
        level: LogLevel=LogLevel.INFO, directory: Union[str, Path]="link",
        to_disk: bool=False
    ):
        """ Initialize the Link State Predictor Instance """
        self.logger = Logger(
            "link-state-predictor", json_format=True, use_console=True,
            to_disk=to_disk, level=level
        )

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logger.debug("Root directory created")
        
        # Model Preprocessors
        self.model, self.link_scaler, self.rx_type_encoder = None, None, None

        # Parameters
        self.rx_types = list(rx_types)
        self.n_unit_links = tuple(n_unit_links)
        self.n_dimensions = int(n_dimensions)
        self.add_zero_los_frac = float(add_zero_los_frac)
        self.dropout_rate = float(dropout_rate)

        self.__version__ = 1
        self.history = None
    

    # ---------------========== Model Construction ==========--------------- #

    def build(self):
        """
        Builds the model, it uses a simplistic `Sequential` model using 
        keras framework for architecture of this structure.
        """
        self.logger.debug("Building the model...")
        layers: List[tfkl.Input(shape=(2 * len(self.rx_types),), name="input")]

        for idx, units in enumerate(self.n_unit_links):
            self.logger.debug(f"Adding dense layer {idx}: {units}")

            layers.append(tfkl.Dense(
                units, activation=None, kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-5), name=f"hidden-{idx}")
            )
            layers.append(tfkl.BatchNormalization())
            layers.append(tfkl.Activation("relu"))

            if self.dropout_rate > 0.0:
                layers.append(tfkl.Dropout(self.dropout_rate))
            
        layers.append(tfkl.Dense(
            LinkState.N_STATES, activation="softmax", name="output"
        ))
        self.model = tfk.models.Sequential(layers)
        self.logger.info("Model has been successfully built.")
    

    # ---------------========== Training Model ==========--------------- #

    def fit(self,
        dtr: Dict[str, np.ndarray], dts: Dict[str, np.ndarray],
        epochs: int = 50, batch_size: int = 512, learning_rate: float = 1e-4
    ) -> tfk.callbacks.History:
        """
        Train the model.
        """
        self.logger.info(
            f"Starting training for {epochs} epochs (batch={batch_size}, lr={learning_rate})"
        )

        with self.logger.time_block("fit() training duration"):
            xtr, ytr = self._prepare_arrays(dtr, fit=True)
            xts, yts = self._prepare_arrays(dts, fit=False)

            self.model.compile(
                optimizer=tfk.optimizers.Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            train_ds = tf.data.Dataset.from_tensor_slices((xtr, ytr)).batch(batch_size).prefetch(AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((xts, yts)).batch(batch_size).prefetch(AUTOTUNE)

            self.history = self.model.fit(
                train_ds, epochs=epochs, validation_data=val_ds, verbose=1
            )

        self.logger.info("Training completed.")
        return self.history


    # ---------------========== Predictions ==========--------------- #

    def predict(self, dvec: np.ndarray, rx_type: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Run model inference."""
        self.logger.debug(f"Running prediction on {len(dvec)} samples.")
        with self.logger.time_block("predict() duration"):
            x = self._transform_links(dvec, rx_type, fit=False)
            predictions = self.model.predict(x, batch_size=batch_size, verbose=0)
        self.logger.debug(f"Prediction complete. Shape: {predictions.shape}")
        return predictions


    # --------------------- Internal preprocessing --------------------- #

    def _prepare_arrays(self, 
        data: Dict[str, np.ndarray], fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature and label arrays for training or inference.
        """
        mode = "fit" if fit else "transform"
        self.logger.debug(f"Preparing arrays in mode: '{mode}'")

        dvec = np.asarray(data["dvec"], dtype=np.float32)
        rx_type = np.asarray(data["rx_type"])
        link_state = np.asarray(data["link_state"], dtype=np.int32)

        if fit:
            self.rx_type_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            self.link_scaler = StandardScaler()

        # Optionally augment with synthetic LOS-zero samples
        dvec, rx_type, link_state = self._add_los_zero(dvec, rx_type, link_state)

        # Transform features
        x = self._transform_links(dvec, rx_type, fit=fit)
        y = link_state

        self.logger.debug(f"Prepared arrays: X={x.shape}, Y={y.shape}")
        return x, y
    

    def _transform_links(self, 
        dvec: np.ndarray, rx_type: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """
        Transforms displacement vectors and receiver types into model 
        features.
        """
        dx = np.linalg.norm(dvec[:, :2], axis=1, keepdims=True)
        dz = dvec[:, 2:3]

        if self.rx_type_encoder is None:
            raise RuntimeError("Encoder not initialized. Call `_prepare_arrays(..., fit=True)` first.")

        rx_one = (
            self.rx_type_encoder.fit_transform(rx_type[:, None])
            if fit
            else self.rx_type_encoder.transform(rx_type[:, None])
        ).astype(np.float32)

        x = np.hstack([rx_one * dx, rx_one * dz])

        if self.link_scaler is None:
            raise RuntimeError("Scaler not initialized. Call `_prepare_arrays(..., fit=True)` first.")

        return self.link_scaler.fit_transform(x) if fit else self.link_scaler.transform(x)

    

    def _add_los_zero(self,
        dvec: np.ndarray, rx_type: np.ndarray, link_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Adds synthetic zero-LOS samples for data augmentation.
        """
        n_samples = len(dvec)
        n_add = int(n_samples * self.add_zero_los_frac)
        if n_add <= 0:
            self.logger.debug("No zero-LOS augmentation added.")
            return dvec, rx_type, link_state

        self.logger.debug(f"Adding {n_add} synthetic LOS-zero samples for augmentation.")
        idx = np.random.choice(n_samples, size=n_add, replace=True)

        dvec_i = np.zeros_like(dvec[idx])
        dvec_i[:, 2] = np.maximum(dvec[idx, 2], 0)

        rx_type_i = rx_type[idx]
        link_state_i = np.full(n_add, LinkState.LOS, dtype=link_state.dtype)

        # Concatenate efficiently
        dvec = np.concatenate([dvec, dvec_i], axis=0)
        rx_type = np.concatenate([rx_type, rx_type_i], axis=0)
        link_state = np.concatenate([link_state, link_state_i], axis=0)

        self.logger.debug(f"Data augmentation complete. New total samples: {len(dvec)}")
        return dvec, rx_type, link_state
