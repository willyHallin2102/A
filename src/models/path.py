"""
    src/models/path.py
    ------------------
    preprocessing the path loss, angles of arrival and departure, and delays 
    within the path object. Once the transformations been made, it proceed 
    by convert the NumPy arrays into TF Tensors before passing these to 
    the generating ai model.
"""
from __future__ import annotations
import pickle

import numpy as np
import tensorflow as tf
tfk = tf.keras

from pathlib import Path
from typing import Dict, Final, List, Tuple, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from logs.logger import Logger, LogLevel
from src.config.model import get_config
from src.config.data import AngleIndex, LinkState
from src.config.const import LIGHT_SPEED
from src.math.coords import cartesian_to_spherical, add_angles, sub_angles

from src.models.utilities.preproc import serialize_preproc, deserialize_preproc
from src.config.const import PREPROC_FN



class PathModel:
    """
    A model for learning and generating wireless multipath characteristics such as
    path loss, angle of arrival/departure, and delays.

    This object provides preprocessing, model building (e.g., assign the
    gen ai model), convert the dataset and pass it to the generative ai 
    model.
        - VAE
    """
    ANGLE_SCALE: Final[float]=180.0

    def __init__(self,
        directory: Union[str, Path], model_type: str, 
        rx_types: List[Union[str, int]], n_max_paths: int, max_path_loss: float,
        loglevel: LogLevel=LogLevel.INFO, to_disk: bool=False, **kwargs
    ):
        """ 
            Initialize Path Model Instance
        """
        # Initialize the logger
        self.loglevel = loglevel
        self.logger = Logger(
            "path-model", json_format=True, use_console=True,
            level=self.loglevel, to_disk=to_disk
        )

        # Create the directory
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        # Hyperparameters for the path-model
        self.model_type = model_type.lower()
        self.config = get_config(self.model_type)
        self.model: tfk.Model = None # Leave Null on construction.

        self.rx_types = list(rx_types)
        self.n_max_paths = int(n_max_paths)
        self.max_path_loss = float(max_path_loss)

        self._initialize_preprocessors()


    # ---------------========== I/O ==========--------------- #

    def save(self):
        """"""
        if self.model is None:
            raise ValueError("Model is not built, call `self.build()`.")
        
        self.model.save(self.directory)
        payload = {
            "path_loss_scaler"   : serialize_preproc(self.path_loss_scaler),
            "condition_scaler"  : serialize_preproc(self.condition_scaler),
            "rx_encoder"        : serialize_preproc(self.rx_encoder),
            "delay_scaler"      : float(self.delay_scale),
            "n_max_paths"       : int(self.n_max_paths),
            "max_path_loss"      : float(self.max_path_loss),
            "rx_types"          : list(self.rx_types)
        }

        with open(self.directory / PREPROC_FN, "wb") as fp:
            pickle.dump(payload, fp, pickle.HIGHEST_PROTOCOL)


    def load(self):
        """"""
        path = self.directory / PREPROC_FN
        if not path.exists():
            raise FileNotFoundError("The loaded model does not exist")
        
        with open(path, "rb") as fp:
            params = pickle.load(fp)
        
        # Deserialize preprocessor
        self.path_loss_scaler = deserialize_preproc(params["path_loss_scaler"])
        self.condition_scaler = deserialize_preproc(params["condition_scaler"])
        self.rx_encoder = deserialize_preproc(params["rx_encoder"])

        # Other parameters
        self.delay_scale = float(params.get("delay_scale", 1.0))
        self.n_max_paths = int(params.get("n_max_paths", self.n_max_paths))
        self.max_path_loss = float(params.get("max_path_loss", self.max_path_loss))
        self.rx_types = list(params.get("rx_types", self.rx_types))

        # Reload model
        self.build()
        self.model.load(self.directory)

    # ---------------========== Model Construction ==========--------------- #

    def build(self) -> None:
        """
        Construct the underlying generative model.
        Supported model types:
            - VAE (Variational autoencoder) beta - annealing
        
        Raises:
        -------
            ValueError: If an unsupported model type is being passed.
        """
        if self.model_type == "vae":
            # 4 + len(self.rx_types) - 1,  # after dropping last one-hot
            from src.models.generators.vae import Vae
            self.model = Vae(
                n_latent=self.config.n_latent,
                n_data=self.n_max_paths * (2 + AngleIndex.N_ANGLES),
                n_conditions=3 + max(len(self.rx_types), 1),
                encoder_layers=self.config.encoder_layers,
                decoder_layers=self.config.decoder_layers,
                min_variance=self.config.min_variance,
                dropout_rate=self.config.dropout_rate, beta=self.config.beta,
                beta_annealing_step=self.config.beta_annealing_step,
                kl_warmup_steps=self.config.kl_warmup_steps,
                init_kernel=self.config.init_kernel, init_bias=self.config.init_bias,
                n_sort=self.n_max_paths, level=self.loglevel
            )
        
        else:
            raise ValueError(f"Unsupported model type: '{self.model_type}'")
    

    # ---------------========== Model Fitting ==========--------------- #

    def fit(self,
        dtr: Dict[str, np.ndarray], dts: Dict[str, np.ndarray],
        epochs: int=100, batch_size: int=512, learning_rate: float=1e-4
    ) -> tfk.callbacks.History:
        """
        Train the generative path model on the given dataset.

        Args:
        -----
            dtr:    Training data dictionary containing multipath parameters.
            dts:    Validation dataset.
            epochs: Number of training epochs.
            batch_size: Batch size for the training loop.
            learning_rate:  Learning rate for the optimizer.

        Returns:
        --------
            Training history with metrics and loss curves.
        """
        # prepare the dataset for training and validation
        xtr = self._prepare_dataset(dtr, batch_size, True)
        xts = self._prepare_dataset(dts, batch_size, False)

        self.logger.info("Compiling model")
        self.model.compile(
            optimizer=tfk.optimizers.Adam(learning_rate=learning_rate,clipvalue=1.0),
            run_eagerly=True
        )

        history = self.model.fit(x=xtr, validation_data=xts, epochs=epochs)
        self.history = history

        return history
    

    # ---------------========== Data Preprocessing ==========--------------- #

    def _initialize_preprocessors(self):
        """
        Initialize scalers and encoders for data preprocessing.

        Components:
            - `path_loss_scaler`: MinMaxScaler for normalizing path loss.
            - `condition_scaler`: StandardScaler for condition variables.
            - `rx_encoder`: OneHotEncoder for receiver types.
            - `delay_scale`: Normalization constant for time delays.
        """
        self.path_loss_scaler = MinMaxScaler()
        self.condition_scaler = StandardScaler()
        self.rx_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.delay_scale = 1.0
    

    def _prepare_dataset(self,
        data: Dict[str, np.ndarray], batch_size: int, fit: bool=False
    ) -> tf.data.Dataset:
        """
        Convert the raw data contained as dictionaries of string representations
        columned by the NumPy array data into a more appropriate Tensorflow
        dataset. This method filters valid samples from the data, applies all 
        the transformations, and constructs a batched, prefetched dataset for
        training and validation.

        Args:
        -----
            data:   Input dictionary with multipath data arrays
            batch_size: Size pf each dataset size (# samples per epoch)
            fit:    Boolean value, `True` of preprocess scalers being fitted.
        
        Returns:
        --------
            Prepared dataset yielding (x_data, conditions) tuples ready to be
            processed by the generative model.
        """
        self.logger.info("Preparing Dataset")
        link_state = np.asarray(data["link_state"])
        valid_mask = link_state != LinkState.NO_LINK

        if not np.any(valid_mask):
            raise ValueError("No valid links found in the input data")
        
        idx = np.flatnonzero(valid_mask)
        dvec = np.asarray(data["dvec"][idx], dtype=np.float32)
        rx = np.asarray(data["rx_type"][idx])
        los = (link_state[idx] == LinkState.LOS).astype(np.float32)

        conditions = self._transform_conditions(dvec, rx, los, fit=fit)
        x_data = self._transform_data(
            dvec, np.asarray(data["nlos_pl"][idx]),
            np.asarray(data["nlos_ang"][idx]), 
            np.asarray(data["nlos_dly"][idx]), fit=fit
        )
        if conditions.shape[0] != x_data.shape[0]:
            raise ValueError(
                f"Conditional / path sample mismatch"
                f"{conditions.shape[0]} vs {x_data.shape[0]}"
            )
        dataset = tf.data.Dataset.from_tensor_slices((x_data, conditions))
        dataset =  dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

        return dataset 


    # ---------------========== Data Transformations ==========--------------- #

    def _transform_conditions(self,
        dvec: np.ndarray, rx_types: np.ndarray, los: np.ndarray, fit: bool=False
    ) -> np.ndarray:
        """
        Transform link conditions (distance, height difference, LOS flag,
        receiver type) into normalized condition vectors.

        Args:
        -----
            dvec:   3D displacement vectors between TX and RX.
            rx_types:   Receiver type IDs for one-hot encoding
            los:    Binary LOS/NLOS indicator.
            fit:    Whether to fit the condition scaler.
        
        Returns:
        --------
            Scaled condition matrix for conditioning model
        """
        d3d = np.maximum(np.linalg.norm(dvec, axis=1), 1.0)
        dh = dvec[:,2]

        base = np.stack([d3d, np.log10(d3d), dh, los], axis=1)

        if fit: rx_one = self.rx_encoder.fit_transform(rx_types.reshape(-1, 1))
        else: rx_one = self.rx_encoder.transform(rx_types.reshape(-1, 1))

        # Drop last dummy column... avoiding redundancy
        if rx_one.shape[1] > 1: rx_one = rx_one[:, :-1]

        conditions = np.concatenate((base, rx_one), axis=1)
        if fit: self.condition_scaler.fit_transform(conditions)
        else: self.condition_scaler.transform(conditions)

        return conditions.astype(np.float32)
    

    def _transform_data(self,
        dvec: np.ndarray, nlos_path_loss: np.ndarray,
        nlos_angles: np.ndarray, nlos_delays: np.ndarray, fit: bool=False
    ) -> np.ndarray:
        """
        Transform raw NLOS parameters into normalized model inputs. Combines 
        scaled path_loss, relative angles, and normalized delays. Transform 
        physical multipath parameters into normalized model.


        Args:
        -----
            dvec:   3D displacement vectors.
            nlos_path_loss:  Path loss values for each path.
            nlos_angles:    AOA/AOD angle matrix.
            nlos_delays:    Path delays values.
            fit:    Whether to fitting the scalers.
        
        Returns:
        --------
            Concatenated and normalized input array for the model, consistent
            of (`path-loss`, `angles`, `delays`).
        """
        path_loss = self._transform_path_loss(nlos_path_loss, fit=fit)
        angles = self._transform_angles(dvec, nlos_angles)
        delays = self._transform_delays(dvec, nlos_delays, fit=fit)

        return np.hstack([path_loss, angles, delays]).astype(np.float32, copy=False)


    def _inverse_transform_data(self,
        dvec: np.ndarray, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Invert normalized model outputs back to the original physical 
        quantities (path loss, angles, and delays).

        Args:
        -----
            dvec:   3D displacement vectors.
            x:  Normalized model output tensor
        
        Returns:
        --------
            (path loss, angles, delays) in their respective units.
        """
        nmp = self.n_max_paths
        n_angles = 4 * nmp
        path_loss = x[:, :nmp]
        angles = x[:, nmp:nmp + n_angles]
        delays = x[:, nmp + n_angles:]
        return (
            self._inverse_transform_path_loss(path_loss),
            self._inverse_transform_angles(dvec, angles),
            self._inverse_transform_delays(dvec, delays)
        )
    

    # ---------------========== Path Loss Transformations ==========--------------- #

    def _transform_path_loss(self,
        nlos_path_loss: np.ndarray, fit: bool=False
    ) -> np.ndarray:
        """
        Normalize NLOS path loss using MinMax scaling to get a 0 to 1 range 
        scale.

        Args:
        -----
            nlos_path_loss: Path loss values for 

        Returns
        -------
            Normalized path_loss values in [0, 1].
        """
        x0 = self.max_path_loss - nlos_path_loss[:, :self.n_max_paths]
        if fit: return self.path_loss_scaler.fit_transform(x0)
        return self.path_loss_scaler.transform(x0)


    def _inverse_transform_path_loss(self,
        path_loss: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """
        Reconstruct original path loss values from normalized data.

        Args:
        -----
            path_loss:  Scaled path loss within range [0, 1]
        
        Returns:
        --------
            Denormalized path loss values back into quantifiable metrics,
            linear scaled dB metric.
        """
        x0 = np.clip(path_loss, 0.0, 1.0)
        x0 = self.path_loss_scaler.inverse_transform(x0)
        x0 = np.fliplr(np.sort(x0, axis=-1))

        return self.max_path_loss - x0
    

    # ---------------========== Angular Transformations ==========--------------- #

    def _transform_angles(self, 
        dvec: np.ndarray, nlos_angles: np.ndarray
    ) -> np.ndarray:
        """
        Convert absolute AOA/AOD angles into relative angular offsets 
        with respect to LOS direction, normalized by ANGLE_SCALE.

        Args:
        -----
            dvec:   3D displacement vectors.
            nlos_angles:    NLOS angle matrix (AOA_phi, AOA_theta, 
                            AOD_phi, AOD_theta).

        Returns:
        --------
            Normalized relative angular offsets for each path.
        """
        _, los_aoa_phi, los_aoa_theta = cartesian_to_spherical(-dvec)
        _, los_aod_phi, los_aod_theta = cartesian_to_spherical(dvec)

        aoa_phi_rel, aoa_theta_rel = sub_angles(
            nlos_angles[..., AngleIndex.AOA_PHI],
            nlos_angles[..., AngleIndex.AOA_THETA],
            los_aoa_phi[:, None],
            los_aoa_theta[:, None],
        )
        aod_phi_rel, aod_theta_rel = sub_angles(
            nlos_angles[..., AngleIndex.AOD_PHI],
            nlos_angles[..., AngleIndex.AOD_THETA],
            los_aod_phi[:, None],
            los_aod_theta[:, None],
        )

        nmp = self.n_max_paths
        out = np.empty((dvec.shape[0], 4 * nmp), dtype=np.float32)
        out[:, 0:nmp] = aoa_phi_rel / self.ANGLE_SCALE
        out[:, nmp : 2 * nmp] = aoa_theta_rel / self.ANGLE_SCALE
        out[:, 2 * nmp : 3 * nmp] = aod_phi_rel / self.ANGLE_SCALE
        out[:, 3 * nmp : 4 * nmp] = aod_theta_rel / self.ANGLE_SCALE

        return out


    def _inverse_transform_angles(self, 
        dvec: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct absolute AOA/AOD angles from normalized relative values.

        Args:
        -----
            dvec:   3D displacement vectors.
            angles: Normalized angular offsets produced by the model.

        Returns:
        --------
            Denormalized AOA/AOD angles in degrees.
        """
        nmp = self.n_max_paths
        aoa_phi_rel     = angles[:, 0:nmp]          * self.ANGLE_SCALE
        aoa_theta_rel   = angles[:, nmp:2*nmp]      * self.ANGLE_SCALE
        aod_phi_rel     = angles[:, 2*nmp:3*nmp]    * self.ANGLE_SCALE
        aod_theta_rel   = angles[:, 3*nmp:4*nmp]    * self.ANGLE_SCALE

        _, los_aoa_phi, los_aoa_theta = cartesian_to_spherical(dvec)
        _, los_aod_phi, los_aod_theta = cartesian_to_spherical(-dvec)

        nlos_aoa_phi, nlos_aoa_theta = add_angles(
            aoa_phi_rel, aoa_theta_rel, los_aoa_phi[:, None], los_aoa_theta[:, None]
        )
        nlos_aod_phi, nlos_aod_theta = add_angles(
            aod_phi_rel, aod_theta_rel, los_aod_phi[:, None], los_aod_theta[:, None]
        )
        out = np.zeros((dvec.shape[0], nmp, AngleIndex.N_ANGLES))
        out[..., AngleIndex.AOA_PHI] = nlos_aoa_phi
        out[..., AngleIndex.AOA_THETA] = nlos_aoa_theta
        out[..., AngleIndex.AOD_PHI] = nlos_aod_phi
        out[..., AngleIndex.AOD_THETA] = nlos_aod_theta

        return out
    

    # ---------------========== Delays Transformations ==========--------------- #

    def _transform_delays(self,
        dvec: np.ndarray, nlos_delays: np.ndarray, fit: bool=False
    ) -> np.ndarray:
        """
        Normalize excess propagation delays relative to LOS delay.

        Args:
        -----
            dvec:   3D displacement vectors.
            nlos_delays:    Absolute delay values per path.
            fit:    Whether to compute a normalization constant (delay_scale).

        Returns:
        --------
            Normalized relative delays.
        """
        distance = np.linalg.norm(dvec, axis=1)
        los_delays = distance / LIGHT_SPEED
        relative = np.maximum(0.0, nlos_delays - los_delays[:, None])
        
        if fit: 
            self.delay_scale = np.mean(relative) or 1.0

        return (relative / self.delay_scale).astype(np.float32)


    def _inverse_transform_delays(self,
        dvec: np.ndarray, delays: np.ndarray
    ) -> np.ndarray:
        """
        Convert normalized delays back to absolute physical delay values.

        Args:
        -----
            dvec:   3D displacement vectors.
            delays: Normalized delay array.

        Returns:
        --------
            Absolute delay values (in seconds).
        """
        # Compute LOS delays
        distance = np.linalg.norm(dvec, axis=1)
        los_delays = distance / LIGHT_SPEED
        
        # return the computed absolute delays
        return delays * self.delay_scale + los_delays[:, None]
