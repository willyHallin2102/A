"""
    src/models/generators/vae.py
    ----------------------------
"""
from __future__ import annotations

import orjson
import tensorflow as tf
tfk, tfkl = tf.keras, tf.keras.layers

from pathlib import Path
from typing import List, Sequence, Tuple, Union

from logs.logger import Logger, LogLevel
from src.models.utilities.common import (
    extract_inputs, set_initialization,
    SplitSortLayer
)

from src.models.utilities.preproc import serialize_preproc, deserialize_preproc
from src.config.const import PREPROC_FN, WEIGHTS_FN, CONFIG_FN


class Reparametrize(tfkl.Layer):
    """
    Applies the **reparametrization trick** for sampling from the `latent gaussian
    distribution`.

    Given latent mean `µ` and some log-variance `log(σ²)` while generating noise 
    by generating noise into latent `z = µ + ε · exp[0.5 · log(σ²)]`, where the
    `ε ~ N(0, I)`.

    The `ε` allow the gradient to propagate through stochastic nodes, specifically
    backpropagation.
    """
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Performs the reparametrization, to refactor the stochastic latent space
        to a stochastic node factorization.

        Args:
        -----
            inputs: Given mean `µ` and log variance `log(σ²)` of latent
                    distributed.
        Returns:
        --------
            Sampled latent variable tensor with the same shape
            as the mean `µ`.
        
        Example:
        --------
            >>> reparam = Reparametrize()
            >>> mu = tf.zeros((16, 8))
            >>> logvar = tf.zeros_like(mu)
            >>> z = reparam((mu, logvar))
            >>> z.shape
            TensorShape([16, 8])
        """
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu), dtype=mu.dtype)
        return mu + eps * tf.eps(0.5 * logvar)



def reconstruction_loss(x: tf.Tensor, mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
    """
    Gaussian reconstruction loss with diagonal covariance measured. It computes 
    the negative `log-likelihood` of observed data under the reconstructed
    gaussian distribution.
        L_rec = 0.5 · Σ_j[ precision_j · (x_j - µ_j)² + log(σ²) ]
    Average across the batch
    
    Args:
    -----
        x:  Tensor of shape (`batch_size`, `n_features`), original data.
        mu: Tensor of shape, reconstruction mean.
        logvar: Tensor of same shape, reconstructed log-variance.
    
    Returns:
    --------
        Scalar tensor, mean reconstructed loss over a given batch.
    """
    logvar = tf.clip_by_value(logvar, -10.0, 10.0)
    kl = -0.5 * tf.reduce_sum(
        1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1
    )
    return tf.reduce_mean(weights * kl)



# ------------------------------------------------------- #
# ---------------========== VAE ==========--------------- #
# ------------------------------------------------------- #

class Vae(tfk.Model):
    """
    Variational Auto Encoder (VAE) implementation with a β-annealing, KL warmup,
    and conditional input parameters.

    Provides training/test steps compatible steps with `keras` with a full 
    serialize/deserialize `save()/load()` functionality.
    """
    def __init__(self,
        n_latent: int, n_data: int, n_conditions: int,
        encoder_layers: Tuple[int, ...], decoder_layers: Tuple[int, ...],
        min_variance: float=1e-4, dropout_rate: float=0.2, beta: float=0.5,
        beta_annealing_step: int=10_000, kl_warmup_steps: int=1000, 
        init_kernel: float=10.0, init_bias: float=10.0, n_sort: int=0,
        level: LogLevel=LogLevel.INFO, to_disk: bool=False, **kwargs
    ):
        """ Initialize Variational AutoEncoder Instance """
        # Calling parent `tfk.Model`
        super().__init__(name="vae", **kwargs)
        
        # Setting up the logger
        self.loglevel = level
        self.logger = Logger(
            "Vae", to_disk=to_disk, use_console=True, json_format=True,
            level=self.loglevel
        )

        # Hyperparameters
        self.n_latent = int(n_latent)
        self.n_data = int(n_data)
        self.n_conditions = int(n_conditions)
        self.min_variance = float(min_variance)
        self.dropout_rate = float(dropout_rate)
        self.n_sort = int(n_sort)

        self.encoder_layers = tuple(int(layer) for layer in encoder_layers)
        self.decoder_layers = tuple(int(layer) for layer in decoder_layers)

        self.init_kernel = float(init_kernel)
        self.init_bias = float(init_bias)

        # Schedulers initialized and setup
        self.beta = tf.Variable(float(beta), trainable=False, dtype=tf.float32)
        self.beta_annealing_step = int(beta_annealing_step)
        self.kl_warmup_steps = tf.constant(int(kl_warmup_steps), dtype=tf.float32)
        self.current_step = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        # Building the networks // Layers // Sub-Models
        self.logger.debug("Building encoder and decoder networks...")
        self.sampler = Reparametrize()

        self.encoder = self._build_encoder(
            self.encoder_layers, self.init_kernel, 
            self.init_bias, self.dropout_rate
        )
        
        self.decoder = self._build_decoder(
            self.decoder_layers, self.init_kernel, self.init_bias,
            self.dropout_rate, self.n_sort
        )
        self._initialize_metrics()

        self.logger.info("Model initialization completed")


    @property
    def metrics(self) -> List[tfk.metrics.Metric]:
        """ Return the metric list """
        return [
            self.total_loss_tracker, self.recon_loss_tracker, 
            self.kl_divergence_tracker, self.beta_tracker,
            self.kl_weight_tracker, self.mse_tracker,
            self.mae_tracker
        ]



    # ---------------========== Forwarding ==========--------------- #

    def call(self,
        inputs, training: bool=False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Performs a forward pass through the Vae network. Input data `x` is 
        encoded by the `self.encoder` network into a latent space representation
        `z`. The latent space representation is being sampled from the 
        `self.sampler`. It completes the forward pass by reconstruct 
        back to the initial input data, `x'` and measures the difference 
        between the reconstruction-loss in mean and log-variance.

        Args:
        -----
            inputs: Input tensor pair (`x`, `conditions`).
            training:   If `True`, applies dropout and updates the batchnorm
                        statistics to the layers.
        Returns:
        --------
            (`x_mu`, `x_logvar`, `z_mu`, `z_logvar`)
        """
        x, conditions = extract_inputs(inputs)
        z_mu, z_logvar = self.encoder([x, conditions], training=training)
        z = self.sampler([z_mu, z_logvar])
        x_mu, x_logvar = self.decoder([z, conditions], training=training)

        return x_mu, x_logvar, z_mu, z_logvar
    

    # ---------------========== Internal Metrics ==========--------------- #

    def _initialize_metrics(self):
        """
        Internal metrics that the model within the TensorFlow object
        keep track on during training and training. These provide a
        customizable and runtime feedback to view the progress of the 
        training for the model.
        """
        self.total_loss_tracker = tfk.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tfk.metrics.Mean(name="recon_loss")
        self.kl_divergence_tracker = tfk.metrics.Mean(name="kl_div")
        self.beta_tracker = tfk.metrics.Mean(name="beta")
        self.kl_weight_tracker = tfk.metrics.Mean(name="kl_weight")
        self.mse_tracker = tfk.metrics.MeanSquaredError(name="mse")
        self.mae_tracker = tfk.metrics.MeanAbsoluteError(name="mae")


    def _update_metrics(self,
        x: tf.Tensor, mu: tf.Tensor, recon_loss_val: tf.Tensor,
        kl_div_val: tf.Tensor, total_loss_val: tf.Tensor, weights: tf.Tensor
    ):
        self.total_loss_tracker.update_state(total_loss_val)
        self.recon_loss_tracker.update_state(recon_loss_val)
        self.kl_divergence_tracker.update_state(kl_div_val)
        self.beta_tracker.update_state(self.beta)
        self.kl_weight_tracker.update_state(weights)
        self.mse_tracker.update_state(x, mu)
        self.mae_tracker.update_state(x, mu)
    

    
    def _update_schedulers(self):
        """
        Increments internal training step counter. This method is an 
        internal process to keep the β-annealing and the KL weighting
        warmup on par with the assigned value `src.config.data.py`.
        """
        # Increment step counter
        self.current_step.assign_add(1.0)

        # Anneal beta if it is enabled
        if self.beta_annealing_step > 0:
            self.beta.assign(tf.minimum(1.0, self.beta + 1.0 / self.beta_annealing_step))
        
        # Potential USELESS does not use TF-Graphs
        tf.cond(tf.equal(tf.math.floormod(self.current_step, 1000.0), 0.0),
            lambda: tf.print(
                "Step", tf.cast(self.current_step, tf.int32),
                " | β = ", tf.round(self.beta * 1000) / 1000
            ), lambda: tf.np_op()
        )
    

    def _kl_weight(self) -> tf.Tensor:
        """
        Computes the current KL warmup weight, ramping linearly from 0 to 1 
        over the parameter `kl_warmup_steps`.

        Returns:
        --------
            Scalar within range [0, 1] controlling KL scaling.
        """
    

    # ---------------========== Training/Testing ==========--------------- #

    def train_step(self, inputs):
        """
        """
        with logger.time_block("train_step duration"):
            x, cond = extract_inputs(inputs)
            self._update_schedulers()
            kl_weight = self._kl_weight()

            with tf.GradientTape() as tape:
                x_mu, x_logvar, z_mu, z_logvar = self([x, cond], training=True)
                recon = reconstruction_loss(x, x_mu, x_logvar)
                kld = kl_divergence(z_mu, z_logvar, kl_weight)
                total = recon + self.beta * kld
            
            gradients = tape.gradient(total, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self._update_metrics(x, x_mu, recon, kld, total, kl_weight)

            if tf.math.reduce_any(tf.math.is_nan(total)):
                self.logger.error("NaN detected in loss...")
        
        return { metric.name: metric.result() for metric in self.metrics }
    

    def test_step(self, inputs):
        """
        """
        with self.logger.time_block("test_step duration"):
            x, cond = extract_inputs(inputs)
            kl_weight = self._kl_weight()
            x_mu, x_logvar, z_mu, z_logvar = self([x, cond], training=False)
            recon = reconstruction_loss(x, x_mu, x_logvar)
            kld = kl_divergence(z_mu, z_logvar, kl_weight)
            total = recon + self.beta * kld
            self._update_metrics(x, x_mu, recon, kld, total, kl_weight)
        
        return { metric.name: metric.result() for metric in self.metrics }


    # ---------------========== Encoder/Decoder ==========--------------- #

    def _build_encoder(self) -> tfk.Model:
        """
        """
        self.logger.debug(f"Building encoder with layers={self.encoder_layers}")
        x_in = tfkl.Input(shape=(self.n_data,), name="enc-x")
        conditions_in = tfkl.Input(shape=(n_conditions,), name="x-conditions")
        h = tfkl.Concatenate(name="enc-concatenate")([x_in, conditions_in])

        names: List[str] = []
        for idx, units in enumerate(self.encoder_layers):
            name = f"enc-hidden-{idx}"
            names.append(name)
            h = tfkl.Dense(units, activation="swish", name=name)(h)
            h = tfkl.BatchNormalization(name=f"{name}-batch")(h)
            if self.dropout_rate > 0.0:
                h = tfkl.Dropout(self.dropout_rate, name=f"{name}-{drop}")(h)
        
        z_mu = tfkl.Dense(self.n_latent, name="enc-mu")(h)
        z_logvar = tfkl.Dense(self.n_latent, name="enc-logvar")(h)
        encoder = tfk.Model(
            [x_in, conditions_in], [z_mu, z_logvar], name="encoder"
        )
        
        set_initialization(encoder, names, self.init_kernel, self.init_bias)
        self.logger.info(f"Encoder built: {encoder.count_params()} params")

        return encoder
    

    def _build_decoder(self) -> tfk.Model:
        """
        """
        self.logger.debug(f"Building encoder with layers = '{self.decoder_layers}'")
        x_in = tfkl.Input(shape=(self.n_data,), name="dec-x")
        conditions_in = tfkl.Input(shape=(self.n_conditions,), name="dec-conditions")
        h = tfkl.Concatenate(name="dec-concatenate")([x_in, conditions_in])

        names: List[str] = []
        for idx, units in enumerate(self.decoder_layers):
            name = f"dec-hidden-{idx}"
            names.append(name)
            
            h = tfkl.Dense(units, activation="swish", name=name)(h)
            h = tfkl.BatchNormalization(name=f"{name}-batch")(h)
            if self.dropout_rate > 0.0:
                h = tfkl.Dropout(self.dropout_rate, name=f"{name}-drop")(h)

        z_mu = tfkl.Dense(self.n_latent, name="dec-mu")(h)
        if self.n_sort > 0:
            x_mu = SplitSortLayer(self.n_sort, name="dec-mu-sort-slice")(x_mu)
        
        x_logvar = tfkl.Dense(self.n_data, name="dec-logvar")(h)
        x_logvar = tfkl.Lambda(
            lambda t: tf.math.log(self.min_variance + tf.nn.softplus(t)),
            name="dec-logvar-act"
        )(x_logvar)

        decoder = tfk.Model([z_in, conditions_in], [x_mu, x_logvar], name="decoder")
        set_initialization(decoder, names, self.init_kernel, self.init_bias)

        self.logger.info(f"Decoder built: '{decoder.count_params()}' params")
        return decoder
