"""
"""

import numpy as np
import tensorflow as tf
import random

from pathlib import Path
from typing import Union

from src.config.data import DataConfig
from src.models.link import LinkStatePredictor
from src.models.path import PathModel

from logs.logger import LogLevel



class ChannelModel:
    def __init__(self,
        config: DataConfig=DataConfig(), model_type: str="vae", 
        directory: Union[str, Path]="beijing", seed: int=42,
        loglevel: LogLevel=LogLevel.INFO
    ):
        """ Initialize Channel Model Instance """
        # Create the directory root for the model
        self.directory = Path(__file__).parent / "store" / directory
        self.directory.mkdir(parents=True, exist_ok=True)

        # Set global seed for reproducibility
        self.set_global_seed(seed)


        self.link = LinkStatePredictor(
            directory=self.directory/"link", rx_types=config.rx_types, 
            n_unit_links=config.n_unit_links, dropout_rate=config.dropout_rate,
            add_zero_los_frac=config.add_zero_los_frac, level=loglevel
        )

        self.path = PathModel(
            directory=self.directory/model_type.lower(), 
            model_type=model_type, rx_types=config.rx_types, 
            n_max_paths=config.n_max_paths, max_path_loss=config.max_path_loss,
            loglevel=loglevel
        )

    def load(self):
        self.link.load()
        self.path.load()
    


    @staticmethod
    def set_global_seed(seed: int=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        # try:
        #     tf.config.experimental.enable_op_determinism()
        # except Exception:
        #     pass
