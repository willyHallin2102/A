"""
    debug/link.py
"""
from __future__ import annotations

import os, sys
import argparse

import numpy as np
from pathlib import Path
from typing import Any, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logs.logger import Logger, LogLevel
from database.loader import DataLoader, shuffle_and_split
from src.models.chanmod import ChannelModel



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Link Model in the ChannelModel on specific data"
    )
    parser.add_argument(
        "--city", type=str, default="beijing",
        help="Comma-separated list of cities or 'all'"
    )
    parser.add_argument(
        "--ratio", type=float, default=0.20,
        help="Validation split ratio"
    )
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging verbosity"
    )
    return parser.parse_args()




args = parse_args()

if args.city.lower() == "all":
    files = ["uav_beijing/train.csv",
             "uav_boston/train.csv",
             "uav_london/train.csv",
             "uav_moscow/train.csv",
             "uav_tokyo/train.csv"]
else:
    city_list = [c.strip().lower() for c in args.city.split(",")]
    supported = {"beijing", "boston", "london", "moscow", "tokyo"}
    invalid = set(city_list) - supported
    if invalid:
        sys.exit(1)
    files = [f"uav_{city}/train.csv" for city in city_list]


loader = DataLoader()
data = loader.load(files)
dtr, dts = shuffle_and_split(data=data, val_ratio=args.ratio)


model = ChannelModel(directory=args.city, seed=args.seed)
# model.link.build()
model.link.load()
history = model.link.fit(
    dtr=dtr, dts=dts, epochs=args.epochs, batch_size=args.batch,
    learning_rate=args.learning_rate,
) 

model.link.save()








### LOOK INTO HOW TO SAMPLE 
# def sample_path(self, dvec, rx_type, link_state=None, return_dict=False):
#         """
#         Generates random samples of the path data using the trained model

#         Parameters
#         ----------
#         dvec : (nlink,ndim) array
#             Vector from cell to UAV
#         rx_type : (nlink,) array of ints
#             Cell type.  One of terr_cell, aerial_cell
#         link_state:  (nlink,) array of {no_link, los_link, nlos_link}            
#             A value of `None` indicates that the link state should be
#             generated randomly from the link state predictor model
#         return_dict:  boolean, default False
#             If set, it will return a dictionary with all the values
#             Otherwise it will return a channel list
   
#         Returns
#         -------
#         chan_list:  (nlink,) list of MPChan object
#             List of random channels from the model.  Returned if
#             return_dict == False
#         data:  dictionary
#             Dictionary in the same format as the data.
#             Returned if return_dict==True
#         """
#         # Get dimensions
#         nlink = dvec.shape[0]

#         # Generate random link states if needed
#         # Use the link state predictor network
#         if link_state is None:
#             prob = self.link_predict(dvec, rx_type) 
#             cdf = np.cumsum(prob, axis=1)            
#             link_state = np.zeros(nlink)
#             u = np.random.uniform(0,1,nlink)
#             for i in range(cdf.shape[1]-1):
#                 I = np.where(u>cdf[:,i])[0]
#                 link_state[I] = i+1
                
#         # Find the indices where there are some link
#         # and where there is a LOS link
#         Ilink = np.where(link_state != LinkState.no_link)[0]
#         Ilos  = np.where(link_state == LinkState.los_link)[0]
#         los   = link_state == LinkState.los_link        
        
#         # Get the condition variables and random noise
#         U = self.transform_cond(dvec[Ilink], rx_type[Ilink], los[Ilink])
#         nlink1 = U.shape[0]
#         Z = np.random.normal(0,1,(nlink1,self.nlatent))
        
#         # Run through the sampling network
#         X = self.path_mod.sampler.predict([Z,U]) 
        
#         # Compute the inverse transform to get back the path loss
#         # and angle data
#         nlos_pl1, nlos_ang1 , nlos_dly1 = self.inv_transform_data(dvec[Ilink], X)
        
#         # Create arrays for the NLOS paths
#         nlos_pl  = np.tile(self.pl_max, (nlink,self.npaths_max)).astype(np.float32)
#         nlos_ang = np.zeros((nlink,self.npaths_max,AngleFormat.nangle), dtype=np.float32)
#         nlos_dly  = np.zeros((nlink,self.npaths_max), dtype=np.float32)
#         nlos_pl[Ilink]  = nlos_pl1
#         nlos_ang[Ilink] = nlos_ang1
#         nlos_dly[Ilink]  = nlos_dly1
        
#         # Compute the PL and angles for the LOS paths
#         los_pl1, los_ang1, los_dly1 = self.get_los_path(dvec[Ilos])
        
#         # Create arrays for the LOS paths
#         los_pl  = np.zeros((nlink,), dtype=np.float32)
#         los_ang = np.zeros((nlink,AngleFormat.nangle), dtype=np.float32)
#         los_dly  = np.zeros((nlink,), dtype=np.float32)
#         los_pl[Ilos]  = los_pl1
#         los_ang[Ilos] = los_ang1
#         los_dly[Ilos]  = los_dly1
        
#         # Store in a data dictionary
#         data = dict()
#         data['dvec'] = dvec
#         data['rx_type'] = rx_type
#         data['link_state'] = link_state
#         data['nlos_pl'] = nlos_pl
#         data['nlos_dly'] = nlos_dly
#         data['nlos_ang'] = nlos_ang
#         data['los_pl'] = los_pl
#         data['los_dly'] = los_dly
#         data['los_ang'] = los_ang
        
#         if return_dict:
#             return data
        
        
#         # Config
#         cfg = DataConfig()
#         cfg.fc = self.fc
#         cfg.npaths_max = self.npaths_max
#         cfg.pl_max = self.pl_max
        
#         # Create list of channels
#         chan_list, link_state = data_to_mpchan(data, cfg)
        
#         return chan_list, link_state