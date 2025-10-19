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
