"""
    tasks/plot_angle_distribution.py
    --------------------------------
"""
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from database.get_data import get_cities
from src.models.chanmod import ChannelModel
from logs.logger import LogLevel, get_loglevel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dog-shit"
    )
    parser.add_argument(
        "--city", type=str, default="beijing",
        help="Comma-separated list of cities or 'all'"
    )
    parser.add_argument(
        "--ratio", type=float, default=0.20,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--model-type", type=str, default="vae",
        choices=["vae"],
        help="Whichever model that is to be used to run the angle-distribution."
    )

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging verbosity"
    )
    return parser.parse_args()




def main():
    args = parse_args()
    dtr, dts = get_cities(cities=args.city, val_ratio=args.ratio)

    model = ChannelModel(
        directory=args.city, model_type=args.model_type, seed=args.seed,
        loglevel=get_loglevel(args.log_level)
    )
    model.load()



if __name__ == "__main__":
    main()
